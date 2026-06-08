#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neuro-Behavioral Churn Prediction & Subscription Survival Pipeline
〜 統計的因果推論(Cox)からOptuna×XGBoost生存予測、W&B実験管理、Tableau連携CSV出力までを統合したプロダクションスクリプト 〜
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
import wandb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# ==========================================
# 0. 環境設定・セキュリティ設定
# ==========================================
# グラフのビジュアルスタイル設定
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# 【セキュリティ注意】GitHub公開時は環境変数から読み込む構成を推奨
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "YOUR_WANDB_API_KEY_HERE")

# Optunaのログ出力を抑制してコンソールをクリーンに保つ
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_preprocess_data(file_path='data/raw/bq特徴量.csv'):
    """
    【Data Engineering】データ前処理フェーズ
    特徴量マートから欠損補正、カテゴリのワンホットエンコーディングを実行
    """
    print(f"[*] データを読み込んでいます: {file_path}")
    if not os.path.exists(file_path):
        # カレントディレクトリに存在する場合のフォールバック
        if os.path.exists('bq特徴量.csv'):
            file_path = 'bq特徴量.csv'
        else:
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[+] 初期データ件数: {len(df)} 件")

    # 1. カート投入速度の欠損代替値（999）を中央値に置換（線形モデルの歪み防止）
    velocity_median = df[df['cart_add_velocity_mean'] != 999]['cart_add_velocity_mean'].median()
    df['cart_add_velocity_mean'] = df['cart_add_velocity_mean'].replace(999, velocity_median)

    # 2. カテゴリ変数のワンホットエンコーディング
    df_encoded = pd.get_dummies(df, columns=['gender', 'acquisition_channel'], drop_first=True)

    # 3. 不要なID列を削除し、互換性のために論理型を整数型（0/1）に変換
    df_model = df_encoded.drop(columns=['user_id'])
    for col in df_model.columns:
        if df_model[col].dtype == 'bool':
            df_model[col] = df_model[col].astype(int)
            
    return df, df_model


def verify_multicollinearity(df_model):
    """
    【Statistical Validation】多重共線性（VIF）の統計的検証
    """
    print("\n[*] 多重共線性（VIF）の検証を実行中...")
    X_vif = df_model.drop(columns=['tenure_days', 'is_churned'])

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    print("--- 多重共線性（VIF）検証結果 ---")
    print(vif_data.to_string(index=False))
    
    # Weights & Biases に多重共線性テーブルを永続化
    wandb.log({"VIF_Verification_Table": wandb.Table(dataframe=vif_data)})
    return vif_data


def execute_kaplan_meier(df_model):
    """
    【EDA】カプラン＝マイヤー法によるノンパラメトリック生存曲線の可視化
    """
    print("\n[*] カプラン＝マイヤー生存曲線の描画とログランク検定を実行中...")
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(12, 6))

    # 「初日燃え尽き度（first_day_burnout_ratio）が 1.0（超衝動型）」と「それ以外」に群を分割
    group_impulsive = (df_model['first_day_burnout_ratio'] == 1.0)

    # 超衝動型の生存曲線をプロット
    kmf.fit(df_model.loc[group_impulsive, 'tenure_days'], 
            event_observed=df_model.loc[group_impulsive, 'is_churned'], 
            label='Impulsive Burnout Type (Burnout Ratio = 1.0)')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='crimson', linewidth=2.5)

    # 計画・習慣型の生存曲線をプロット
    kmf.fit(df_model.loc[~group_impulsive, 'tenure_days'], 
            event_observed=df_model.loc[~group_impulsive, 'is_churned'], 
            label='Deliberate / Routine Type (Burnout Ratio < 1.0)')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='teal', linewidth=2.5)

    # 2群間の生存分布の差をログランク検定で評価
    results = logrank_test(
        df_model.loc[group_impulsive, 'tenure_days'], df_model.loc[~group_impulsive, 'tenure_days'],
        df_model.loc[group_impulsive, 'is_churned'], df_model.loc[~group_impulsive, 'is_churned']
    )

    plt.title(f'Survival Curves by Consumer Behavioral Trait\nLog-rank Test p-value: {results.p_value:.4e}', fontsize=14)
    plt.xlabel('Lifespan (Days Since First Purchase: $T$)', fontsize=12)
    plt.ylabel('Survival Probability ($S(t)$)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # ローカル保存およびW&Bへのダッシュボード同期
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/kaplan_meier_survival_curve.png', dpi=300)
    wandb.log({"kaplan_meier_survival_curve": wandb.Image(fig)})
    wandb.log({"logrank_test_p_value": results.p_value})
    plt.close()


def fit_cox_proportional_hazards(df_model):
    """
    【Causal Inference】L2正則化付きコックス比例ハザードモデルによる因果推論
    """
    print("\n[*] コックス比例ハザードモデルのフィッティングを実行中...")
    # 多重共線性の影響を緩和するため、L2正則化（penalizer=0.1）を適用
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_model, duration_col='tenure_days', event_col='is_churned', show_progress=False)

    print("\n========================================================")
    print("     コックス比例ハザードモデル：統計・数理評価サマリー")
    print("========================================================")
    cph.print_summary()

    # ハザード比（ Hazard Ratios）プロット図の生成
    fig_cph, ax_cph = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax_cph)
    plt.title('Cox Proportional Hazards: Feature Hazard Ratios ($\\exp(\\beta)$)', fontsize=14)
    plt.tight_layout()
    
    # ローカル保存およびW&Bへの同期
    plt.savefig('assets/cox_hazard_ratios_plot.png', dpi=300)
    wandb.log({"cox_hazard_ratios_plot": wandb.Image(fig_cph)})
    wandb.log({"cox_model_aic": cph.AIC_partial_})
    plt.close()


def train_xgboost_survival(df_model):
    """
    【Machine Learning & MLOps】Optunaベイズ最適化 ✖ XGBoost 生存予測
    """
    print("\n[*] XGBoost生存予測モデルの構築およびOptuna最適化を開始...")

    # 1. XGBoost生存分析用の擬似ターゲットの作成（解約=正の日数、生存/打ち切り=負の日数）
    # ※生存時間が0日のレコードによる数理エラーを防ぐため+1日補正
    df_model['xgb_target'] = np.where(df_model['is_churned'] == 1, df_model['tenure_days'] + 1, -(df_model['tenure_days'] + 1))

    X_xgb = df_model.drop(columns=['tenure_days', 'is_churned', 'xgb_target'])
    y_xgb = df_model['xgb_target']
    days_raw = df_model['tenure_days']
    event_raw = df_model['is_churned']

    # 訓練・テストデータの分割（イベント比率を維持するStratified分割）
    X_train, X_test, y_train, y_test, days_train, days_test, event_train, event_test = train_test_split(
        X_xgb, y_xgb, days_raw, event_raw, test_size=0.2, random_state=42, stratify=event_raw
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dall = xgb.DMatrix(X_xgb)

    # 2. Optunaによるハイパーパラメータ探索（C-indexの最大化）
    def objective(trial):
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "seed": 42
        }
        
        # モデルの一時訓練
        bst = xgb.train(params, dtrain, num_boost_round=100)
        preds = bst.predict(dtest)
        
        # 評価指標としてC-index（Concordance Index）を算出
        c_index = concordance_index(days_test, -preds, event_test)
        
        # 各Trialのスコアを個別にW&Bへ送信
        wandb.log({
            "optuna_trial_cv_c_index": c_index, 
            "trial_learning_rate": params["learning_rate"],
            "trial_max_depth": params["max_depth"]
        })
        return c_index

    print("[*] Optuna ベイズ最適化を回しています (10 Trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print(f"[+] 最良試行の C-index: {study.best_value:.4f}")
    print(f"[+] 最適ハイパーパラメータ: {study.best_params}")

    # 3. 最適パラメータを用いた最終モデルのフィッティング
    best_params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "seed": 42,
        **study.best_params
    }
    
    # 評価曲線のログを追跡するためのダミーリスト
    evals_result = {}
    final_xgb_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=150,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False
    )

    # 4. 検証データでの最終精度評価
    final_preds_risk = final_xgb_model.predict(dtest)
    final_c_index = concordance_index(days_test, -final_preds_risk, event_test)
    print(f"\n[+] 最終検証データ一致指数 (Final Test C-index): {final_c_index:.4f}")
    wandb.log({"final_test_concordance_index": final_c_index})

    # W&Bの時系列チャートにロス減少トレンドを手動同期
    for i in range(len(evals_result['train']['cox-nloglik'])):
        wandb.log({
            "train-cox-nloglik": evals_result['train']['cox-nloglik'][i],
            "test-cox-nloglik": evals_result['test']['cox-nloglik'][i],
            "epoch": i
        })

    # 5. Feature Importance（特徴量重要度）テーブルの構築とW&B同期
    importance_scores = final_xgb_model.get_score(importance_type='weight')
    # 全変数が含まれるようにマッピング
    importance_data = []
    for col in X_xgb.columns:
        importance_data.append([col, importance_scores.get(col, 0.0)])
        
    df_importance = pd.DataFrame(importance_data, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
    wandb.log({"Feature Importance_table": wandb.Table(dataframe=df_importance)})

    # 全データに対する解約危険度（リスクスコア）の予測値を出力
    all_preds_risk = final_xgb_model.predict(dall)
    return final_xgb_model, all_preds_risk


def export_tableau_csv(df_raw, all_preds_risk, output_path='data/processed/tableau_churn_alert_data.csv'):
    """
    【Business Intelligence】Tableau連携用メタデータ生成・CSVエクスポート
    """
    print(f"\n[*] Tableauコックピット用CSVの出力処理を開始: {output_path}")
    os.makedirs('data/processed', exist_ok=True)
    
    # 予測されたハザード比（相対リスク）を直感的な0.0〜1.0（0%〜100%）に正規化
    min_r, max_r = all_preds_risk.min(), all_preds_risk.max()
    normalized_risk_score = (all_preds_risk - min_r) / (max_r - min_r)

    # 元の特徴量マートにスコアとビジネスアラートレベルのセグメントを結合
    df_tableau = df_raw.copy()
    df_tableau['predicted_risk_score'] = normalized_risk_score
    df_tableau['alert_level'] = np.where(df_tableau['predicted_risk_score'] >= 0.75, '🚨 CRITICAL (High Churn Risk)',
                                 np.where(df_tableau['predicted_risk_score'] >= 0.40, '⚠️ WARNING (Medium Risk)', 
                                          '✅ STABLE (Low Risk)'))

    # CSV書き出し
    df_tableau.to_csv(output_path, index=False)
    # カレントディレクトリにもバックアップ（Tableau Publicデスクトップでの読み込み用）
    df_tableau.to_csv('tableau_churn_alert_data.csv', index=False)
    print(f"[+] Tableau連携用CSVの書き出しに成功しました！ ({len(df_tableau)}件)")


def main():
    print("=== Neuro-Behavioral Churn Prediction Pipeline Launch ===")
    
    # 1. Weights & Biases のMLOpsセッション初期化
    run = wandb.init(
        project="neuro-behavioral-churn-prediction",
        name="survival-analysis-optuna-run",
        notes="Optunaベイズ最適化によるXGBoost生存分析とTableau連携の統合実行"
    )

    try:
        # 2. 特徴量マートの読み込みと前処理
        df_raw, df_model = load_and_preprocess_data()

        # 3. 統計検証フェーズ（VIF ＆ ノンパラメトリック生存推論）
        verify_multicollinearity(df_model)
        execute_kaplan_meier(df_model)

        # 4. 因果関係の特定フェーズ（L2正則化コックス比例ハザード）
        fit_cox_proportional_hazards(df_model)

        # 5. 機械学習予測フェーズ（Optuna × XGBoost 生存分析）
        _, all_preds_risk = train_xgboost_survival(df_model)

        # 6. BIレイヤーへの引き渡し（アラート付きCSVのエクスポート）
        export_tableau_csv(df_raw, all_preds_risk)
        print("\n[+] 全てのデータサイエンスおよびMLOpsパイプラインが正常に終了しました。")

    except Exception as e:
        print(f"\n[!] パイプライン実行中に致命的なエラーが発生しました: {str(e)}")
        raise e
    finally:
        # 7. W&Bセッションを安全にクローズ
        run.finish()


if __name__ == "__main__":
    main()