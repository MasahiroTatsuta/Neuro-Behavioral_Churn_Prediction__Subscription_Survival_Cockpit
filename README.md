# 🧠 Neuro-Behavioral Churn Prediction & Subscription Survival Cockpit

> デジタルテレメトリログからの行動心理プロキシ抽出、Optuna × XGBoost 生存時間最適化、および Tableau 連携リアルタイムアラートシステム

本プロジェクトは、一般的なデモグラフィック属性（年齢・性別など）に依存した既存の解約予測アプローチを脱却し、顧客のプロダクト内における **行動心理・認知特性（欲求遅延耐性の低さ、持続的注意の困難さなどの ADHD 的行動特性）** に着目した、エンドツーエンドの生存分析予測・MLOpsプロダクションパイプラインです。

---
# ファイル構成

```
neuro-behavioral-churn-prediction/
├── README.md                          # 最高傑作の分析報告書・解説書
├── .gitignore                         # 不要ファイルや大容量データ、秘密情報の除外設定
├── requirements.txt                   # 再現性を担保するためのPython依存パッケージリスト
│
├── sql/                               # [Data Engineering]
│   └── get_behavioral_features.sql     # BigQueryから行動ログ特徴量マートを抽出したSQL
│
├── src/                               # [Production / MLOps Pipeline]
│   └── survival_mlops_pipeline.py     # Optuna最適化とW&B連携を含む統合本番Pythonコード
│
├── assets/                            # [Visualization Assets for README]
│   ├── kaplan_meier_survival_curve.png # ノンパラ生存分析曲線（W&Bから出力）
│   ├── cox_hazard_ratios_plot.png     # コックスハザード比プロット（W&Bから出力）
│
└── data/                              # [Data Sandbox] (※Git管理からは基本的に除外)
    ├── raw/                           # Google BigQueryから抽出した直後の生データ
    │   └── bq特徴量.csv (Git管理除外)
    └── processed/                     # モデル予測結果付きのTableau用データ
        └── tableau_churn_alert_data.csv (Git LFS管理またはGit除外推奨)
```
---

---

# 🚀 1. プロジェクト概要とビジネス背景

サブスクリプション型ビジネスにおいて、解約（チャーン）の防止はLTV（顧客生涯価値）を最大化するための最重要課題です。

しかし、従来の「解約一歩手前の挙動」を捉える静的な分類モデル（解約する／しないの2値分類）には、実務上以下の重大な課題が存在しました。

## 📌 従来の解約予測における課題

### ① 「いつ解約するか」の時間軸の欠如

2値分類では解約確率しか算出できず、LTV計算に必要な「顧客がいつまで生き残るか」という生存期間（寿命）の評価ができません。

### ② 表面的な属性情報の限界

年齢や獲得チャネルといった静的属性は、獲得時に決定してしまい、入会後のプロダクト体験を通じた能動的な介入トリガーになり得ません。

### ③ 行動心理（認知バイアス）の無視

購入直後の熱量の暴走（初日燃え尽き）や、サイト内迷走による認知的過負荷が引き起こす「バイヤーズ・リモース（購入後の後悔行動）」といった顧客の内面的バイアスをデータ化できていませんでした。

本システムではこれらの課題を解決するため、

- BigQueryによる行動テレメトリログ特徴量生成
- 統計的因果推論（Cox）
- 非線形生存分析（XGBoost）
- 実験管理（Weights & Biases）
- 現場介入用BI（Tableau）

をシームレスに統合した実務的アーキテクチャを実装しました。

---

# 🔄 2. E2E（エンドツーエンド）パイプライン設計

```text
+------------------------------------------------------------+
| 1. Data Engineering Phase (BigQuery SQL)                   |
|    - 57,174名の時系列挙動ログから行動心理特徴量マート構築      |
+------------------------------------------------------------+
                             │
                             ▼ bq特徴量.csv
+------------------------------------------------------------+
| 2. Statistical Validation & Causal Inference Phase         |
|    - VIFによる多重共線性検証                               |
|    - Kaplan-Meier生存曲線可視化 & Log-rank検定             |
|    - L2 Ridge付きCox比例ハザードモデルによる因果推論         |
+------------------------------------------------------------+
                             │
                             ▼
+------------------------------------------------------------+
| 3. Machine Learning & MLOps Phase                          |
|    (XGBoost × Optuna × Weights & Biases)                   |
|    - Stratified 3-Fold CVによるC-index最大化               |
|    - Feature Importance自動追跡                            |
+------------------------------------------------------------+
                             │
                             ▼ tableau_churn_alert_data.csv
+------------------------------------------------------------+
| 4. Business Intelligence Phase (Tableau Public)            |
|    - 行動特性マッピング散布図                              |
|    - リアルタイム解約アラート名簿                          |
+------------------------------------------------------------+
```

---

# 🛠 3. 特徴量エンジニアリングと認知心理学的選定理由

本システムのために独自設計した特徴量マート（57,174レコード）の概要です。

| 特徴量 | 数理的定義 | 認知心理学的仮説 |
|----------|----------|----------|
| **first_day_burnout_ratio** | 登録初日の決済総額 ÷ 登録後30日間総決済額 | 欲求遅延耐性の低さ（衝動性） |
| **context_switch_density_mean** | 画面切替回数 ÷ 滞在時間 | 持続的注意の困難さ |
| **cart_add_velocity_mean** | 商品閲覧→カート投入までの平均秒数 | 衝動的意思決定 |
| **session_interval_volatility_30d** | セッション間隔の標準偏差 | 先延ばし傾向 |
| **cancel_return_rate_30d** | (キャンセル＋返品) ÷ 総注文数 | バイヤーズ・リモース |
| **midnight_purchase_ratio** | 深夜購入数 ÷ 総注文数 | セルフモニタリング能力低下 |

---

# 📉 4. 探索的データ解析（EDA）と生存分析

モデリング初期段階として、Kaplan-Meier法による生存曲線分析を実施しました。

## 📊 Kaplan-Meier 生存曲線分析

以下2群を比較。

- **first_day_burnout_ratio = 1.0**
  - 超衝動燃え尽き型
- **first_day_burnout_ratio < 1.0**
  - 計画利用型

生存確率

$$
S(t)=P(T>t)
$$

を比較した結果、

- 計画利用型は高い生存率を維持
- 初日燃え尽き型は登録初期に急激な離脱

が確認されました。

### Log-rank Test

$$
\chi^2 = \text{極めて高値}
$$

$$
p < 4.0\times10^{-308}
$$

統計的に極めて有意な差が確認されました。

---

# 🔮 5. 統計的因果推論：Cox比例ハザードモデル

## ① 多重共線性検証（VIF）

### VIF結果

| 特徴量 | VIF |
|----------|----------:|
| first_day_burnout_ratio | 15.76 |
| acquisition_channel_Search | 9.72 |
| age | 6.06 |
| cart_add_velocity_mean | 5.12 |
| context_switch_density_mean | 5.09 |

高い共線性を持つ特徴量が存在したため、

```python
penalizer = 0.1
```

によるL2リッジ正則化を適用しました。

---

## ② Coxモデル結果

Cox比例ハザードは

$$
\lambda(t|X) =
\lambda_0(t) \\
\times \exp\left(
\sum_i \beta_i X_i
\right)
$$

で表されます。

### 結果サマリー

| Feature | β | Hazard Ratio | z-value | p-value | 評価 |
|----------|----------:|----------:|----------:|----------:|----------|
| first_day_burnout_ratio | 0.47 | 1.61 | 24.54 | <0.005 | 🚨 極めて強力 |
| context_switch_density_mean | 0.18 | 1.20 | 14.33 | <0.005 | 🚨 危険因子 |
| cancel_return_rate_30d | 0.14 | 1.15 | 4.55 | <0.005 | 🚨 危険因子 |
| age | -0.00 | 1.00 | -0.34 | 0.74 | 有意差なし |
| gender_M | -0.02 | 0.98 | -1.92 | 0.06 | 有意差なし |

---

## 主要な知見

### 初日燃え尽き度

Hazard Ratio = **1.61**

登録初日にリソースを使い果たした顧客は、解約リスクが **61%上昇**。

### 画面遷移密度

Hazard Ratio = **1.20**

サイト内迷走度が高い顧客は、解約リスクが **20%上昇**。

---

# 🤖 6. 非線形予測の極大化

## XGBoost Survival Analysis

```python
objective = "survival:cox"
```

を採用。

### Stratified 3-Fold CV

解約イベント比率を維持しながら、

- Stratified K-Fold
- C-index最大化

を実施。

### Optunaベイズ最適化

10 Trialの自動探索を実行。

### 最終評価

```text
C-index = 0.8732
```

生存分析では

- 0.5：ランダム
- 0.7：実務利用可能

とされる中、

**0.87超の高精度予測性能を達成しました。**

---

# 🌐 7. MLOps実験管理（Weights & Biases）

すべての実験履歴をWeights & Biasesで管理し、トレーサビリティを確保しました。
* **🌐 Weights & Biases 開発ログ:** [W&B Experiment Tracking Dashboard（実験履歴を見る）](https://wandb.ai/tcumepapam3d2-personal-project/neuro-behavioral-churn-pipeline/workspace?nw=nwusertcumepapam3d2)

## 🧠 Feature Importance

| Rank | Feature | Importance |
|----------|----------|----------:|
| 1 | first_day_burnout_ratio | 65.28% |
| 2 | context_switch_density_mean | 10.81% |
| 3 | acquisition_channel_Facebook | 8.73% |
| 4 | cancel_return_rate_30d | 8.15% |
| 5 | session_interval_volatility_30d | 7.89% |
| 6 | cart_add_velocity_mean | 6.01% |

### 因果推論と予測モデルの一致

Coxモデルで最大ハザード比を示した

```text
first_day_burnout_ratio
```

が、

XGBoostでも圧倒的な重要度1位となりました。

---

# 🎨 8. Tableauダッシュボード設計

## 🏆 Dashboard Name

**Neuro-Behavioral Churn Prediction & Subscription Survival Cockpit**
* **[Tableauダッシュボード]** (https://public.tableau.com/views/Neuro-BehavioralChurnPredictionSubscriptionSurvivalCockpit/1?:language=ja-JP&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## 📐 主要コンポーネント

### ① 行動特性セグメンテーションマップ

- X軸：Cart Add Velocity Mean
- Y軸：Context Switch Density Mean
- 色：Alert Level

```text
🚨 CRITICAL
⚠️ WARNING
✅ STABLE
```

---

### ② リアルタイム解約アラートリスト

予測スコア75%以上の

```text
🚨 CRITICAL
```

ユーザーのみを抽出。

顧客対応チーム向けの優先アプローチ対象リストとして活用。

---

### ③ 流入別累積ハザード推移

時間経過に伴う流入チャネル別のリスク推移を可視化。

---

## 🛠 Tableau特有の課題解決

### 描画順序問題の解決

CRITICALドットが埋もれる問題に対し、

- Legend Order調整
- CRITICALを最前面描画

を実施。

---

### ノイズトリミング

異常値（999秒）による軸歪みを防ぐため、

```text
0 ～ 180秒
```

に表示範囲を固定。

---

### フィルターアクション

散布図の危険セグメント選択に応じて、

顧客リストがリアルタイム連動するインタラクションを実装。

---

# 🎯 9. 認知バイアスへの介入施策提案

## 🚨 ① 注意散漫 × 衝動セグメント

以下をリアルタイム検知。

- 画面遷移密度急上昇
- カート投入速度異常

その瞬間に

```text
「お探しの情報はこちらですか？」
```

などの支援UIを表示し、認知的過負荷を軽減。

---

## 💌 ② 初日燃え尽き型ユーザー

購入後

- 3日目
- 7日目

に価値教育コンテンツを配信。

### 例

- 商品活用事例
- 利用レポート
- ベストプラクティス紹介

これにより、

- 認知的不協和の緩和
- 保有効果の促進
- 継続利用率向上

を狙います。

---

# 🏁 Conclusion

本プロジェクトでは、

- 認知心理学
- 生存分析
- 因果推論
- 機械学習
- MLOps
- BIダッシュボード

を統合し、

**「誰が解約するか」だけでなく「いつ解約するか」まで予測可能な実務レベルの解約予測基盤**

を構築しました。

さらに、行動心理学的特徴量を通じて顧客の認知バイアスを可視化し、予測から介入までを一貫して実現するエンドツーエンドの意思決定支援システムとして実装しています。
