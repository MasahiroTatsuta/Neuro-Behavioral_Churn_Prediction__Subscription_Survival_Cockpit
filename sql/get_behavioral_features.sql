WITH global_settings AS (
  -- 1. データセット全体の最終日時を取得（生存分析の右側打ち切り判定の基準）
  SELECT MAX(created_at) AS max_dataset_time FROM `bigquery-public-data.thelook_ecommerce.order_items`
),

user_core_lifecycle AS (
  -- 2. 顧客ごとの「初回購入（＝オンボーディング開始）」と「最終アクティビティ」を特定
  SELECT
    user_id,
    MIN(created_at) AS first_purchase_time,
    MAX(created_at) AS last_purchase_time
  FROM
    `bigquery-public-data.thelook_ecommerce.order_items`
  WHERE
    status NOT IN ('Cancelled', 'Returned')
  GROUP BY
    user_id
),

clickstream_raw AS (
  -- 3. 初回購入から【30日以内】の生の行動ログ（events）のみを抽出し、時間差計算の準備
  SELECT
    e.user_id,
    e.session_id,
    e.sequence_number,
    e.event_type,
    e.created_at,
    ucl.first_purchase_time,
    -- 次のイベントのタイプと発生時刻を同一セッション内で取得
    LEAD(e.event_type) OVER (PARTITION BY e.session_id ORDER BY e.sequence_number) AS next_event_type,
    TIMESTAMP_DIFF(
      LEAD(e.created_at) OVER (PARTITION BY e.session_id ORDER BY e.sequence_number),
      e.created_at,
      SECOND
    ) AS seconds_to_next
  FROM
    `bigquery-public-data.thelook_ecommerce.events` e
  JOIN
    user_core_lifecycle ucl ON e.user_id = ucl.user_id
  WHERE
    e.created_at <= TIMESTAMP_ADD(ucl.first_purchase_time, INTERVAL 30 DAY)
),

session_level_metrics AS (
  -- 4. セッション単位での「過密さ（注意散漫指標）」と「カート投入速度」の計算
  SELECT
    user_id,
    session_id,
    MIN(created_at) AS session_start_time,
    -- 画面遷移密度（context_switch_density）：1分あたりの画面遷移数
    SAFE_DIVIDE(
      COUNT(1),
      (TIMESTAMP_DIFF(MAX(created_at), MIN(created_at), SECOND) / 60.0) + 1.0
    ) AS context_switch_density,
    -- カート追加速度（cart_add_velocity_mean）：商品を見てからカートに入れるまでの平均秒数
    AVG(CASE WHEN event_type = 'product' AND next_event_type = 'cart' THEN seconds_to_next END) AS avg_cart_add_velocity
  FROM
    clickstream_raw
  GROUP BY
    user_id,
    session_id
),

session_intervals AS (
  -- 5. 【修正：重要】ウィンドウ関数（LAG）を一度ここで実行し、セッション間の時間差（時間単位）を算出
  SELECT
    user_id,
    context_switch_density,
    avg_cart_add_velocity,
    TIMESTAMP_DIFF(
      session_start_time,
      LAG(session_start_time) OVER (PARTITION BY user_id ORDER BY session_start_time),
      HOUR
    ) AS hours_since_last_session
  FROM
    session_level_metrics
),

user_behavior_30d_agg AS (
  -- 6. ユーザー単位への集約と、セッション間隔のボラティリティの算出（前段の計算結果を使うことでネストを回避）
  SELECT
    user_id,
    -- ADHDプロキシ：カート追加速度の平均
    ROUND(COALESCE(AVG(avg_cart_add_velocity), 999), 2) AS cart_add_velocity_mean,
    -- ADHDプロキシ：注意散漫度（画面遷移密度）の平均
    ROUND(AVG(context_switch_density), 2) AS context_switch_density_mean,
    -- ADHDプロキシ：セッション間隔の不規則さ（標準偏差）
    ROUND(COALESCE(STDDEV(hours_since_last_session), 0), 2) AS session_interval_volatility_30d
  FROM
    session_intervals
  GROUP BY
    user_id
),

purchase_metrics_30d AS (
  -- 7. 初期30日間の購買行動（深夜購入率、キャンセル・返品率など）を集計
  SELECT
    oi.user_id,
    -- 衝動性：深夜（0時〜4時）の購入比率
    SAFE_DIVIDE(
      COUNT(CASE WHEN EXTRACT(HOUR FROM oi.created_at) IN (0, 1, 2, 3, 4) THEN 1 END),
      COUNT(1)
    ) AS midnight_purchase_ratio,
    -- 衝動性・不注意：初期キャンセル・返品率
    SAFE_DIVIDE(
      COUNT(CASE WHEN oi.status IN ('Cancelled', 'Returned') THEN 1 END),
      COUNT(1)
    ) AS cancel_return_rate_30d,
    -- 認知バイアス：最初の24時間（初日）での消費金額の割合（燃え尽き度）
    SAFE_DIVIDE(
      SUM(CASE WHEN TIMESTAMP_DIFF(oi.created_at, ucl.first_purchase_time, HOUR) <= 24 THEN oi.sale_price END),
      SUM(oi.sale_price)
    ) AS first_day_burnout_ratio
  FROM
    `bigquery-public-data.thelook_ecommerce.order_items` oi
  JOIN
    user_core_lifecycle ucl ON oi.user_id = ucl.user_id
  WHERE
    oi.created_at <= TIMESTAMP_ADD(ucl.first_purchase_time, INTERVAL 30 DAY)
  GROUP BY
    oi.user_id
)

-- 8. 全てを結合し、生存分析ターゲット（Tenure / Churn）を含めた最終マートを出力
SELECT
  u.id AS user_id,
  u.age,
  u.gender,
  u.traffic_source AS acquisition_channel,
  
  -- 抽出した高度デジタルテレメトリ・プロキシ特徴量
  COALESCE(ub.cart_add_velocity_mean, 999) AS cart_add_velocity_mean,
  COALESCE(ub.context_switch_density_mean, 0) AS context_switch_density_mean,
  COALESCE(ub.session_interval_volatility_30d, 0) AS session_interval_volatility_30d,
  ROUND(COALESCE(pm.midnight_purchase_ratio, 0), 3) AS midnight_purchase_ratio,
  ROUND(COALESCE(pm.cancel_return_rate_30d, 0), 3) AS cancel_return_rate_30d,
  ROUND(COALESCE(pm.first_day_burnout_ratio, 0), 3) AS first_day_burnout_ratio,

  -- 【生存分析ターゲット】生存日数（Tenure_days）
  CASE
    WHEN DATE_DIFF(DATE(gs.max_dataset_time), DATE(ucl.last_purchase_time), DAY) >= 90 
      THEN DATE_DIFF(DATE(ucl.last_purchase_time), DATE(ucl.first_purchase_time), DAY)
    ELSE DATE_DIFF(DATE(gs.max_dataset_time), DATE(ucl.first_purchase_time), DAY)
  END AS tenure_days,

  -- 【生存分析ターゲット】最後の活動から90日以上経過＝解約（1）、それ以外＝生存（0）
  CASE
    WHEN DATE_DIFF(DATE(gs.max_dataset_time), DATE(ucl.last_purchase_time), DAY) >= 90 THEN 1
    ELSE 0
  END AS is_churned

FROM
  `bigquery-public-data.thelook_ecommerce.users` u
JOIN
  user_core_lifecycle ucl ON u.id = ucl.user_id
LEFT JOIN
  user_behavior_30d_agg ub ON u.id = ub.user_id
LEFT JOIN
  purchase_metrics_30d pm ON u.id = pm.user_id
CROSS JOIN
  global_settings gs
WHERE
  -- 観察期間が90日以上確保できている顧客のみ（フェアな生存追跡のため）
  DATE_DIFF(DATE(gs.max_dataset_time), DATE(ucl.first_purchase_time), DAY) >= 90;