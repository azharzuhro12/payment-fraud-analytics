-- 01. Summary of overall payment transactions and fraud distribution
-- name: query_transaction_summary
SELECT 
    COUNT(*) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_volume,
    SUM(is_fraud) AS total_fraud_cases,
    ROUND(AVG(is_fraud) * 100.0, 3) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_loss_volume,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) * 100.0 / SUM(amount), 3) AS fraud_loss_share_pct
FROM transactions;

-- 02. Fraud concentration across transaction types
-- name: query_fraud_by_type
SELECT 
    type,
    COUNT(*) AS total_tx,
    ROUND(SUM(amount), 2) AS total_volume,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100.0, 3) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY type
ORDER BY fraud_count DESC;

-- 03. Fraud distribution across EDC & terminal devices
-- name: query_fraud_by_terminal
SELECT 
    terminal_type,
    COUNT(*) AS total_tx,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100.0, 3) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_amount
FROM transactions
GROUP BY terminal_type
ORDER BY fraud_rate_pct DESC;

-- 04. Top high-risk merchants by fraud rate and fraud loss
-- name: query_high_risk_merchants
SELECT 
    name_dest AS merchant_id,
    merchant_category,
    location,
    COUNT(*) AS total_tx,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100.0, 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_amount
FROM transactions
GROUP BY name_dest, merchant_category, location
HAVING total_tx >= 20 AND fraud_count > 0
ORDER BY fraud_rate_pct DESC, total_fraud_amount DESC
LIMIT 20;

-- 05. Hourly temporal fraud risk analysis (24-hour cycle)
-- name: query_fraud_by_hour
SELECT 
    (step % 24) AS hour_of_day,
    COUNT(*) AS total_tx,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100.0, 3) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_tx_amount
FROM transactions
GROUP BY (step % 24)
ORDER BY hour_of_day;

-- 06. Account draining anomaly detection (balance reduction vs transaction amount)
-- name: query_balance_drain_analysis
SELECT 
    CASE 
        WHEN newbalance_orig = 0 AND oldbalance_orig > 0 THEN 'Account Emptied to Zero'
        WHEN newbalance_orig > 0 THEN 'Normal Remaining Balance'
        ELSE 'Zero Initial Balance'
    END AS balance_drain_category,
    COUNT(*) AS total_tx,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100.0, 2) AS fraud_rate_pct
FROM transactions
GROUP BY balance_drain_category;

