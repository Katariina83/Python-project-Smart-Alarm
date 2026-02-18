/* ============================================================
   SMART ALARM – DATA-DRIVEN BUSINESS ANALYST DEMO
   Author: Katariina Rikkonen
   Database: smart_alarm.db
   ============================================================ */


/* ============================================================
   01) DATA VALIDATION – Table Row Counts
   Purpose: Validate dataset integrity
   ============================================================ */

SELECT 'customers' AS table_name, COUNT(*) AS rows FROM customers
UNION ALL SELECT 'subscriptions', COUNT(*) FROM subscriptions
UNION ALL SELECT 'installations', COUNT(*) FROM installations
UNION ALL SELECT 'support_tickets', COUNT(*) FROM support_tickets
UNION ALL SELECT 'upsells', COUNT(*) FROM upsells
UNION ALL SELECT 'payments', COUNT(*) FROM payments;



/* ============================================================
   02) EXECUTIVE KPI SNAPSHOT
   Purpose: High-level business overview
   ============================================================ */

WITH churn_rate AS (
    SELECT ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_pct
    FROM subscriptions
),
avg_install_delay AS (
    SELECT ROUND(AVG(install_delay_days), 2) AS avg_delay
    FROM installations
),
ticket_customers AS (
    SELECT COUNT(DISTINCT customer_id) AS ticket_customers
    FROM support_tickets
),
upsell_customers AS (
    SELECT COUNT(DISTINCT customer_id) AS upsell_customers
    FROM upsells
),
total_customers AS (
    SELECT COUNT(*) AS total FROM customers
),
mrr AS (
    SELECT ROUND(SUM(amount), 2) AS total_revenue
    FROM payments
)

SELECT
    total,
    churn_pct,
    avg_delay,
    ROUND(100.0 * ticket_customers / total, 2) AS ticket_rate_pct,
    ROUND(100.0 * upsell_customers / total, 2) AS upsell_rate_pct,
    total_revenue
FROM churn_rate,
     avg_install_delay,
     ticket_customers,
     upsell_customers,
     total_customers,
     mrr;



/* ============================================================
   03) CHURN RATE BY INSTALLATION DELAY BUCKET
   Purpose: Identify operational churn drivers
   ============================================================ */

WITH delay_bucket AS (
    SELECT
        s.customer_id,
        s.churned,
        CASE
            WHEN i.install_delay_days <= 2 THEN '0–2 days'
            WHEN i.install_delay_days <= 5 THEN '3–5 days'
            ELSE '6+ days'
        END AS delay_group
    FROM subscriptions s
    JOIN installations i ON s.customer_id = i.customer_id
)

SELECT
    delay_group,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM delay_bucket
GROUP BY delay_group
ORDER BY churn_rate_pct DESC;



/* ============================================================
   04) CHURN RATE – TICKETS VS NO TICKETS
   Purpose: Evaluate impact of customer support issues
   ============================================================ */

WITH ticket_flag AS (
    SELECT
        c.customer_id,
        s.churned,
        CASE
            WHEN t.customer_id IS NULL THEN 0
            ELSE 1
        END AS has_ticket
    FROM customers c
    JOIN subscriptions s ON c.customer_id = s.customer_id
    LEFT JOIN (
        SELECT DISTINCT customer_id
        FROM support_tickets
    ) t ON c.customer_id = t.customer_id
)

SELECT
    has_ticket,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM ticket_flag
GROUP BY has_ticket;



/* ============================================================
   05) CHURN RATE – UPSELL VS NO UPSELL
   Purpose: Measure retention impact of product expansion
   ============================================================ */

WITH upsell_flag AS (
    SELECT
        c.customer_id,
        s.churned,
        CASE
            WHEN u.customer_id IS NULL THEN 0
            ELSE 1
        END AS has_upsell
    FROM customers c
    JOIN subscriptions s ON c.customer_id = s.customer_id
    LEFT JOIN (
        SELECT DISTINCT customer_id
        FROM upsells
    ) u ON c.customer_id = u.customer_id
)

SELECT
    has_upsell,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM upsell_flag
GROUP BY has_upsell;



/* ============================================================
   06) MONTHLY REVENUE TREND
   Purpose: Financial growth & revenue monitoring
   ============================================================ */

SELECT
    strftime('%Y-%m', payment_date) AS month,
    ROUND(SUM(amount), 2) AS monthly_revenue,
    COUNT(DISTINCT customer_id) AS paying_customers
FROM payments
GROUP BY month
ORDER BY month;



/* ============================================================
   07) CUSTOMER LIFETIME VALUE (CLV)
   Purpose: Identify high-value customers
   ============================================================ */

SELECT
    customer_id,
    ROUND(SUM(amount), 2) AS lifetime_value
FROM payments
GROUP BY customer_id
ORDER BY lifetime_value DESC
LIMIT 20;



/* ============================================================
   08) COMBINED CHURN DRIVER MODEL (MULTI-FACTOR)
   Purpose: Simulate advanced business analysis logic
   ============================================================ */

WITH customer_features AS (
    SELECT
        c.customer_id,
        s.churned,
        i.install_delay_days,
        CASE WHEN t.customer_id IS NULL THEN 0 ELSE 1 END AS has_ticket,
        CASE WHEN u.customer_id IS NULL THEN 0 ELSE 1 END AS has_upsell
    FROM customers c
    JOIN subscriptions s ON c.customer_id = s.customer_id
    JOIN installations i ON c.customer_id = i.customer_id
    LEFT JOIN (
        SELECT DISTINCT customer_id FROM support_tickets
    ) t ON c.customer_id = t.customer_id
    LEFT JOIN (
        SELECT DISTINCT customer_id FROM upsells
    ) u ON c.customer_id = u.customer_id
)

SELECT
    has_ticket,
    has_upsell,
    CASE
        WHEN install_delay_days <= 2 THEN '0–2'
        WHEN install_delay_days <= 5 THEN '3–5'
        ELSE '6+'
    END AS delay_bucket,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customer_features
GROUP BY has_ticket, has_upsell, delay_bucket
ORDER BY churn_rate_pct DESC;
