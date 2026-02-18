-- ================================
-- Smart Alarm demo: Analytics Views
-- ================================

-- 1) Customer base (one row per customer, joins + derived churn flag)
DROP VIEW IF EXISTS v_customer_base;

CREATE VIEW v_customer_base AS
SELECT
    c.customer_id,
    c.signup_date,
    c.region,
    c.age_group,
    c.acquisition_channel,
    c.contract_length_months,

    s.subscription_id,
    s.start_date AS subscription_start_date,
    s.monthly_fee,
    s.status AS subscription_status,
    s.cancellation_date,

    CASE WHEN s.cancellation_date IS NOT NULL THEN 1 ELSE 0 END AS is_churned,

    i.completed_date AS install_date,
    i.installation_delay_days AS install_delay_days,

    -- tickets last 60d (based on created_date)
    COALESCE(t.tickets_60d, 0) AS tickets_60d,
    COALESCE(t.avg_resolution_time_hours_60d, NULL) AS avg_resolution_time_hours_60d,

    -- upsell flag (any upsell ever)
    COALESCE(u.has_upsell, 0) AS has_upsell

FROM customers c
LEFT JOIN subscriptions s
    ON s.customer_id = c.customer_id
LEFT JOIN installations i
    ON i.customer_id = c.customer_id
LEFT JOIN (
    SELECT
        customer_id,
        COUNT(*) AS tickets_60d,
        AVG(resolution_time_hours) AS avg_resolution_time_hours_60d
    FROM support_tickets
    WHERE date(created_date) >= date('now', '-60 day')
    GROUP BY customer_id
) t
    ON t.customer_id = c.customer_id
LEFT JOIN (
    SELECT
        customer_id,
        1 AS has_upsell
    FROM upsells
    GROUP BY customer_id
) u
    ON u.customer_id = c.customer_id
;


-- 2) Payments enriched (payments + customer + subscription attributes)
-- NOTE: payments table has NO subscription_id in your schema.
DROP VIEW IF EXISTS v_payments_enriched;

CREATE VIEW v_payments_enriched AS
SELECT
    p.payment_id,
    p.customer_id,
    p.payment_date,
    p.amount,
    p.payment_status,

    c.region,
    c.age_group,
    c.acquisition_channel,
    c.contract_length_months,

    s.subscription_id,
    s.monthly_fee,
    s.status AS subscription_status,
    s.cancellation_date,
    CASE WHEN s.cancellation_date IS NOT NULL THEN 1 ELSE 0 END AS is_churned

FROM payments p
LEFT JOIN customers c
    ON c.customer_id = p.customer_id
LEFT JOIN subscriptions s
    ON s.customer_id = p.customer_id
;


-- 3) Churn drivers (features for churn analysis; one row per customer)
DROP VIEW IF EXISTS v_churn_drivers;

CREATE VIEW v_churn_drivers AS
SELECT
    customer_id,
    region,
    age_group,
    acquisition_channel,
    contract_length_months,

    install_delay_days,
    tickets_60d,
    avg_resolution_time_hours_60d,
    has_upsell,
    monthly_fee,
    is_churned

FROM v_customer_base
;


-- 4) Monthly KPIs (revenue + churn count by year-month)
DROP VIEW IF EXISTS v_monthly_kpis;

CREATE VIEW v_monthly_kpis AS
WITH pay AS (
    SELECT
        substr(payment_date, 1, 7) AS year_month,
        SUM(CASE WHEN payment_status = 'paid' THEN amount ELSE 0 END) AS revenue_paid,
        COUNT(*) AS payments_total,
        SUM(CASE WHEN payment_status = 'paid' THEN 1 ELSE 0 END) AS payments_paid
    FROM payments
    GROUP BY substr(payment_date, 1, 7)
),
ch AS (
    SELECT
        substr(cancellation_date, 1, 7) AS year_month,
        COUNT(*) AS churned_customers
    FROM subscriptions
    WHERE cancellation_date IS NOT NULL
    GROUP BY substr(cancellation_date, 1, 7)
)
SELECT
    pay.year_month,
    pay.revenue_paid,
    pay.payments_total,
    pay.payments_paid,
    COALESCE(ch.churned_customers, 0) AS churned_customers
FROM pay
LEFT JOIN ch
    ON ch.year_month = pay.year_month
;

SELECT *
FROM v_monthly_kpis;