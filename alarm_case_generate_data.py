# alarm_case_generate_data.py
# Generates a realistic subscription security service dataset for my BA portfolio case.
# This is a case of 5000 customers.
# Output: CSV files + SQLite database (smart_alarm.db)
#
# Run:
#   python alarm_case_generate_data.py
#
# Requirements:
#   pip install pandas numpy

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class Config:
    n_customers: int = 5000
    seed: int = 42

    # Observation window
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"

    # Contract / fees
    contract_lengths: Tuple[int, ...] = (24, 36)  # months
    base_fee_min: float = 24.90
    base_fee_max: float = 49.90

    # Installation
    sla_days: int = 7

    # Upsell
    upsell_adoption_rate: float = 0.18  # ~18% adopt at least one upsell

    # Support tickets
    base_ticket_rate_60d: float = 0.22  # probability of at least one ticket in first 60 days

    # Payments
    payment_failure_rate: float = 0.03  # random failed payments among charges


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def random_dates(rng: np.random.Generator, n: int, start: datetime, end: datetime) -> np.ndarray:
    """Random date between start and end, inclusive."""
    span_days = (end - start).days
    offsets = rng.integers(0, span_days + 1, size=n)
    return np.array([start + timedelta(days=int(o)) for o in offsets], dtype="datetime64[ns]")


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def pick_weighted(rng: np.random.Generator, choices, probs, size: int):
    return rng.choice(choices, size=size, p=probs)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    out_dir = Path("smart_alarm_demo") / "data"
    ensure_dir(out_dir)

    start = dt(cfg.start_date)
    end = dt(cfg.end_date)

    # ----------------------------
    # 1) Customers
    # ----------------------------
    customer_ids = np.arange(1, cfg.n_customers + 1)

    regions = ["Madrid", "Barcelona", "Valencia", "Andalucía", "País Vasco", "Galicia", "Canarias"]
    region_probs = [0.21, 0.19, 0.12, 0.20, 0.10, 0.10, 0.08]

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    age_probs = [0.08, 0.28, 0.26, 0.20, 0.18]

    channels = ["Web", "Phone", "Partner", "In-store", "Referral"]
    channel_probs = [0.38, 0.22, 0.14, 0.16, 0.10]

    signup_dates = random_dates(rng, cfg.n_customers, start, end)

    contract_length = pick_weighted(rng, list(cfg.contract_lengths), [0.62, 0.38], cfg.n_customers)

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": pd.to_datetime(signup_dates).date.astype(str),
        "region": pick_weighted(rng, regions, region_probs, cfg.n_customers),
        "age_group": pick_weighted(rng, age_groups, age_probs, cfg.n_customers),
        "acquisition_channel": pick_weighted(rng, channels, channel_probs, cfg.n_customers),
        "contract_length_months": contract_length,
    })

    # ----------------------------
    # 2) Subscriptions (1 per customer)
    # ----------------------------
    # Monthly fee: base influenced by region + channel
    region_fee_adj = {
        "Madrid": 2.5, "Barcelona": 2.0, "Valencia": 1.0,
        "Andalucía": 0.0, "País Vasco": 1.5, "Galicia": -0.5, "Canarias": 0.5
    }
    channel_fee_adj = {"Web": -0.5, "Phone": 0.0, "Partner": 0.8, "In-store": 0.3, "Referral": -1.0}

    base_fee = rng.uniform(cfg.base_fee_min, cfg.base_fee_max, size=cfg.n_customers)
    fee = base_fee.copy()
    fee += customers["region"].map(region_fee_adj).to_numpy()
    fee += customers["acquisition_channel"].map(channel_fee_adj).to_numpy()
    fee += rng.normal(0, 1.2, size=cfg.n_customers)  # noise
    fee = clamp(fee, 19.90, 69.90)
    fee = np.round(fee, 2)

    subscriptions = pd.DataFrame({
        "subscription_id": customer_ids,  # simple 1:1 id
        "customer_id": customer_ids,
        "start_date": customers["signup_date"],
        "monthly_fee": fee,
        "status": "active",  # updated after churn simulation
        "cancellation_date": pd.Series([None] * cfg.n_customers, dtype="object"),
    })

    # ----------------------------
    # 3) Installations (1 per customer)
    # ----------------------------
    # Scheduled 1–5 days after signup; completion depends on delay.
    signup_dt = pd.to_datetime(customers["signup_date"])
    scheduled = signup_dt + pd.to_timedelta(rng.integers(1, 6, size=cfg.n_customers), unit="D")

    # Delay: mixture (most ok, some long delays)
    # Base delay around 4 days, with a tail
    base_delay = rng.normal(loc=4.0, scale=2.0, size=cfg.n_customers)
    tail = rng.exponential(scale=4.0, size=cfg.n_customers)
    tail_mask = rng.random(cfg.n_customers) < 0.18
    delay_days = base_delay
    delay_days[tail_mask] = base_delay[tail_mask] + tail[tail_mask]

    delay_days = np.round(clamp(delay_days, 0, 30)).astype(int)
    completed = scheduled + pd.to_timedelta(delay_days, unit="D")

    # First-time success probability drops if delay high
    success_prob = clamp(0.95 - (delay_days / 60.0), 0.75, 0.97)
    success_flag = rng.random(cfg.n_customers) < success_prob

    installations = pd.DataFrame({
        "installation_id": customer_ids,
        "customer_id": customer_ids,
        "scheduled_date": scheduled.dt.date.astype(str),
        "completed_date": completed.dt.date.astype(str),
        "installation_delay_days": delay_days,
        "over_sla_flag": (delay_days > cfg.sla_days),
        "first_time_success_flag": success_flag,
    })

    # ----------------------------
    # 4) Upsells
    # ----------------------------
    upsell_types = ["Indoor Camera", "Outdoor Camera", "Extra Sensors", "Smoke Detector", "Premium Monitoring"]
    upsell_fee = {
        "Indoor Camera": 4.99,
        "Outdoor Camera": 6.99,
        "Extra Sensors": 3.49,
        "Smoke Detector": 2.99,
        "Premium Monitoring": 7.99
    }

    adopt_mask = rng.random(cfg.n_customers) < cfg.upsell_adoption_rate
    adopters = customers.loc[adopt_mask, ["customer_id", "signup_date"]].copy()
    n_adopters = len(adopters)

    # 1–2 upsells per adopter
    upsell_counts = rng.integers(1, 3, size=n_adopters)

    upsells_rows = []
    upsell_id = 1
    for idx, row in adopters.reset_index(drop=True).iterrows():
        cid = int(row["customer_id"])
        sdate = dt(row["signup_date"])
        k = int(upsell_counts[idx])
        chosen = rng.choice(upsell_types, size=k, replace=False)
        for t in chosen:
            # upsell date 10–180 days after signup
            udate = sdate + timedelta(days=int(rng.integers(10, 181)))
            upsells_rows.append({
                "upsell_id": upsell_id,
                "customer_id": cid,
                "upsell_type": t,
                "upsell_date": udate.date().isoformat(),
                "upsell_monthly_fee": upsell_fee[t]
            })
            upsell_id += 1

    upsells = pd.DataFrame(upsells_rows)

    # Aggregate upsell monthly fee per customer for churn & payment calcs
    upsell_fee_by_customer = upsells.groupby("customer_id")["upsell_monthly_fee"].sum() if not upsells.empty else pd.Series([], dtype=float)
    upsell_fee_arr = np.array([float(upsell_fee_by_customer.get(cid, 0.0)) for cid in customer_ids])

    has_upsell = upsell_fee_arr > 0

    # ----------------------------
    # 5) Support Tickets
    # ----------------------------
    # Ticket likelihood rises with delay and first-time failure.
    delay = installations["installation_delay_days"].to_numpy()
    ft_fail = (~installations["first_time_success_flag"].to_numpy()).astype(int)

    # Base probability of ticket in first 60 days, then add effects
    p_ticket = cfg.base_ticket_rate_60d + (delay / 80.0) + (0.18 * ft_fail)
    p_ticket = clamp(p_ticket, 0.05, 0.75)

    ticket_mask = rng.random(cfg.n_customers) < p_ticket
    ticket_customers = customers.loc[ticket_mask, ["customer_id", "signup_date"]].copy()
    n_ticket_customers = len(ticket_customers)

    issue_types = ["App/Login", "False Alarm", "Sensor Issue", "Installation Follow-up", "Billing", "Connectivity"]
    issue_probs = [0.16, 0.18, 0.22, 0.20, 0.10, 0.14]

    tickets_rows = []
    ticket_id = 1
    for idx, row in ticket_customers.reset_index(drop=True).iterrows():
        cid = int(row["customer_id"])
        sdate = dt(row["signup_date"])

        # 1–4 tickets, more if delay high
        # Compute a customer-specific expected ticket count
        d = int(installations.loc[installations["customer_id"] == cid, "installation_delay_days"].iloc[0])
        lam = 1.0 + (d / 12.0)
        tcount = int(clamp(rng.poisson(lam=lam), 1, 4))
        for _ in range(tcount):
            tdate = sdate + timedelta(days=int(rng.integers(0, 61)))
            itype = pick_weighted(rng, issue_types, issue_probs, 1)[0]
            resolution_hours = float(clamp(rng.normal(18, 10), 1, 96))
            resolved = rng.random() < 0.93
            tickets_rows.append({
                "ticket_id": ticket_id,
                "customer_id": cid,
                "created_date": tdate.date().isoformat(),
                "issue_type": itype,
                "resolved_flag": resolved,
                "resolution_time_hours": round(resolution_hours, 1),
            })
            ticket_id += 1

    support_tickets = pd.DataFrame(tickets_rows)

    # Tickets count in first 60 days for churn logic
    ticket_counts_60d = support_tickets.groupby("customer_id")["ticket_id"].count() if not support_tickets.empty else pd.Series([], dtype=int)
    tickets_60d_arr = np.array([int(ticket_counts_60d.get(cid, 0)) for cid in customer_ids])

    # ----------------------------
    # 6) Churn simulation (hybrid BA logic)
    # ----------------------------
    # Risk factors:
    # - delay > SLA increases risk
    # - more tickets in first 60 days increases risk
    # - no upsell increases risk slightly (lower engagement)
    # - payment failures also increase risk a bit (later)
    #
    # We'll build churn probability and then sample a cancellation date.

    over_sla = (delay > cfg.sla_days).astype(int)
    no_upsell = (~has_upsell).astype(int)

    # Normalize tickets
    t_norm = clamp(tickets_60d_arr / 3.0, 0, 2)  # 0..2
    d_norm = clamp(delay / 10.0, 0, 3)          # 0..3

    # Base churn propensity
    # Choose coefficients to get a realistic overall churn in the window.
    z = (
        -2.1
        + 0.75 * d_norm
        + 0.55 * t_norm
        + 0.35 * over_sla
        + 0.20 * no_upsell
        + rng.normal(0, 0.25, size=cfg.n_customers)
    )
    churn_prob = sigmoid(z)  # 0..1

    churn_flag = rng.random(cfg.n_customers) < churn_prob

    # Cancellation happens between day 30 and min(contract_length*30, obs_end-start)
    # We also bias earlier churn for higher risk
    cancellation_dates = [None] * cfg.n_customers
    status = np.array(["active"] * cfg.n_customers, dtype=object)

    for i, cid in enumerate(customer_ids):
        if not churn_flag[i]:
            continue

        sdate = dt(subscriptions.loc[i, "start_date"])
        cl_months = int(customers.loc[i, "contract_length_months"])

        max_end = min(end, sdate + timedelta(days=int(cl_months * 30)))
        if max_end <= sdate + timedelta(days=30):
            # Too short window
            continue

        # earlier churn for higher risk
        risk = float(churn_prob[i])
        min_days = 30
        max_days = (max_end - sdate).days

        # Sample days with a skew: higher risk -> smaller expected days
        # Use a beta distribution that skews towards 0 with higher risk.
        a = 1.2
        b = clamp(5.0 - 3.5 * risk, 1.2, 5.0)  # higher risk -> smaller b -> more mass near 0
        frac = rng.beta(a, b)
        days_after = int(min_days + frac * (max_days - min_days))

        cdate = sdate + timedelta(days=days_after)
        cancellation_dates[i] = cdate.date().isoformat()
        status[i] = "cancelled"

    subscriptions["status"] = status
    subscriptions["cancellation_date"] = cancellation_dates

    # ----------------------------
    # 7) Payments
    # ----------------------------
    # Monthly payments from start until cancellation_date (or end of observation window).
    payments_rows = []
    payment_id = 1

    # Total monthly fee includes upsell fees
    total_monthly_fee = subscriptions["monthly_fee"].to_numpy() + upsell_fee_arr
    total_monthly_fee = np.round(total_monthly_fee, 2)

    for i, cid in enumerate(customer_ids):
        sdate = dt(subscriptions.loc[i, "start_date"])
        cdate_str = subscriptions.loc[i, "cancellation_date"]
        active_until = dt(cdate_str) if isinstance(cdate_str, str) else end

        # Create charges on the same day-of-month as start_date (approx)
        # We'll simulate month steps as 30-day increments for simplicity.
        current = sdate
        while current <= active_until and current <= end:
            amount = float(total_monthly_fee[i])
            # occasional payment failure
            ok = rng.random() > cfg.payment_failure_rate
            status_pay = "paid" if ok else "failed"
            payments_rows.append({
                "payment_id": payment_id,
                "customer_id": int(cid),
                "payment_date": current.date().isoformat(),
                "amount": amount,
                "payment_status": status_pay
            })
            payment_id += 1
            current = current + timedelta(days=30)

    payments = pd.DataFrame(payments_rows)

    # ----------------------------
    # Save CSVs
    # ----------------------------
    customers.to_csv(out_dir / "customers.csv", index=False, encoding="utf-8-sig")
    subscriptions.to_csv(out_dir / "subscriptions.csv", index=False, encoding="utf-8-sig")
    installations.to_csv(out_dir / "installations.csv", index=False, encoding="utf-8-sig")
    support_tickets.to_csv(out_dir / "support_tickets.csv", index=False, encoding="utf-8-sig")
    upsells.to_csv(out_dir / "upsells.csv", index=False, encoding="utf-8-sig")
    payments.to_csv(out_dir / "payments.csv", index=False, encoding="utf-8-sig")

    # ----------------------------
    # Save SQLite DB (for SQL analysis)
    # ----------------------------
    db_path = out_dir / "smart_alarm.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        customers.to_sql("customers", conn, index=False)
        subscriptions.to_sql("subscriptions", conn, index=False)
        installations.to_sql("installations", conn, index=False)
        support_tickets.to_sql("support_tickets", conn, index=False)
        upsells.to_sql("upsells", conn, index=False)
        payments.to_sql("payments", conn, index=False)
    finally:
        conn.close()

    # ----------------------------
    # Quick summary prints (sanity checks)
    # ----------------------------
    churn_rate = (subscriptions["status"] == "cancelled").mean()
    avg_delay = installations["installation_delay_days"].mean()
    tickets_rate = (support_tickets["customer_id"].nunique() / cfg.n_customers) if not support_tickets.empty else 0.0
    upsell_rate = (upsells["customer_id"].nunique() / cfg.n_customers) if not upsells.empty else 0.0

    print("✅ Dataset generated:", out_dir.resolve())
    print(f"- customers: {len(customers):,}")
    print(f"- subscriptions: {len(subscriptions):,} | churn_rate: {churn_rate:.2%}")
    print(f"- installations: {len(installations):,} | avg_delay_days: {avg_delay:.2f}")
    print(f"- support_tickets: {len(support_tickets):,} | customers_with_tickets_60d: {tickets_rate:.2%}")
    print(f"- upsells: {len(upsells):,} | upsell_adopters: {upsell_rate:.2%}")
    print(f"- payments: {len(payments):,}")
    print(f"- sqlite db: {db_path.name}")


if __name__ == "__main__":
    main()
