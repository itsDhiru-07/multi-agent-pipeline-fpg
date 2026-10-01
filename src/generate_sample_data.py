"""
Generates a synthetic dataset that mirrors the schema of the public
"Hotel Booking Demand" dataset (Antonio, Almeida & Nunes, 2019 - Kaggle:
jessemostipak/hotel-booking-demand), remapped for the FPG frontline-agent
upsell scenario.

Run this if you don't have a Kaggle account / haven't downloaded the real
CSV into data/raw/hotel_bookings.csv. Column names match the real dataset
exactly, so swapping in the real file later requires no code changes.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 12000

HOTELS = ["Resort Hotel", "City Hotel"]
MEAL_PLANS = ["BB", "HB", "FB", "SC", "Undefined"]
ROOM_TYPES = ["A", "B", "C", "D", "E", "F", "G"]
COUNTRIES = ["USA", "GBR", "DEU", "FRA", "ARE", "IND", "ESP", "ITA", "PRT", "BRA"]
MARKET_SEGMENTS = ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups"]
DEPOSIT_TYPES = ["No Deposit", "Refundable", "Non Refund"]
CUSTOMER_TYPES = ["Transient", "Contract", "Transient-Party", "Group"]

# Frontline agent pool (front-desk agent IDs) - mimics IN-Gauge agent roster
N_AGENTS = 45
AGENT_IDS = [f"AG-{1000 + i}" for i in range(N_AGENTS)]
# a slice of "no agent captured" to simulate real-world unmatched/missing agent records
AGENT_POOL_WITH_NULLS = AGENT_IDS + [None] * 8

PROPERTY_IDS = [f"PROP-{100 + i}" for i in range(12)]


def generate() -> pd.DataFrame:
    n = N_ROWS
    lead_time = RNG.integers(0, 400, n)
    arrival_year = RNG.choice([2023, 2024, 2025], n)
    arrival_month = RNG.integers(1, 13, n)
    stays_weekend = RNG.integers(0, 4, n)
    stays_week = RNG.integers(0, 10, n)
    adults = RNG.integers(1, 4, n)
    children = RNG.choice([0, 0, 0, 1, 2], n)
    is_repeated_guest = RNG.choice([0, 1], n, p=[0.88, 0.12])
    previous_cancellations = RNG.poisson(0.1, n)
    booking_changes = RNG.poisson(0.2, n)
    deposit_type = RNG.choice(DEPOSIT_TYPES, n, p=[0.75, 0.05, 0.20])
    days_in_waiting_list = RNG.choice([0] * 9 + [5], n)
    customer_type = RNG.choice(CUSTOMER_TYPES, n, p=[0.7, 0.05, 0.2, 0.05])

    reserved_room_type = RNG.choice(ROOM_TYPES, n, p=[0.35, 0.2, 0.15, 0.1, 0.1, 0.06, 0.04])
    # upgrade probability drives the upsell story: ~22% of stays get upgraded by the agent
    upgraded_mask = RNG.random(n) < 0.22
    assigned_room_type = reserved_room_type.copy()
    upgrade_choices = np.array(ROOM_TYPES)
    for i in np.where(upgraded_mask)[0]:
        cur_idx = ROOM_TYPES.index(reserved_room_type[i])
        upgrade_options = upgrade_choices[min(cur_idx + 1, len(ROOM_TYPES) - 1):]
        if len(upgrade_options) > 0:
            assigned_room_type[i] = RNG.choice(upgrade_options)

    base_adr = RNG.normal(110, 35, n).clip(30, 450)
    # upgraded stays and better meal plans carry higher ADR - the "incremental revenue" signal
    meal = RNG.choice(MEAL_PLANS, n, p=[0.55, 0.15, 0.05, 0.20, 0.05])
    meal_uplift = pd.Series(meal).map({"BB": 0, "HB": 12, "FB": 25, "SC": -5, "Undefined": 0}).to_numpy()
    upgrade_uplift = np.where(upgraded_mask, RNG.normal(28, 8, n).clip(5, None), 0)
    adr = (base_adr + meal_uplift + upgrade_uplift).round(2)

    required_car_parking_spaces = RNG.choice([0, 1, 2], n, p=[0.82, 0.16, 0.02])
    total_of_special_requests = RNG.poisson(0.6, n).clip(0, 5)

    is_canceled = RNG.choice([0, 1], n, p=[0.82, 0.18])
    reservation_status = np.where(
        is_canceled == 1, "Canceled",
        np.where(RNG.random(n) < 0.03, "No-Show", "Check-Out")
    )

    agent = RNG.choice(AGENT_POOL_WITH_NULLS, n)
    company = RNG.choice([None] * 9 + ["CORP-77"], n)
    hotel = RNG.choice(HOTELS, n, p=[0.4, 0.6])
    property_id = RNG.choice(PROPERTY_IDS, n)
    country = RNG.choice(COUNTRIES, n)
    market_segment = RNG.choice(MARKET_SEGMENTS, n)

    # synthesized guest satisfaction / CSAT score (1-5), inversely nudged by
    # cancellations and lead time, positively nudged by upgrades - stands in
    # for a real CSAT feed FPG would source from post-stay surveys
    base_csat = RNG.normal(4.1, 0.6, n)
    base_csat -= is_canceled * 0.6
    base_csat -= (lead_time > 180).astype(float) * 0.15
    base_csat += upgraded_mask.astype(float) * 0.35
    csat_score = base_csat.clip(1, 5).round(2)

    # inject a few deliberately dirty rows for the Data Quality agent to catch
    dirty_idx = RNG.choice(n, size=int(n * 0.015), replace=False)
    adr_dirty = adr.copy()
    adr_dirty[dirty_idx[: len(dirty_idx) // 2]] = -adr_dirty[dirty_idx[: len(dirty_idx) // 2]]
    duplicate_block = pd.DataFrame  # placeholder to keep flake happy

    df = pd.DataFrame({
        "hotel": hotel,
        "property_id": property_id,
        "is_canceled": is_canceled,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": arrival_month,
        "stays_in_weekend_nights": stays_weekend,
        "stays_in_week_nights": stays_week,
        "adults": adults,
        "children": children,
        "country": country,
        "market_segment": market_segment,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr_dirty,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "meal": meal,
        "reservation_status": reservation_status,
        "csat_score": csat_score,
    })

    # duplicate a handful of rows to give the Data Quality agent a dedup case
    dup_rows = df.sample(n=15, random_state=7)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    out_path = "data/raw/hotel_bookings.csv"
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
