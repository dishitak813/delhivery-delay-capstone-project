"""Shared constants and the two functions used by training and by the scoring script."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
MEDIAN_COLS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Clean a raw shipments table. Returns (clean_df, medians)."""
    df = raw.drop_duplicates().copy()                                   # 1. exact duplicates
    df["service_type"] = df["service_type"].str.strip().str.title()     # 2. spelling
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")          # 3. missing payment mode
    val = df["declared_value"].astype("string")                         # 4. rupee text -> number
    val = val.str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(val).astype(float)
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])           # 5. datetime
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan   # 6. scanner sentinel
    if medians is None:                                                 # 7. fill gaps with medians
        medians = {c: float(df[c].median()) for c in MEDIAN_COLS}
    else:
        medians = {c: float(medians[c]) for c in MEDIAN_COLS}
    for c in MEDIAN_COLS:
        df[c] = df[c].fillna(medians[c])
    return df.reset_index(drop=True), medians


def build_features(clean, hubs):
    """Attach hub columns and create model features. Never uses delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    out = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype(int)
    out["is_cod"] = (out["payment_mode"] == "COD").astype(int)
    out["is_weekend"] = (out["dispatch_time"].dt.dayofweek >= 5).astype(int)
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype(int)
    out["log_declared_value"] = np.log(out["declared_value_inr"])
    return out
