import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

dedup = raw.drop_duplicates()
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
assert len(table) == len(clean)
table.to_csv("data/modelling_table.csv", index=False)

q1, q3 = table["delivery_hours"].quantile(0.25), table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
dv = table["declared_value_inr"]
ldv = table["log_declared_value"]
svc_raw = dedup["service_type"]
metrics = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", dedup["distance_km"].isna().sum()),
    ("missing_parcel_weight_before", dedup["parcel_weight_kg"].isna().sum()),
    ("sentinel_weight_rows", (dedup["parcel_weight_kg"] > 100).sum()),
    ("missing_declared_value_before", dedup["declared_value"].isna().sum()),
    ("missing_payment_mode_before", dedup["payment_mode"].isna().sum()),
    ("service_type_spellings_before", svc_raw.nunique()),
    ("service_type_spellings_after", table["service_type"].nunique()),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("declared_value_mean_inr", dv.mean()),
    ("declared_value_median_inr", dv.median()),
    ("log_declared_value_mean", ldv.mean()),
    ("log_declared_value_median", ldv.median()),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", table["is_delayed"].sum()),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1), ("hours_q3", q3),
    ("hours_lower_bound", lo), ("hours_upper_bound", hi),
    ("hours_outlier_count", ((table["delivery_hours"] < lo) | (table["delivery_hours"] > hi)).sum()),
]
checks = pd.DataFrame(metrics, columns=["metric", "value"])
checks["value"] = pd.Series([int(round(float(v))) if float(v).is_integer() else round(float(v), 4) for v in checks["value"]], dtype=object)
checks.to_csv("outputs/part_a_checks.csv", index=False)

table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    g = g.reset_index().rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    rows.append(g)
pd.concat(rows).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

corr = table.drop(columns=["distance_band"]).select_dtypes(include="number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv", index=True)

top = corr["delivery_hours"].drop("delivery_hours")
print(top.reindex(top.abs().sort_values(ascending=False).index).head(3))
print(corr.loc["delivery_hours", "promised_hours"], corr.loc["delivery_hours", "distance_km"], corr.loc["is_delayed", "distance_km"])
