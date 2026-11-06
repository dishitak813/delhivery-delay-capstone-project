import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage
scaler = StandardScaler().fit(X_train)
tr_s, te_s = scaler.transform(X_train), scaler.transform(X_test)
chk = pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
                    "value": pd.Series([len(X_train), len(X_val), len(X_test),
                                        round(float(np.abs(tr_s.mean(axis=0)).max()), 4),
                                        round(float(np.abs(te_s.mean(axis=0)).max()), 4)], dtype=object)})
chk.to_csv("outputs/part_b_checks.csv", index=False)


def scores(yt, yp):
    mse = mean_squared_error(yt, yp)
    return dict(MAE=mean_absolute_error(yt, yp), MSE=mse, RMSE=np.sqrt(mse), R2=r2_score(yt, yp))


# 3. baseline
rows = [dict(model="mean_baseline", split="val", **scores(y_val, np.full(len(y_val), y_train.mean())))]

# 4. tree depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
dd = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    dd.append(dict(max_depth="unlimited" if d is None else d,
                   train_rmse=np.sqrt(mean_squared_error(y_train, t.predict(X_train))),
                   val_rmse=np.sqrt(mean_squared_error(y_val, t.predict(X_val)))))
dd = pd.DataFrame(dd)
dd.round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)
best_i = int(dd["val_rmse"].values.argmin())          # first minimum = smaller depth on ties
best_depth = depths[best_i]

# 5. four candidates
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, m in candidates.items():
    m.fit(X_train, y_train)
    rows.append(dict(model=name, split="val", **scores(y_val, m.predict(X_val))))

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append(dict(learning_rate=lr, val_rmse=np.sqrt(mean_squared_error(y_val, p)), val_r2=r2_score(y_val, p)))
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. CV on the two best + final model
val_rmse = {r["model"]: r["RMSE"] for r in rows if r["model"] != "mean_baseline"}
ranked = sorted(val_rmse, key=val_rmse.get)
X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranked[:2]:
    s = cross_val_score(candidates[name], X_tv, y_tv, cv=kf)
    cv_rows.append(dict(model=name, mean_r2=s.mean(), std_r2=s.std()))
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final = ranked[0]
rows.append(dict(model=final, split="test", **scores(y_test, candidates[final].predict(X_test))))
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print("ranking:", ranked, "best depth:", best_depth)
