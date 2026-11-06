import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(name, yt, yp, **extra):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return dict(**{"model" if name is not None else "x": name}, **extra, TN=tn, FP=fp, FN=fn, TP=tp,
                accuracy=accuracy_score(yt, yp), precision=precision_score(yt, yp, zero_division=0),
                recall=recall_score(yt, yp, zero_division=0), f1=f1_score(yt, yp, zero_division=0))


# 2. baseline + three classifiers
rows = [cm_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]
logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
rows.append(cm_row("logistic_regression", y_val, logit.predict(X_val)))
tree_f1 = {}
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    tree_f1[d] = (f1_score(y_val, t.predict(X_val)), t)
best_d = max(tree_f1, key=lambda d: (tree_f1[d][0], -d))
rows.append(cm_row(f"decision_tree_depth_{best_d}", y_val, tree_f1[best_d][1].predict(X_val)))
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append(cm_row("random_forest_default", y_val, rf.predict(X_val)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. feature signal
sel_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
print("noise importance share", fs.loc[fs.is_noise, "rf_importance"].sum())

sk = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    p.fit(X_train, y_train)
    sk.append(dict(k=k, val_f1=f1_score(y_val, p.predict(X_val))))
pd.DataFrame(sk).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning (k = 7, selector fitted on training rows only)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
dflt = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)


def clean_params(p):
    return {k: (None if v is None else int(v)) for k, v in p.items()}


cands = {
    "grid_search": (clean_params(gs.best_params_), f1_score(y_val, gs.best_estimator_.predict(Xva7)),
                    len(gs.cv_results_["params"]), 5, gs.best_score_),
    "randomized_search": (clean_params(rs.best_params_), f1_score(y_val, rs.best_estimator_.predict(Xva7)),
                          len(rs.cv_results_["params"]), 5, rs.best_score_),
    "default_forest_same_7_features": ({"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
                                       f1_score(y_val, dflt.predict(Xva7)), 1, 0, np.nan),
}
tune = pd.DataFrame([dict(search=n, candidates=c[2], cv_folds=c[3], total_fits=(c[2] * c[3] if c[3] else 1),
                          best_params=json.dumps(c[0]), best_cv_accuracy=c[4], val_f1=c[1]) for n, c in cands.items()])
tune.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1, ties -> default, grid, randomized
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
chosen = max(order, key=lambda n: (round(cands[n][1], 4), -order.index(n)))
params = cands[chosen][0]
final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, **{k: v for k, v in params.items()}))]).fit(X_train, y_train)

# 6. threshold
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pv = final.predict_proba(X_val)[:, 1]
th_rows = []
for t in thresholds:
    r = cm_row("t", y_val, (pv >= t).astype(int))
    r.pop("model")
    th_rows.append(dict(threshold=t, **r))
th = pd.DataFrame(th_rows)
th.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
ok = th[th["recall"] >= 0.80]
alert = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation
pt = final.predict_proba(X_test)[:, 1]
r = cm_row("t", y_test, (pt >= alert).astype(int))
r.pop("model")
pd.DataFrame([dict(threshold=alert, **r)]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([dict(selector_k=K, n_estimators=params["n_estimators"],
                   max_depth="unlimited" if params["max_depth"] is None else params["max_depth"],
                   min_samples_leaf=params["min_samples_leaf"], threshold=alert, chosen_from=chosen)]
             ).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen", chosen, params, "threshold", alert)
