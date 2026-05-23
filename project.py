import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

from sklearn.linear_model import Ridge

from sklearn.metrics import accuracy_score

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("train_2.csv")

# Очистка колонок
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Переименование
df = df.rename(columns={
    "год": "year",
    "месяц": "month",
    "рто": "target_value"
})

# Сортировка
df = df.sort_values(
    ["new_id", "year", "month"]
).reset_index(drop=True)

# ==========================================================
# 2. TARGET
# ==========================================================

# delta
df["delta"] = (
    df.groupby("new_id")["target_value"]
    .diff()
)

# classification target
df["target_class"] = (
    df["delta"] > 0
).astype(int)

# regression target
df["target_reg"] = df["target_value"]

# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

# LAGS
for lag in [1,2,3,4,5,6]:
    df[f"lag_{lag}"] = (
        df.groupby("new_id")["target_value"]
        .shift(lag)
    )

# rolling
lag_cols = [f"lag_{i}" for i in [1,2,3]]

df["mean_3"] = df[lag_cols].mean(axis=1)
df["std_3"] = df[lag_cols].std(axis=1)

# trend
df["trend"] = (
    df["lag_1"] - df["lag_3"]
)

# ratios
df["ratio_1_2"] = (
    df["lag_1"] / (df["lag_2"] + 1)
)

df["ratio_1_3"] = (
    df["lag_1"] / (df["lag_3"] + 1)
)

# seasonality
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# ==========================================================
# 4. CATEGORY ENCODING
# ==========================================================

cat_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

for col in cat_cols:
    df[col] = pd.factorize(df[col])[0]

# ==========================================================
# 5. CLEAN
# ==========================================================

required_cols = (
    ["target_class"]
    +
    [f"lag_{i}" for i in [1,2,3,4,5,6]]
)

df_clean = df.dropna(
    subset=required_cols
).copy()

# ==========================================================
# 6. FEATURES
# ==========================================================

drop_cols = [
    "target_class",
    "target_reg",
    "target_value",
    "new_id",
    "delta"
]

features = [
    c for c in df_clean.columns
    if c not in drop_cols
]

# ==========================================================
# 7. VALIDATION SPLIT
# ==========================================================

val_mask = (
    (df_clean["year"] == 2025)
    &
    (df_clean["month"] == 2)
)

train_df = df_clean.loc[~val_mask].copy()
val_df = df_clean.loc[val_mask].copy()

X_train = train_df[features]
y_train_class = train_df["target_class"]
y_train_reg = train_df["target_reg"]

X_val = val_df[features]
y_val_class = val_df["target_class"]
y_val_reg = val_df["target_reg"]

# ==========================================================
# 8. BASE MODELS
# ==========================================================

print("="*50)
print("TRAINING BASE MODELS")
print("="*50)

# ----------------------------------------------------------
# Logistic Regression
# ----------------------------------------------------------

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train,
    y_train_class
)

log_val_prob = log_model.predict_proba(X_val)[:,1]

# ----------------------------------------------------------
# Decision Tree
# ----------------------------------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    random_state=42
)

tree_model.fit(
    X_train,
    y_train_class
)

tree_val_prob = tree_model.predict_proba(X_val)[:,1]

# ----------------------------------------------------------
# Random Forest
# ----------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train_class
)

rf_val_prob = rf_model.predict_proba(X_val)[:,1]

# ----------------------------------------------------------
# CatBoost
# ----------------------------------------------------------

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    verbose=False,
    random_seed=42
)

cat_model.fit(
    X_train,
    y_train_class,
    verbose=False
)

cat_val_prob = cat_model.predict_proba(X_val)[:,1]

# ==========================================================
# 9. STACKING FEATURES
# ==========================================================

stack_train = pd.DataFrame({
    "logistic": log_val_prob,
    "tree": tree_val_prob,
    "rf": rf_val_prob,
    "cat": cat_val_prob
})

# ==========================================================
# 10. META MODEL
# ==========================================================

print("="*50)
print("TRAINING META MODEL")
print("="*50)

meta_model = Ridge(alpha=1.0)

meta_model.fit(
    stack_train,
    np.log1p(y_val_reg)
)

meta_pred_log = meta_model.predict(stack_train)

meta_pred = np.expm1(meta_pred_log)

meta_pred = np.clip(
    meta_pred,
    0,
    None
)

# ==========================================================
# 11. VALIDATION SCORE
# ==========================================================

val_class_pred = (
    stack_train.mean(axis=1) > 0.5
).astype(int)

acc = accuracy_score(
    y_val_class,
    val_class_pred
)

print(f"STACKING ACCURACY: {acc:.4f}")

# ==========================================================
# 12. FINAL TRAIN
# ==========================================================

print("="*50)
print("FINAL TRAIN")
print("="*50)

# retrain all models on full data

X_full = df_clean[features]

# logistic
log_model.fit(
    X_full,
    df_clean["target_class"]
)

# tree
tree_model.fit(
    X_full,
    df_clean["target_class"]
)

# rf
rf_model.fit(
    X_full,
    df_clean["target_class"]
)

# catboost
cat_model.fit(
    X_full,
    df_clean["target_class"],
    verbose=False
)

# ==========================================================
# 13. FORECAST
# ==========================================================

latest = (
    df_clean
    .sort_values(
        ["new_id", "year", "month"]
    )
    .groupby("new_id")
    .tail(1)
    .copy()
)

X_test = latest[features]

# base predictions

log_test = log_model.predict_proba(X_test)[:,1]

tree_test = tree_model.predict_proba(X_test)[:,1]

rf_test = rf_model.predict_proba(X_test)[:,1]

cat_test = cat_model.predict_proba(X_test)[:,1]

# stacking dataframe

stack_test = pd.DataFrame({
    "logistic": log_test,
    "tree": tree_test,
    "rf": rf_test,
    "cat": cat_test
})

# meta prediction

meta_test_log = meta_model.predict(stack_test)

meta_test = np.expm1(meta_test_log)

# ==========================================================
# 14. FINAL FORECAST
# ==========================================================

last_rto = latest["lag_1"].values

forecast = (
    last_rto
    *
    (
        0.9
        +
        np.clip(meta_test, 0, 2) * 0.1
    )
)

forecast = np.clip(
    forecast,
    0,
    None
)

# ==========================================================
# 15. SUBMISSION
# ==========================================================

submission = pd.DataFrame({
    "new_id": latest["new_id"].astype(int),
    "rto": forecast.astype(float)
})

submission["rto"] = (
    submission["rto"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

submission["rto"] = submission["rto"].clip(
    lower=0,
    upper=submission["rto"].quantile(0.999)
)

submission = submission.sort_values(
    "new_id"
).reset_index(drop=True)

print(submission.head())
print(submission.shape)

submission.to_csv(
    "_submission.csv",
    index=False
)

print("="*50)
print("STACKING SUBMISSION SAVED")
print("="*50)
