import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Ridge

from catboost import CatBoostClassifier

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("train_2.csv")

# CLEAN COLUMNS
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

df = df.rename(columns={
    "год": "year",
    "месяц": "month",
    "рто": "target_value"
})

# SORT
df = df.sort_values(
    ["new_id", "year", "month"]
).reset_index(drop=True)

# ==========================================================
# 2. TARGET ENGINEERING
# ==========================================================

# TARGET RATIO
df["target_ratio"] = (
    df.groupby("new_id")["target_value"]
    .shift(-1)
    /
    (df["target_value"] + 1)
)

# MULTICLASS TARGET
df["target_class"] = pd.cut(
    df["target_ratio"],
    bins=[
        -np.inf,
        0.90,
        0.98,
        1.02,
        1.10,
        np.inf
    ],
    labels=[0,1,2,3,4]
)

# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

# LAGS
for lag in [1,2,3,4,5,6,9,12]:

    df[f"lag_{lag}"] = (
        df.groupby("new_id")["target_value"]
        .shift(lag)
    )

# ROLLING FEATURES

df["mean_3"] = (
    df[[f"lag_{i}" for i in [1,2,3]]]
    .mean(axis=1)
)

df["mean_6"] = (
    df[[f"lag_{i}" for i in [1,2,3,4,5,6]]]
    .mean(axis=1)
)

df["std_3"] = (
    df[[f"lag_{i}" for i in [1,2,3]]]
    .std(axis=1)
)

# TREND

df["trend_1_3"] = (
    df["lag_1"] - df["lag_3"]
)

df["trend_1_6"] = (
    df["lag_1"] - df["lag_6"]
)

# RATIOS

df["ratio_1_2"] = (
    df["lag_1"] / (df["lag_2"] + 1)
)

df["ratio_1_3"] = (
    df["lag_1"] / (df["lag_3"] + 1)
)

df["ratio_1_6"] = (
    df["lag_1"] / (df["lag_6"] + 1)
)

# MOMENTUM

df["momentum"] = (
    df["lag_1"]
    /
    (df["mean_3"] + 1)
)

# VOLATILITY

df["volatility"] = (
    df["std_3"]
    /
    (df["mean_3"] + 1)
)

# SEASONALITY

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

for col in numeric_cols:

    df[col] = df[col].fillna(
        df[col].median()
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

required_cols = [
    "target_class",
    "target_ratio"
]

required_cols += [
    f"lag_{i}"
    for i in [1,2,3,4,5,6]
]

df_clean = df.dropna(
    subset=required_cols
).copy()

df_clean["target_class"] = (
    df_clean["target_class"]
    .astype(int)
)

# ==========================================================
# 6. FEATURES
# ==========================================================

drop_cols = [
    "target_value",
    "target_ratio",
    "target_class",
    "new_id"
]

features = [
    c for c in df_clean.columns
    if c not in drop_cols
]

# ==========================================================
# 7. VALIDATION
# ==========================================================

val_mask = (
    (df_clean["year"] == 2025)
    &
    (df_clean["month"] == 1)
)

train_df = df_clean.loc[
    ~val_mask
].copy()

val_df = df_clean.loc[
    val_mask
].copy()

X_train = train_df[features]
y_train = train_df["target_class"]

X_val = val_df[features]
y_val = val_df["target_class"]

# ==========================================================
# 8. BASE MODELS
# ==========================================================

print("="*60)
print("TRAINING BASE MODELS")
print("="*60)

models = {

    "logistic": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "rf": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "extra": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=10,
        random_state=42
    ),

    "cat": CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        loss_function="MultiClass",
        verbose=False,
        random_seed=42
    )
}

# ==========================================================
# 9. TRAIN BASE MODELS
# ==========================================================

stack_train = pd.DataFrame()

for name, model in models.items():

    print(f"Training {name}")

    model.fit(
        X_train,
        y_train
    )

    # probabilities
    probs = model.predict_proba(X_val)

    # use all classes as meta-features
    for i in range(probs.shape[1]):

        stack_train[f"{name}_class_{i}"] = probs[:, i]

# ==========================================================
# 10. META FEATURES
# ==========================================================

stack_train["mean_prob"] = (
    stack_train.mean(axis=1)
)

stack_train["std_prob"] = (
    stack_train.std(axis=1)
)

stack_train["max_prob"] = (
    stack_train.max(axis=1)
)

# ==========================================================
# 11. META MODEL
# ==========================================================

print("="*60)
print("TRAINING META MODEL")
print("="*60)

meta_model = Ridge(
    alpha=3.0,
    random_state=42
)

# target = future ratio
meta_target = (
    val_df["target_ratio"]
    .clip(0.5, 1.5)
)

meta_model.fit(
    stack_train,
    meta_target
)

# validation prediction
meta_pred = meta_model.predict(
    stack_train
)

meta_pred = np.clip(
    meta_pred,
    0.7,
    1.3
)

# FINAL FORECAST
val_forecast = (
    val_df["lag_1"].values
    *
    meta_pred
)

# DIRECTION SCORE
direction_pred = (
    meta_pred > 1
).astype(int)

direction_true = (
    meta_target > 1
).astype(int)

acc = accuracy_score(
    direction_true,
    direction_pred
)

print("="*60)
print(f"DIRECTION ACCURACY: {acc:.4f}")
print("="*60)

# ==========================================================
# 12. FINAL TRAIN
# ==========================================================

print("="*60)
print("FINAL TRAIN")
print("="*60)

X_full = df_clean[features]
y_full = df_clean["target_class"]

for name, model in models.items():

    print(f"Final training {name}")

    model.fit(
        X_full,
        y_full
    )

# ==========================================================
# 13. TEST FORECAST
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

stack_test = pd.DataFrame()

for name, model in models.items():

    probs = model.predict_proba(X_test)

    for i in range(probs.shape[1]):

        stack_test[f"{name}_class_{i}"] = probs[:, i]

# meta features

stack_test["mean_prob"] = (
    stack_test.mean(axis=1)
)

stack_test["std_prob"] = (
    stack_test.std(axis=1)
)

stack_test["max_prob"] = (
    stack_test.max(axis=1)
)

# ==========================================================
# 14. META PREDICTION
# ==========================================================

ratio_pred = meta_model.predict(
    stack_test
)

ratio_pred = np.clip(
    ratio_pred,
    0.7,
    1.3
)

# FINAL FORECAST

forecast = (
    latest["lag_1"].values
    *
    ratio_pred
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

print("="*60)
print("IMPROVED STACKING SUBMISSION SAVED")
print("="*60)