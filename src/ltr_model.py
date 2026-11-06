import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker


# ============================================================
# PATHS
# ============================================================

LTR_DATASET = "models/ltr_dataset.csv"
MODEL_PATH = "models/ltr_model.pkl"
IMPORTANCE_PATH = "models/ltr_feature_importance.csv"
PREDICTIONS_PATH = "models/ltr_predictions.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("============================================================")
print("TRAINING CORRECTED LTR MODEL")
print("============================================================")

df = pd.read_csv(LTR_DATASET)

print(f"LTR dataset loaded")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# CORRECTED FEATURE LIST
# ============================================================

FEATURE_COLUMNS = [
    "cf_score",
    "content_similarity",
    "price_delta",
    "category_match",
    "brand_match",
    "material_match",
    "style_match"
]

TARGET_COLUMN = "label"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + [
    "user_id",
    "product_id",
    TARGET_COLUMN
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# CLEAN FEATURES
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y = df[TARGET_COLUMN].astype(int)

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)


# ============================================================
# CREATE RANKING GROUPS
# ============================================================

df["_row_order"] = np.arange(len(df))

df = df.sort_values(
    ["user_id", "_row_order"]
).reset_index(drop=True)

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].astype(int)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

group_sizes = (
    df.groupby("user_id", sort=False)
    .size()
    .tolist()
)


print()
print("============================================================")
print("RANKING GROUPS")
print("============================================================")

print(f"Number of users : {len(group_sizes)}")
print(f"Group sizes     : {group_sizes}")


# ============================================================
# TRAIN XGBOOST LTR
# ============================================================

print()
print("Training corrected XGBoost LTR model...")

model = XGBRanker(
    objective="rank:ndcg",
    eval_metric="ndcg@10",

    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=1,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,

    random_state=42
)


model.fit(
    X,
    y,
    group=group_sizes
)

print("LTR model training completed.")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print()
print("============================================================")
print("FEATURE IMPORTANCE")
print("============================================================")

print(importance.to_string(index=False))


# ============================================================
# GENERATE LTR PREDICTIONS
# ============================================================

df["ltr_score"] = model.predict(X)


# ============================================================
# SHOW SAMPLE RANKED RECOMMENDATIONS
# ============================================================

sample_user = df["user_id"].iloc[0]

sample = (
    df[df["user_id"] == sample_user]
    .sort_values("ltr_score", ascending=False)
    .head(10)
)

print()
print("============================================================")
print("SAMPLE RANKED RECOMMENDATIONS")
print("============================================================")

display_columns = [
    "user_id",
    "product_id",
    "cf_score",
    "content_similarity",
    "category_match",
    "brand_match",
    "material_match",
    "style_match",
    "label",
    "ltr_score"
]

print(
    sample[display_columns].to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_columns = [
    "user_id",
    "product_id",
    "label",
    "ltr_score"
]

df[prediction_columns].to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("============================================================")
print("CORRECTED LTR MODEL TRAINING COMPLETED")
print("============================================================")

print(f"Training rows       : {len(df)}")
print(f"Features used       : {len(FEATURE_COLUMNS)}")
print(f"Ranking groups      : {len(group_sizes)}")
print("Model               : XGBoost XGBRanker")
print("Objective           : rank:ndcg")
print("Evaluation metric   : ndcg@10")

print()
print("Features:")
for feature in FEATURE_COLUMNS:
    print(f" - {feature}")

print()
print(f"Model saved to:")
print(f" {MODEL_PATH}")

print()
print(f"Feature importance saved to:")
print(f" {IMPORTANCE_PATH}")

print()
print(f"Predictions saved to:")
print(f" {PREDICTIONS_PATH}")

print()
print("============================================================")
print("DONE")
print("============================================================")