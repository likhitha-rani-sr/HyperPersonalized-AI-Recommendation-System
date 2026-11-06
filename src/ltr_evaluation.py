import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "models/ltr_dataset.csv"
PREDICTIONS_PATH = "models/ltr_predictions.csv"
MODEL_PATH = "models/ltr_model.pkl"

OUTPUT_DIR = "models"
RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "ltr_evaluation_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    PREDICTIONS_PATH
)

model = joblib.load(
    MODEL_PATH
)

print("LTR predictions loaded")
print("Rows :", len(df))


# ============================================================
# VALIDATE
# ============================================================

required_columns = [
    "user_id",
    "product_id",
    "label",
    "ltr_score"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
    )


# ============================================================
# DCG
# ============================================================

def dcg_at_k(
    relevance,
    k
):

    relevance = np.asarray(
        relevance
    )[:k]

    if len(relevance) == 0:

        return 0.0

    positions = np.arange(
        1,
        len(relevance) + 1
    )

    discounts = np.log2(
        positions + 1
    )

    return np.sum(
        (2 ** relevance - 1)
        / discounts
    )


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(
    relevance,
    k
):

    actual_dcg = dcg_at_k(
        relevance,
        k
    )

    ideal_relevance = sorted(
        relevance,
        reverse=True
    )

    ideal_dcg = dcg_at_k(
        ideal_relevance,
        k
    )

    if ideal_dcg == 0:

        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    relevance,
    k
):

    relevance = np.asarray(
        relevance
    )[:k]

    if len(relevance) == 0:

        return 0.0

    return np.sum(
        relevance > 0
    ) / len(relevance)


# ============================================================
# EVALUATE EACH USER
# ============================================================

results = []

users = sorted(
    df["user_id"].unique()
)

print("\nEvaluating ranking performance...")

for user_id in users:

    user_df = df[
        df["user_id"] == user_id
    ].copy()

    # --------------------------------------------------------
    # Sort products according to LTR score
    # --------------------------------------------------------

    ranked_df = user_df.sort_values(
        by="ltr_score",
        ascending=False
    )

    relevance = (
        ranked_df["label"]
        .astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision_5 = precision_at_k(
        relevance,
        5
    )

    precision_10 = precision_at_k(
        relevance,
        10
    )

    ndcg_5 = ndcg_at_k(
        relevance,
        5
    )

    ndcg_10 = ndcg_at_k(
        relevance,
        10
    )

    results.append({

        "user_id":
            user_id,

        "precision_at_5":
            precision_5,

        "precision_at_10":
            precision_10,

        "ndcg_at_5":
            ndcg_5,

        "ndcg_at_10":
            ndcg_10
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# AVERAGE METRICS
# ============================================================

average_precision_5 = (
    results_df[
        "precision_at_5"
    ].mean()
)

average_precision_10 = (
    results_df[
        "precision_at_10"
    ].mean()
)

average_ndcg_5 = (
    results_df[
        "ndcg_at_5"
    ].mean()
)

average_ndcg_10 = (
    results_df[
        "ndcg_at_10"
    ].mean()
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LTR EVALUATION RESULTS")
print("=" * 60)

print(
    "\nPer-user results:\n"
)

print(
    results_df.to_string(
        index=False
    )
)

print("\nAverage metrics")
print(
    "-" * 40
)

print(
    f"Precision@5  : "
    f"{average_precision_5:.4f}"
)

print(
    f"Precision@10 : "
    f"{average_precision_10:.4f}"
)

print(
    f"NDCG@5       : "
    f"{average_ndcg_5:.4f}"
)

print(
    f"NDCG@10      : "
    f"{average_ndcg_10:.4f}"
)


# ============================================================
# POSITIVE LABEL INFORMATION
# ============================================================

total_positive = int(
    df["label"].sum()
)

total_rows = len(df)

positive_rate = (
    total_positive / total_rows
)


print("\nDataset information")
print(
    "-" * 40
)

print(
    "Total rows       :",
    total_rows
)

print(
    "Positive labels  :",
    total_positive
)

print(
    f"Positive rate    : "
    f"{positive_rate:.4f}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print(
    "LTR EVALUATION COMPLETED"
)

print(
    "Results saved to:"
)

print(
    RESULTS_PATH
)

print("=" * 60)