import os
import pickle
import warnings

import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "models/ltr_dataset.csv"

MODEL_PATH = "models/ltr_model_optuna.pkl"

RESULTS_PATH = "models/optuna_ltr_results.csv"

PARAMS_PATH = "models/optuna_best_params.json"

IMPORTANCE_PATH = "models/ltr_optuna_feature_importance.csv"


# ============================================================
# SETTINGS
# ============================================================

N_TRIALS = 50

RANDOM_STATE = 42


# ============================================================
# FEATURES
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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading LTR dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows                : {len(df)}")
print(f"Columns             : {len(df.columns)}")
print(f"Users               : {df['user_id'].nunique()}")
print(f"Products            : {df['product_id'].nunique()}")

print("\nFeature columns:")
for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y = df["label"].astype(float)

groups = df["user_id"].values


# ============================================================
# GROUP-AWARE TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating group-aware train/validation split...")

unique_users = np.array(
    sorted(df["user_id"].unique())
)

rng = np.random.RandomState(
    RANDOM_STATE
)

shuffled_users = rng.permutation(
    unique_users
)

split_point = max(
    1,
    int(len(shuffled_users) * 0.8)
)

train_users = shuffled_users[
    :split_point
]

validation_users = shuffled_users[
    split_point:
]


# If the dataset has very few users, use the
# last user as validation.
if len(validation_users) == 0:

    validation_users = train_users[-1:]

    train_users = train_users[:-1]


train_mask = df["user_id"].isin(
    train_users
)

validation_mask = df["user_id"].isin(
    validation_users
)


X_train = X[train_mask]
y_train = y[train_mask]

X_valid = X[validation_mask]
y_valid = y[validation_mask]

groups_train = (
    df.loc[
        train_mask,
        "user_id"
    ]
    .value_counts()
    .sort_index()
    .tolist()
)

groups_valid = (
    df.loc[
        validation_mask,
        "user_id"
    ]
    .value_counts()
    .sort_index()
    .tolist()
)


print(
    f"Training rows        : {len(X_train)}"
)

print(
    f"Validation rows      : {len(X_valid)}"
)

print(
    f"Training users       : {len(train_users)}"
)

print(
    f"Validation users     : {len(validation_users)}"
)


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial):

    params = {

        "objective": "rank:ndcg",

        "eval_metric": "ndcg@10",

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0.0,
            2.0
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.1,
            5.0,
            log=True
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            2.0
        ),

        "random_state": RANDOM_STATE,

        "verbosity": 0
    }


    model = XGBRanker(
        **params
    )


    model.fit(
        X_train,
        y_train,
        group=groups_train
    )


    predictions = model.predict(
        X_valid
    )


    # --------------------------------------------------------
    # Calculate validation NDCG@10
    # --------------------------------------------------------

    validation_df = df[
        validation_mask
    ].copy()

    validation_df[
        "prediction"
    ] = predictions


    ndcg_values = []


    for user_id, user_df in validation_df.groupby(
        "user_id"
    ):

        true_values = (
            user_df["label"]
            .values
            .reshape(1, -1)
        )

        predicted_values = (
            user_df["prediction"]
            .values
            .reshape(1, -1)
        )


        k = min(
            10,
            len(user_df)
        )


        if k > 0:

            score = ndcg_score(
                true_values,
                predicted_values,
                k=k
            )

            ndcg_values.append(
                score
            )


    if len(ndcg_values) == 0:

        return 0.0


    return float(
        np.mean(ndcg_values)
    )


# ============================================================
# CREATE OPTUNA STUDY
# ============================================================

print("\n")
print("=" * 70)
print("STARTING OPTUNA LTR TUNING")
print("=" * 70)

print(
    f"\nNumber of trials : {N_TRIALS}"
)

print(
    "Optimization metric: Validation NDCG@10"
)


study = optuna.create_study(
    direction="maximize",
    study_name="ltr_xgboost_optimization"
)


study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)


# ============================================================
# BEST RESULT
# ============================================================

print("\n")
print("=" * 70)
print("OPTUNA OPTIMIZATION COMPLETED")
print("=" * 70)


print(
    f"\nBest trial number : "
    f"{study.best_trial.number}"
)

print(
    f"Best NDCG@10      : "
    f"{study.best_value:.6f}"
)


print("\nBest parameters:")

for parameter, value in study.best_params.items():

    print(
        f"{parameter:20s}: {value}"
    )


# ============================================================
# SAVE OPTUNA RESULTS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


trials_df = study.trials_dataframe()

trials_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# SAVE BEST PARAMETERS
# ============================================================

import json


with open(
    PARAMS_PATH,
    "w"
) as file:

    json.dump(
        study.best_params,
        file,
        indent=4
    )


# ============================================================
# RETRAIN FINAL MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING FINAL OPTIMIZED LTR MODEL")
print("=" * 70)


best_params = study.best_params.copy()

best_params.update({

    "objective": "rank:ndcg",

    "eval_metric": "ndcg@10",

    "random_state": RANDOM_STATE,

    "verbosity": 0
})


final_model = XGBRanker(
    **best_params
)


# Rebuild complete group sizes

groups_full = (
    df["user_id"]
    .value_counts()
    .sort_index()
    .tolist()
)


final_model.fit(
    X,
    y,
    group=groups_full
)


print(
    "\nFinal optimized model trained."
)


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        final_model,
        file
    )


print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance": (
        final_model.feature_importances_
    )
})


importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False
)


# ============================================================
# DISPLAY FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance:")

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("OPTUNA LTR PIPELINE COMPLETED")
print("=" * 70)

print(
    f"\nTrials completed : "
    f"{len(study.trials)}"
)

print(
    f"Best NDCG@10     : "
    f"{study.best_value:.6f}"
)

print(
    f"\nSaved model:"
)

print(
    MODEL_PATH
)

print(
    "\nSaved tuning results:"
)

print(
    RESULTS_PATH
)

print(
    "\nSaved best parameters:"
)

print(
    PARAMS_PATH
)

print(
    "\nSaved feature importance:"
)

print(
    IMPORTANCE_PATH
)

print("\nDone.")