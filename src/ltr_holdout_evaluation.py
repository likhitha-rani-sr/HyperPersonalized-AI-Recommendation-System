import os
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from xgboost import XGBRanker


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PRODUCTS_PATH = "data/processed/processed_products.csv"
INTERACTIONS_PATH = "data/processed/user_product_interactions.csv"
EMBEDDINGS_PATH = "embeddings/product_embeddings.npy"
RESULTS_PATH = "models/ltr_holdout_results.csv"

RANDOM_STATE = 42
HOLDOUT_RATIO = 0.20
LTR_LABEL_RATIO = 0.20
NEGATIVES_PER_POSITIVE = 10

FEATURE_COLUMNS = [
    "cf_score",
    "content_similarity",
    "price_delta",
    "category_match",
    "brand_match",
    "material_match",
    "style_match",
]


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def precision_at_k(recommended, relevant, k):
    """Precision@k uses k as the denominator, as is standard."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    recommended = list(recommended)[:k]
    if not recommended:
        return 0.0
    hits = sum(item in relevant for item in recommended)
    return hits / k


def ndcg_at_k(recommended, relevant, k):
    """Binary-relevance NDCG@k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    recommended = list(recommended)[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )
    return float(dcg / ideal_dcg) if ideal_dcg else 0.0


def split_strong_interactions(interactions, rng, holdout_ratio):
    """Hold out strong interactions per user, retaining at least one for training."""
    train_parts = []
    heldout_parts = []

    for _, user_df in interactions.groupby("user_id", sort=True):
        user_df = user_df.copy()
        strong = user_df[user_df["interaction"] >= 5]
        weak = user_df[user_df["interaction"] < 5]

        if len(strong) >= 2:
            count = max(1, int(np.floor(len(strong) * holdout_ratio)))
            count = min(count, len(strong) - 1)
            chosen = rng.choice(strong.index.to_numpy(), size=count, replace=False)
            heldout = strong.loc[chosen]
            remaining_strong = strong.drop(index=chosen)
        else:
            heldout = strong.iloc[0:0]
            remaining_strong = strong

        train_parts.append(pd.concat([weak, remaining_strong], axis=0))
        heldout_parts.append(heldout)

    train_df = pd.concat(train_parts, ignore_index=True)
    heldout_df = pd.concat(heldout_parts, ignore_index=True)
    return train_df, heldout_df


def build_interaction_matrix(interactions, user_to_index, num_products):
    rows, cols, values = [], [], []

    for row in interactions.itertuples(index=False):
        user_id = row.user_id
        product_id = int(row.product_id)

        if user_id not in user_to_index:
            continue
        if not 1 <= product_id <= num_products:
            continue

        rows.append(user_to_index[user_id])
        cols.append(product_id - 1)
        values.append(float(row.interaction))

    return csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_to_index), num_products),
        dtype=np.float32,
    )


def train_als(interactions, user_to_index, num_products):
    matrix = build_interaction_matrix(
        interactions, user_to_index, num_products
    )
    model = AlternatingLeastSquares(
        factors=20,
        iterations=20,
        regularization=0.1,
        random_state=RANDOM_STATE,
    )
    # Avoid excessive BLAS thread oversubscription where threadpoolctl is installed.
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            model.fit(matrix)
    except ImportError:
        model.fit(matrix)
    return model, matrix


def build_user_profiles(profile_interactions, user_ids, products, embeddings):
    profiles = {}
    strong = profile_interactions[
        profile_interactions["interaction"] >= 5
    ]

    for user_id in user_ids:
        product_ids = (
            strong.loc[strong["user_id"] == user_id, "product_id"]
            .astype(int)
            .tolist()
        )
        valid = [
            product_id - 1
            for product_id in product_ids
            if 1 <= product_id <= len(embeddings)
        ]

        if not valid:
            profile = np.zeros(embeddings.shape[1], dtype=np.float32)
        else:
            profile = embeddings[valid].mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm

        profiles[user_id] = profile

    return profiles, {
        user_id: set(
            strong.loc[strong["user_id"] == user_id, "product_id"].astype(int)
        )
        for user_id in user_ids
    }


def make_feature_builder(
    products,
    embeddings,
    user_ids,
    user_to_index,
    als,
    profiles,
    strong_ids_by_user,
):
    num_products = len(products)
    product_lookup = products.set_index("product_id")

    def calculate_features(user_id, candidate_product_ids):
        candidate_product_ids = [
            int(pid) for pid in candidate_product_ids
            if 1 <= int(pid) <= num_products
        ]
        if not candidate_product_ids:
            return pd.DataFrame(columns=["user_id", "product_id"] + FEATURE_COLUMNS)

        user_index = user_to_index[user_id]
        user_factor = als.user_factors[user_index]
        raw_scores = np.asarray(als.item_factors @ user_factor).reshape(-1)
        score_min = float(raw_scores.min()) if len(raw_scores) else 0.0
        score_max = float(raw_scores.max()) if len(raw_scores) else 0.0

        if score_max > score_min:
            normalized_cf = (raw_scores - score_min) / (score_max - score_min)
        else:
            normalized_cf = np.zeros_like(raw_scores)

        user_profile = profiles[user_id]
        preferred_products = products[
            products["product_id"].isin(strong_ids_by_user[user_id])
        ]

        preferred_categories = set(preferred_products["category"])
        preferred_brands = set(preferred_products["brand"])
        preferred_materials = set(preferred_products["material"])
        preferred_styles = set(preferred_products["style"])
        preferred_price = (
            float(preferred_products["price_numeric"].mean())
            if len(preferred_products) else float(products["price_numeric"].mean())
        )

        rows = []
        for product_id in candidate_product_ids:
            product = product_lookup.loc[product_id]
            idx = product_id - 1
            similarity = float(np.dot(user_profile, embeddings[idx]))
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            price = float(product["price_numeric"])
            price_delta = (
                abs(price - preferred_price) / max(preferred_price, 1.0)
                if preferred_price > 0 else 0.0
            )

            rows.append({
                "user_id": user_id,
                "product_id": product_id,
                "cf_score": float(normalized_cf[idx]),
                "content_similarity": similarity,
                "price_delta": float(price_delta),
                "category_match": int(product["category"] in preferred_categories),
                "brand_match": int(product["brand"] in preferred_brands),
                "material_match": int(product["material"] in preferred_materials),
                "style_match": int(product["style"] in preferred_styles),
            })

        return pd.DataFrame(rows)

    return calculate_features


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def main():
    print("Loading data...")

    products = pd.read_csv(PRODUCTS_PATH)
    interactions = pd.read_csv(INTERACTIONS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    required_interaction_cols = {"user_id", "product_id", "interaction"}
    missing = required_interaction_cols - set(interactions.columns)
    if missing:
        raise ValueError(f"Interactions file is missing columns: {sorted(missing)}")

    if "price" not in products.columns:
        raise ValueError("Products file must contain a 'price' column.")
    if len(products) != len(embeddings):
        raise ValueError(
            f"Product/embedding count mismatch: {len(products)} products, "
            f"{len(embeddings)} embeddings."
        )

    products = products.copy()
    products["product_id"] = np.arange(1, len(products) + 1)
    for column in ["category", "brand", "color", "material", "style"]:
        if column not in products.columns:
            products[column] = ""
        products[column] = products[column].apply(clean_text)
    products["price_numeric"] = pd.to_numeric(
        products["price"], errors="coerce"
    ).fillna(0.0)

    interactions = interactions.copy()
    interactions["product_id"] = pd.to_numeric(
        interactions["product_id"], errors="coerce"
    )
    interactions["interaction"] = pd.to_numeric(
        interactions["interaction"], errors="coerce"
    )
    interactions = interactions.dropna(
        subset=["user_id", "product_id", "interaction"]
    )
    interactions["product_id"] = interactions["product_id"].astype(int)
    interactions = interactions[
        interactions["product_id"].between(1, len(products))
    ].copy()

    print(f"Products loaded       : {len(products)}")
    print(f"Interactions loaded   : {len(interactions)}")
    print(f"Embeddings shape      : {embeddings.shape}")

    rng = np.random.default_rng(RANDOM_STATE)
    training_interactions, heldout_interactions = split_strong_interactions(
        interactions, rng, HOLDOUT_RATIO
    )

    user_ids = sorted(training_interactions["user_id"].unique().tolist())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    num_products = len(products)

    print("\nCreating holdout split...")
    print("-" * 40)
    print(f"Training interactions : {len(training_interactions)}")
    print(f"Held-out positives    : {len(heldout_interactions)}")
    print(f"Users                 : {len(user_ids)}")

    # Make a second, internal split from the remaining strong interactions.
    # The base recommender sees only base_interactions; LTR labels come from
    # ltr_label_interactions, preventing each LTR training positive from being
    # included in the ALS/profile features used to train its label.
    base_parts, ltr_label_parts = [], []
    for user_id, user_df in training_interactions.groupby("user_id", sort=True):
        strong = user_df[user_df["interaction"] >= 5]
        weak = user_df[user_df["interaction"] < 5]

        if len(strong) >= 2:
            count = max(1, int(np.floor(len(strong) * LTR_LABEL_RATIO)))
            count = min(count, len(strong) - 1)
            local_rng = np.random.default_rng(RANDOM_STATE + 2000 + int(user_id))
            chosen = local_rng.choice(strong.index.to_numpy(), size=count, replace=False)
            ltr_labels = strong.loc[chosen]
            base_strong = strong.drop(index=chosen)
        else:
            ltr_labels = strong.iloc[0:0]
            base_strong = strong

        base_parts.append(pd.concat([weak, base_strong], axis=0))
        ltr_label_parts.append(ltr_labels)

    base_interactions = pd.concat(base_parts, ignore_index=True)
    ltr_label_interactions = pd.concat(ltr_label_parts, ignore_index=True)

    print(f"Base-model interactions: {len(base_interactions)}")
    print(f"LTR training positives : {len(ltr_label_interactions)}")

    print("\nTraining ALS on base interactions...")
    als, interaction_matrix = train_als(
        base_interactions, user_to_index, num_products
    )
    print(f"Interaction matrix : {interaction_matrix.shape}")
    print("ALS training completed.")

    profiles, strong_ids_by_user = build_user_profiles(
        base_interactions, user_ids, products, embeddings
    )
    calculate_features = make_feature_builder(
        products, embeddings, user_ids, user_to_index,
        als, profiles, strong_ids_by_user
    )

    # Build LTR training rows from internal held-out positives plus sampled
    # products with no recorded interactions in the base or label sets.
    print("\nCreating leakage-reduced LTR training features...")
    training_rows = []
    for user_id in user_ids:
        base_ids = set(
            base_interactions.loc[
                base_interactions["user_id"] == user_id, "product_id"
            ].astype(int)
        )
        label_ids = set(
            ltr_label_interactions.loc[
                ltr_label_interactions["user_id"] == user_id, "product_id"
            ].astype(int)
        )
        outer_holdout_ids = set(
            heldout_interactions.loc[
                heldout_interactions["user_id"] == user_id, "product_id"
            ].astype(int)
        )

        available_negatives = [
            pid for pid in range(1, num_products + 1)
            if pid not in base_ids
            and pid not in label_ids
            and pid not in outer_holdout_ids
        ]
        local_rng = np.random.default_rng(RANDOM_STATE + int(user_id))
        negative_count = min(
            len(available_negatives),
            max(len(label_ids) * NEGATIVES_PER_POSITIVE, 20),
        )
        negative_ids = (
            local_rng.choice(available_negatives, size=negative_count, replace=False).tolist()
            if negative_count else []
        )
        candidate_ids = sorted(label_ids) + negative_ids
        features = calculate_features(user_id, candidate_ids)
        if features.empty:
            continue
        features["label"] = features["product_id"].isin(label_ids).astype(int)
        training_rows.append(features)

    if not training_rows:
        raise ValueError(
            "No LTR training examples could be created. Check strong interaction counts."
        )

    training_ltr = pd.concat(training_rows, ignore_index=True)
    training_ltr = training_ltr.sort_values(
        ["user_id", "product_id"], kind="stable"
    ).reset_index(drop=True)

    group_sizes = training_ltr.groupby("user_id", sort=True).size().tolist()
    if training_ltr["label"].sum() == 0:
        raise ValueError("LTR training data contains no positive labels.")

    print("\nTraining XGBoost LTR...")
    ltr_model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    ltr_model.fit(
        training_ltr[FEATURE_COLUMNS],
        training_ltr["label"],
        group=group_sizes,
    )
    print("LTR training completed.")

    # Evaluate against every product the user has not interacted with in the
    # outer training split. Held-out positives are included; all other unseen
    # items are treated as non-relevant for this offline implicit-feedback test.
    # This is stricter than sampled negatives, but unobserved items are not
    # guaranteed true negatives, so interpret metrics cautiously.
    print("\nCreating full-catalog held-out test features...")
    results = []
    for user_id in user_ids:
        outer_train_ids = set(
            training_interactions.loc[
                training_interactions["user_id"] == user_id, "product_id"
            ].astype(int)
        )
        heldout_ids = set(
            heldout_interactions.loc[
                heldout_interactions["user_id"] == user_id, "product_id"
            ].astype(int)
        )
        if not heldout_ids:
            continue

        candidate_ids = [
            pid for pid in range(1, num_products + 1)
            if pid not in outer_train_ids
        ]
        test_features = calculate_features(user_id, candidate_ids)
        if test_features.empty:
            continue

        test_features["ltr_score"] = ltr_model.predict(
            test_features[FEATURE_COLUMNS]
        )
        ranked = test_features.sort_values(
            ["ltr_score", "product_id"],
            ascending=[False, True],
            kind="stable",
        )
        ranked_ids = ranked["product_id"].astype(int).tolist()

        results.append({
            "user_id": user_id,
            "heldout_positive_count": len(heldout_ids),
            "candidate_count": len(candidate_ids),
            "hits_at_5": len(set(ranked_ids[:5]) & heldout_ids),
            "hits_at_10": len(set(ranked_ids[:10]) & heldout_ids),
            "precision_at_5": precision_at_k(ranked_ids, heldout_ids, 5),
            "precision_at_10": precision_at_k(ranked_ids, heldout_ids, 10),
            "ndcg_at_5": ndcg_at_k(ranked_ids, heldout_ids, 5),
            "ndcg_at_10": ndcg_at_k(ranked_ids, heldout_ids, 10),
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("FULL-CATALOG LTR HOLDOUT EVALUATION")
    print("=" * 70)
    if results_df.empty:
        print("No users could be evaluated.")
    else:
        print("\nPer-user results:")
        print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

        print("\nAverage metrics")
        print("-" * 40)
        for metric in [
            "precision_at_5", "precision_at_10", "ndcg_at_5", "ndcg_at_10"
        ]:
            print(f"{metric.replace('_', ' ').title():<18}: {results_df[metric].mean():.4f}")

        print("\nHoldout information")
        print("-" * 40)
        print(f"Base-model interactions : {len(base_interactions)}")
        print(f"LTR training positives  : {int(training_ltr['label'].sum())}")
        print(f"Outer held-out positives: {len(heldout_interactions)}")
        print(f"Users evaluated         : {len(results_df)}")
        print(f"Average candidates/user : {results_df['candidate_count'].mean():.1f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False, float_format="%.6f")
    print("\n" + "=" * 70)
    print("HOLDOUT EVALUATION COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(
        "\nInterpretation note: this is an offline implicit-feedback evaluation. "
        "Unobserved products are treated as non-relevant, and only five users are "
        "available, so use these metrics as preliminary evidence rather than a "
        "production-quality guarantee."
    )


if __name__ == "__main__":
    main()
