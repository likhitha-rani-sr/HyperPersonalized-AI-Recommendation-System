# ============================================================
# HYBRID RECOMMENDER SYSTEM
# Sprint 2 - Week 4 | FastAPI Integration - Sprint 3 Week 6
# ALS collaborative filtering + sentence embeddings + optional LTR
# Returned product IDs are zero-based and match product embedding indices.
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_products.csv")
INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "user_product_interactions.csv")
LTR_MODEL_PATH = os.path.join(BASE_DIR, "models", "ltr_model_optuna.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "hybrid_recommendations.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "product_embeddings.npy")

STRONG_INTERACTION_THRESHOLD = 5
ALS_FACTORS = 20
ALS_ITERATIONS = 20
ALS_REGULARIZATION = 0.1
DEFAULT_TOP_K = 10
CANDIDATE_POOL_SIZE = 100
CF_WEIGHT = 0.20
CONTENT_WEIGHT = 0.30
LTR_WEIGHT = 0.50

LTR_FEATURE_COLUMNS = [
    "cf_score", "content_similarity", "price_delta", "category_match",
    "brand_match", "material_match", "style_match"
]


def clean_text(value):
    """Convert a value to clean lowercase text."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def safe_float(value, default=0.0):
    """Convert a value to a finite float safely."""
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_scores(values):
    """Normalize scores to the range 0 to 1."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if maximum - minimum < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def find_column(dataframe, possible_names):
    """Find the first matching column name."""
    for name in possible_names:
        if name in dataframe.columns:
            return name
    return None


def normalize_vector(vector):
    """Normalize a vector safely."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector if norm < 1e-12 else vector / norm


# -------------------- Load data --------------------

if not os.path.exists(PRODUCTS_PATH):
    raise FileNotFoundError(f"Products file not found: {PRODUCTS_PATH}")
if not os.path.exists(INTERACTIONS_PATH):
    raise FileNotFoundError(f"Interactions file not found: {INTERACTIONS_PATH}")

products = pd.read_csv(PRODUCTS_PATH).reset_index(drop=True)
interactions = pd.read_csv(INTERACTIONS_PATH)
if products.empty:
    raise ValueError("The product dataset is empty.")

required_interaction_columns = {"user_id", "product_id", "interaction"}
missing_columns = required_interaction_columns - set(interactions.columns)
if missing_columns:
    raise ValueError(f"Missing interaction columns: {missing_columns}")

# Product rows and embeddings are zero-based; interaction CSV IDs are 1-based.
products["product_id"] = np.arange(len(products), dtype=int)
interactions["user_id"] = pd.to_numeric(interactions["user_id"], errors="raise").astype(int)
interactions["product_id"] = (
    pd.to_numeric(interactions["product_id"], errors="raise").astype(int) - 1
)
interactions["interaction"] = pd.to_numeric(
    interactions["interaction"], errors="raise"
).astype(float)

if not interactions["product_id"].between(0, len(products) - 1).all():
    raise ValueError("Interaction product IDs do not match the product dataset.")

NAME_COLUMN = find_column(products, ["product_name", "name", "title"])
CATEGORY_COLUMN = find_column(products, ["category", "product_category"])
BRAND_COLUMN = find_column(products, ["brand", "product_brand"])
COLOR_COLUMN = find_column(products, ["color", "colour"])
MATERIAL_COLUMN = find_column(products, ["material", "product_material"])
PRICE_COLUMN = find_column(products, ["price", "product_price"])
STYLE_COLUMN = find_column(products, ["style", "product_style"])

products["_category_clean"] = products[CATEGORY_COLUMN].map(clean_text) if CATEGORY_COLUMN else ""
products["_brand_clean"] = products[BRAND_COLUMN].map(clean_text) if BRAND_COLUMN else ""
products["_material_clean"] = products[MATERIAL_COLUMN].map(clean_text) if MATERIAL_COLUMN else ""
products["_style_clean"] = products[STYLE_COLUMN].map(clean_text) if STYLE_COLUMN else ""
products["_price_numeric"] = (
    pd.to_numeric(products[PRICE_COLUMN], errors="coerce") if PRICE_COLUMN else np.nan
)

print("Loading content recommender...")
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"Product embeddings not found: {EMBEDDINGS_PATH}")

content_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer("all-MiniLM-L6-v2")

if len(content_embeddings) != len(products):
    raise ValueError(
        "Product count does not match embedding count. "
        f"Products: {len(products)}, embeddings: {len(content_embeddings)}"
    )

embedding_norms = np.linalg.norm(content_embeddings, axis=1, keepdims=True)
embedding_norms[embedding_norms < 1e-12] = 1.0
normalized_embeddings = (content_embeddings / embedding_norms).astype(np.float32)
print(f"Content embeddings loaded: {normalized_embeddings.shape}")

ltr_model = None
if os.path.exists(LTR_MODEL_PATH):
    try:
        with open(LTR_MODEL_PATH, "rb") as file:
            ltr_model = pickle.load(file)
        print("LTR model loaded.")
    except Exception as error:
        warnings.warn(f"Could not load LTR model: {error}. Using fallback ranking.")
else:
    warnings.warn("LTR model file was not found. Using fallback ranking.")

# -------------------- ALS matrix --------------------

product_ids = list(range(len(products)))
user_ids = sorted(interactions["user_id"].unique().tolist())
user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

matrix_df = interactions.pivot_table(
    index="user_id", columns="product_id", values="interaction",
    aggfunc="sum", fill_value=0
)
matrix_df = matrix_df.reindex(index=user_ids, columns=product_ids, fill_value=0)
user_item_matrix = csr_matrix(matrix_df.to_numpy(dtype=np.float32))
interaction_matrix = matrix_df

print("Training ALS collaborative filtering model...")
als_model = AlternatingLeastSquares(
    factors=ALS_FACTORS,
    iterations=ALS_ITERATIONS,
    regularization=ALS_REGULARIZATION,
    random_state=42
)
if user_item_matrix.shape[0] > 0:
    als_model.fit(user_item_matrix.T.tocsr())
print("ALS model ready.")


def get_product_details(product_id):
    """Return product details for a zero-based product ID."""
    try:
        product_id = int(product_id)
    except (TypeError, ValueError):
        return None
    if not 0 <= product_id < len(products):
        return None
    return products.iloc[product_id]


def get_user_interacted_ids(user_id):
    """Return all zero-based product IDs interacted with by a user."""
    user_data = interactions[interactions["user_id"] == int(user_id)]
    return set(user_data["product_id"].astype(int).tolist())


def get_user_strong_interactions(user_id, threshold=STRONG_INTERACTION_THRESHOLD):
    """Return product IDs with strong interactions."""
    user_data = interactions[
        (interactions["user_id"] == int(user_id))
        & (interactions["interaction"] >= threshold)
    ]
    return set(user_data["product_id"].astype(int).tolist())


def get_query_embedding(query):
    """Encode a text query using the sentence-transformer."""
    query = str(query or "").strip()
    if not query:
        return None
    vector = text_model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    )
    return normalize_vector(vector)


def get_query_content_scores(query):
    """Calculate query-to-product cosine similarities."""
    query_vector = get_query_embedding(query)
    if query_vector is None:
        return np.zeros(len(products), dtype=np.float32)
    return np.asarray(np.dot(normalized_embeddings, query_vector), dtype=np.float32)


def build_user_content_profile(user_id):
    """Build a content profile from products with strong interactions."""
    strong_ids = sorted(get_user_strong_interactions(user_id))
    valid_ids = [
        product_id for product_id in strong_ids
        if 0 <= product_id < len(normalized_embeddings)
    ]
    if not valid_ids:
        return None
    return normalize_vector(np.mean(normalized_embeddings[valid_ids], axis=0))


def get_personal_content_scores(user_id):
    """Calculate similarity to the user's content profile."""
    profile = build_user_content_profile(user_id)
    if profile is None:
        return np.zeros(len(products), dtype=np.float32)
    return np.asarray(np.dot(normalized_embeddings, profile), dtype=np.float32)


def get_cf_candidates(user_id, candidate_count):
    """Return zero-based product IDs and ALS scores."""
    if user_id not in user_id_to_index:
        return {}
    user_index = user_id_to_index[user_id]
    user_items = user_item_matrix[user_index]
    try:
        item_indices, scores = als_model.recommend(
            user_index, user_items,
            N=min(candidate_count, len(products)),
            filter_already_liked_items=False
        )
    except Exception as error:
        warnings.warn(f"ALS recommendation failed for user {user_id}: {error}")
        return {}
    return {
        int(item_index): safe_float(score)
        for item_index, score in zip(item_indices, scores)
        if 0 <= int(item_index) < len(products)
    }


def get_user_preferences(user_id):
    """Collect common category, brand, material, style, and price."""
    strong_ids = sorted(get_user_strong_interactions(user_id))
    valid_ids = [i for i in strong_ids if 0 <= i < len(products)]
    user_products = products.iloc[valid_ids] if valid_ids else products.iloc[0:0]

    preferences = {
        "categories": set(), "brands": set(), "materials": set(),
        "styles": set(), "average_price": None
    }
    if CATEGORY_COLUMN:
        preferences["categories"] = {
            clean_text(v) for v in user_products[CATEGORY_COLUMN].dropna() if clean_text(v)
        }
    if BRAND_COLUMN:
        preferences["brands"] = {
            clean_text(v) for v in user_products[BRAND_COLUMN].dropna() if clean_text(v)
        }
    if MATERIAL_COLUMN:
        preferences["materials"] = {
            clean_text(v) for v in user_products[MATERIAL_COLUMN].dropna() if clean_text(v)
        }
    if STYLE_COLUMN:
        preferences["styles"] = {
            clean_text(v) for v in user_products[STYLE_COLUMN].dropna() if clean_text(v)
        }
    if not user_products.empty:
        prices = pd.to_numeric(user_products["_price_numeric"], errors="coerce").dropna()
        if not prices.empty:
            preferences["average_price"] = float(prices.mean())
    return preferences


CATEGORY_KEYWORDS = {
    "sofa": ["sofa", "couch", "sectional", "loveseat"],
    "table": ["table", "desk"],
    "chair": ["chair", "stool", "recliner"],
    "wardrobe": ["wardrobe", "closet"],
    "bed": ["bed", "mattress"],
}


def detect_query_categories(query):
    """Detect requested categories with whole-word matching."""
    words = set(query.lower().replace("-", " ").replace("/", " ").split())
    return {
        category for category, keywords in CATEGORY_KEYWORDS.items()
        if any(keyword in words for keyword in keywords)
    }


def build_features(user_id, candidate_ids, cf_scores, content_scores, preferences):
    """Build the seven features expected by the existing LTR model."""
    rows = []
    for product_id in candidate_ids:
        product = products.iloc[int(product_id)]
        cf_score = safe_float(cf_scores.get(int(product_id), 0.0))
        content_score = safe_float(content_scores[int(product_id)])

        average_price = preferences["average_price"]
        product_price = safe_float(product["_price_numeric"], default=np.nan)
        if average_price is not None and np.isfinite(product_price) and average_price > 0:
            price_delta = abs(product_price - average_price) / average_price
        else:
            price_delta = 0.0

        rows.append({
            "cf_score": cf_score,
            "content_similarity": content_score,
            "price_delta": price_delta,
            "category_match": int(
                bool(product["_category_clean"])
                and product["_category_clean"] in preferences["categories"]
            ),
            "brand_match": int(
                bool(product["_brand_clean"])
                and product["_brand_clean"] in preferences["brands"]
            ),
            "material_match": int(
                bool(product["_material_clean"])
                and product["_material_clean"] in preferences["materials"]
            ),
            "style_match": int(
                bool(product["_style_clean"])
                and product["_style_clean"] in preferences["styles"]
            ),
        })
    return pd.DataFrame(rows, columns=LTR_FEATURE_COLUMNS)


def calculate_ltr_scores(feature_df):
    """Use the saved LTR model if possible, otherwise a safe fallback."""
    if feature_df.empty:
        return np.array([], dtype=float)
    if ltr_model is not None:
        try:
            predictions = np.asarray(ltr_model.predict(feature_df), dtype=float).reshape(-1)
            if len(predictions) == len(feature_df) and np.isfinite(predictions).all():
                return predictions
            warnings.warn("LTR predictions were invalid. Using fallback ranking.")
        except Exception as error:
            warnings.warn(
                "Saved LTR model could not score these features. "
                f"Using fallback ranking. Details: {error}"
            )
    cf = normalize_scores(feature_df["cf_score"].to_numpy())
    content = normalize_scores(feature_df["content_similarity"].to_numpy())
    return 0.5 * cf + 0.5 * content


def recommend_for_user(user_id, top_k=DEFAULT_TOP_K, query=""):
    """Generate personalized or query-based hybrid recommendations."""
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return pd.DataFrame()
    if user_id not in user_id_to_index:
        return pd.DataFrame()

    try:
        top_k = max(1, min(int(top_k), len(products)))
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K

    query = str(query or "").strip()
    content_scores = (
        get_query_content_scores(query) if query
        else get_personal_content_scores(user_id)
    )
    pool_size = min(max(CANDIDATE_POOL_SIZE, top_k * 10), len(products))
    cf_scores = get_cf_candidates(user_id, candidate_count=pool_size)

    semantic_ids = set()
    if query:
        semantic_ids = set(np.argsort(content_scores)[::-1][:pool_size].tolist())
    candidate_ids = semantic_ids.union(set(cf_scores.keys()))
    if not candidate_ids:
        candidate_ids = set(np.argsort(content_scores)[::-1][:pool_size].tolist())

    # If the query names a category, keep only that category.
    requested_categories = detect_query_categories(query) if query else set()
    if requested_categories and CATEGORY_COLUMN:
        category_matches = set(
            products.index[products["_category_clean"].isin(requested_categories)].astype(int)
        )
        candidate_ids = candidate_ids.intersection(category_matches)

    # Remove items already seen by this user.
    already_interacted = get_user_interacted_ids(user_id)
    candidate_ids = {
        int(i) for i in candidate_ids
        if 0 <= int(i) < len(products) and int(i) not in already_interacted
    }

    # Fallback 1: search the full catalog for unseen products.
    if not candidate_ids:
        eligible_ids = set(range(len(products))) - already_interacted
        if requested_categories and CATEGORY_COLUMN:
            category_matches = set(
                products.index[products["_category_clean"].isin(requested_categories)].astype(int)
            )
            eligible_ids &= category_matches

        candidate_ids = set(sorted(
            eligible_ids, key=lambda i: content_scores[i], reverse=True
        )[:pool_size])

    # Fallback 2: if no unseen products match a requested category, allow
    # relevant repeat recommendations from that category.
    repeat_recommendations = False
    if not candidate_ids and requested_categories and CATEGORY_COLUMN:
        category_matches = set(
            products.index[products["_category_clean"].isin(requested_categories)].astype(int)
        )
        candidate_ids = set(sorted(
            category_matches, key=lambda i: content_scores[i], reverse=True
        )[:pool_size])
        repeat_recommendations = bool(candidate_ids)

    if not candidate_ids:
        return pd.DataFrame()

    candidate_ids = sorted(candidate_ids)
    preferences = get_user_preferences(user_id)
    feature_df = build_features(
        user_id, candidate_ids, cf_scores, content_scores, preferences
    )
    ltr_scores = calculate_ltr_scores(feature_df)

    cf_values = feature_df["cf_score"].to_numpy(dtype=float)
    content_values = feature_df["content_similarity"].to_numpy(dtype=float)
    cf_normalized = normalize_scores(cf_values)
    content_normalized = normalize_scores(content_values)
    ltr_normalized = normalize_scores(ltr_scores)

    if query:
        hybrid_scores = (
            0.80 * content_normalized
            + 0.10 * cf_normalized
            + 0.10 * ltr_normalized
        )
    else:
        hybrid_scores = (
            CF_WEIGHT * cf_normalized
            + CONTENT_WEIGHT * content_normalized
            + LTR_WEIGHT * ltr_normalized
        )

    # Rank by the combined score, not content similarity alone.
    # Scan the full ranked candidate list so duplicate product names do not
    # consume the top_k slots.
    ranked_indices = np.argsort(hybrid_scores)[::-1]
    records = []
    seen_product_names = set()

    for position in ranked_indices:
        product_id = candidate_ids[int(position)]
        product = products.iloc[product_id]

        # Product datasets can contain multiple rows with the same displayed
        # product name. Keep only the highest-ranked row for each name.
        if NAME_COLUMN:
            product_name_key = clean_text(product.get(NAME_COLUMN, ""))
            if product_name_key:
                if product_name_key in seen_product_names:
                    continue
                seen_product_names.add(product_name_key)

        record = product.drop(
            labels=[
                "_category_clean", "_brand_clean", "_material_clean",
                "_style_clean", "_price_numeric"
            ],
            errors="ignore"
        ).to_dict()
        record["product_id"] = int(product_id)
        record["rank"] = len(records) + 1
        record["is_repeat_recommendation"] = int(product_id) in already_interacted
        record["cf_score"] = safe_float(cf_values[position])
        record["content_similarity"] = safe_float(content_values[position])
        record["ltr_score"] = safe_float(ltr_scores[position])
        record["hybrid_score"] = safe_float(hybrid_scores[position])
        records.append(record)
        if len(records) >= top_k:
            break

    return pd.DataFrame(records)


def hybrid_recommend(user_id, query=None, top_k=DEFAULT_TOP_K):
    """API-compatible function returning JSON-friendly product dictionaries."""
    try:
        results = recommend_for_user(user_id=user_id, query=query, top_k=top_k)
        if results.empty:
            return []
        results = results.replace({np.nan: None, np.inf: None, -np.inf: None})
        return results.to_dict(orient="records")
    except Exception as error:
        warnings.warn(f"Hybrid recommendation failed: {error}")
        return []


if __name__ == "__main__":
    print("\nTesting query-based hybrid recommendations...")
    test_results = hybrid_recommend(
        user_id=1, query="comfortable fabric sofa", top_k=5
    )
    if not test_results:
        print("No recommendations returned.")
    else:
        for item in test_results:
            print(
                "ID:", item.get("product_id"),
                "| Name:", item.get(NAME_COLUMN) if NAME_COLUMN else "",
                "| Category:", item.get(CATEGORY_COLUMN) if CATEGORY_COLUMN else "",
                "| Material:", item.get(MATERIAL_COLUMN) if MATERIAL_COLUMN else "",
                "| Content similarity:", item.get("content_similarity"),
                "| Hybrid score:", item.get("hybrid_score")
            )

    print("\nGenerating recommendations for all users...")
    all_results = []
    for current_user_id in user_ids:
        user_results = hybrid_recommend(
            user_id=current_user_id, query="", top_k=DEFAULT_TOP_K
        )
        for item in user_results:
            item["user_id"] = int(current_user_id)
            all_results.append(item)

    if all_results:
        pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False)
        print(f"Saved recommendations to: {OUTPUT_PATH}")
    else:
        print("No recommendations were generated.")
