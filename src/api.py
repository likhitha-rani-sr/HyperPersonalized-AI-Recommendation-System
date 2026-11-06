from pathlib import Path
from typing import Optional
import inspect
import time
import logging

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.query_processor import process_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PRODUCT_FILE = PROCESSED_DIR / "processed_products.csv"
INTERACTION_FILE = PROCESSED_DIR / "user_product_interactions.csv"

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hyper-Personalized AI Recommendation API",
    description=(
        "FastAPI service for the Hyper-Personalized AI "
        "Recommendation System."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================
# REQUEST SCHEMA
# ============================================================

class RecommendationRequest(BaseModel):
    query: Optional[str] = ""
    top_k: int = Field(default=5, ge=1, le=20)
    user_id: int = Field(default=1, ge=1)
    category: Optional[str] = None
    brand: Optional[str] = None
    max_price: Optional[float] = Field(default=None, gt=0)

# ============================================================
# DATA LOADING
# ============================================================

def load_products():
    if not PRODUCT_FILE.exists():
        raise FileNotFoundError(
            f"Product file not found: {PRODUCT_FILE}"
        )

    df = pd.read_csv(PRODUCT_FILE)

    # Product catalog IDs are zero-based: 0 to 799.
    if "product_id" not in df.columns:
        df.insert(0, "product_id", range(len(df)))

    return df


def load_interactions():
    if not INTERACTION_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(INTERACTION_FILE)


try:
    products_df = load_products()
except Exception:
    logger.exception("Could not load product data.")
    products_df = pd.DataFrame()

try:
    interactions_df = load_interactions()
except Exception:
    logger.exception("Could not load interaction data.")
    interactions_df = pd.DataFrame()

# ============================================================
# PRODUCT HELPERS
# ============================================================

def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def format_product(product):
    if isinstance(product, pd.Series):
        product = product.to_dict()

    if not isinstance(product, dict):
        return {"product_name": str(product)}

    result = {}
    for key, value in product.items():
        try:
            result[key] = clean_value(value)
        except Exception:
            result[key] = value

    if "product_id" not in result and "id" in result:
        result["product_id"] = result["id"]

    return result


def format_products(items):
    if items is None:
        return []

    if isinstance(items, pd.DataFrame):
        items = items.to_dict(orient="records")
    elif isinstance(items, pd.Series):
        items = [items.to_dict()]
    elif isinstance(items, dict):
        if "products" in items:
            items = items["products"]
        elif "recommendations" in items:
            items = items["recommendations"]
        else:
            items = [items]

    try:
        items = list(items)
    except Exception:
        return []

    return [format_product(item) for item in items]


def find_product(product_id):
    if products_df.empty:
        return None

    try:
        product_id = int(product_id)
    except (TypeError, ValueError):
        return None

    matches = products_df[
        products_df["product_id"] == product_id
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def enrich_product_ids(items):
    result = []

    for item in format_products(items):
        product_id = item.get("product_id")

        if product_id is not None:
            original = find_product(product_id)
            if original is not None:
                merged = original.copy()
                merged.update(item)
                item = merged

        result.append(item)

    return result

# ============================================================
# FLEXIBLE MODEL CALLER
# ============================================================

def call_function(function, user_id, query, top_k):
    signature = inspect.signature(function)
    parameters = signature.parameters
    kwargs = {}

    if "user_id" in parameters:
        kwargs["user_id"] = user_id
    if "query" in parameters:
        kwargs["query"] = query
    if "top_k" in parameters:
        kwargs["top_k"] = top_k

    try:
        return function(**kwargs)
    except TypeError:
        # Try common positional signatures if keyword arguments
        # do not match the function's signature.
        attempts = [
            lambda: function(user_id, query, top_k),
            lambda: function(query, top_k),
            lambda: function(user_id, top_k),
            lambda: function(query),
            lambda: function(user_id),
            lambda: function(top_k),
        ]

        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except Exception as error:
                last_error = error

        raise last_error

# ============================================================
# POPULARITY
# Rank popular products that match the request query.
# ============================================================

def popularity_recommend(top_k, query="", category=None, brand=None, max_price=None):
    if products_df.empty:
        return []

    def diversify_by_brand(frame):
        if "brand" not in frame.columns:
            return frame.head(top_k)

        diversified = frame.copy()
        diversified["_brand_key"] = (
            diversified["brand"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        diversified["_brand_rank"] = diversified.groupby(
            "_brand_key", sort=False
        ).cumcount()
        return (
            diversified.sort_values(
                ["_brand_rank", "interaction_count", "product_id"],
                ascending=[True, False, True],
                kind="stable",
            )
            .drop(columns=["_brand_key", "_brand_rank"])
            .head(top_k)
        )

    parsed_query = process_query(query)
    candidates = products_df.copy()
    for column in ("brand", "color"):
        if column in candidates.columns:
            values = (
                candidates[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
            )
            candidates = candidates[
                values.ne("") & values.ne("unknown")
            ]
    if "price" in candidates.columns:
        candidates = candidates[
            pd.to_numeric(candidates["price"], errors="coerce").notna()
        ]

    query_filters = {
        "category": "category",
        "material": "material",
        "color": "color",
        "brand": "brand",
    }

    for query_field, product_field in query_filters.items():
        value = parsed_query[query_field]
        if value:
            if product_field not in candidates.columns:
                raise ValueError(
                    f"Cannot filter popularity recommendations: "
                    f"missing product column '{product_field}'."
                )
            candidates = candidates[
                candidates[product_field]
                .astype(str)
                .str.strip()
                .str.casefold()
                == value.strip().casefold()
            ]

    for product_field, value in (("category", category), ("brand", brand)):
        if value:
            if product_field not in candidates.columns:
                raise ValueError(
                    f"Cannot filter popularity recommendations: "
                    f"missing product column '{product_field}'."
                )
            candidates = candidates[
                candidates[product_field]
                .astype(str)
                .str.strip()
                .str.casefold()
                == value.strip().casefold()
            ]

    price_column = "price"
    effective_max_price = parsed_query["max_price"]
    if max_price is not None:
        effective_max_price = (
            min(effective_max_price, max_price)
            if effective_max_price is not None
            else max_price
        )
    if parsed_query["min_price"] is not None or effective_max_price is not None:
        if price_column not in candidates.columns:
            raise ValueError(
                "Cannot filter popularity recommendations: "
                f"missing product column '{price_column}'."
            )
        prices = pd.to_numeric(candidates[price_column], errors="coerce")
        if parsed_query["min_price"] is not None:
            candidates = candidates[prices >= parsed_query["min_price"]]
            prices = pd.to_numeric(candidates[price_column], errors="coerce")
        if effective_max_price is not None:
            candidates = candidates[prices <= effective_max_price]

    if interactions_df.empty:
        candidates["interaction_count"] = 0
        return diversify_by_brand(candidates).to_dict(orient="records")

    if "product_id" not in interactions_df.columns:
        candidates["interaction_count"] = 0
        return diversify_by_brand(candidates).to_dict(orient="records")

    interaction_counts = (
        interactions_df
        .groupby("product_id")
        .size()
        .reset_index(name="interaction_count")
    )
    interaction_counts["product_id"] = (
        pd.to_numeric(interaction_counts["product_id"], errors="raise")
        .astype(int)
        - 1
    )
    interaction_counts = interaction_counts[
        interaction_counts["product_id"].between(0, len(products_df) - 1)
    ]

    ranked = interaction_counts.sort_values(
        "interaction_count",
        ascending=False
    )

    merged = ranked.merge(
        candidates,
        on="product_id",
        how="left"
    )
    merged = merged[merged["product_name"].notna()]

    return diversify_by_brand(merged).to_dict(orient="records")


def enforce_query_relevance(
    products, query, top_k, category=None, brand=None, max_price=None
):
    parsed_query = process_query(query)
    query_filters = {
        "category": "category",
        "material": "material",
        "color": "color",
        "brand": "brand",
    }

    def matches_query(product):
        product_brand = product.get("brand")
        if (
            product_brand is None
            or not str(product_brand).strip()
            or str(product_brand).strip().casefold() == "unknown"
        ):
            return False

        color = product.get("color")
        if (
            color is None
            or not str(color).strip()
            or str(color).strip().casefold() == "unknown"
        ):
            return False

        price = pd.to_numeric(product.get("price"), errors="coerce")
        if pd.isna(price):
            return False

        for query_field, product_field in query_filters.items():
            expected = parsed_query[query_field]
            actual = product.get(product_field)
            if expected and (
                actual is None
                or str(actual).strip().casefold() != expected.strip().casefold()
            ):
                return False

        for field, expected in (("category", category), ("brand", brand)):
            actual = product.get(field)
            if expected and (
                actual is None
                or str(actual).strip().casefold() != expected.strip().casefold()
            ):
                return False

        effective_max_price = parsed_query["max_price"]
        if max_price is not None:
            effective_max_price = (
                min(effective_max_price, max_price)
                if effective_max_price is not None
                else max_price
            )
        if parsed_query["min_price"] is not None and (
            price < parsed_query["min_price"]
        ):
            return False
        if effective_max_price is not None and (
            price > effective_max_price
        ):
            return False
        return True

    relevant = []
    seen_ids = set()
    for product in products:
        if not isinstance(product, dict) or not matches_query(product):
            continue
        product_id = product.get("product_id")
        if product_id is not None and product_id in seen_ids:
            continue
        if product_id is not None:
            seen_ids.add(product_id)
        relevant.append(product)
        if len(relevant) == top_k:
            return relevant

    for product in popularity_recommend(
        top_k, query, category, brand, max_price
    ):
        if not matches_query(product):
            continue
        product_id = product.get("product_id")
        if product_id is not None and product_id in seen_ids:
            continue
        if product_id is not None:
            seen_ids.add(product_id)
        relevant.append(product)
        if len(relevant) == top_k:
            break

    return relevant

# ============================================================
# COLLABORATIVE FILTERING - ALS
# ============================================================

def collaborative_recommend(user_id, top_k):
    try:
        from src.als_model import model, user_item_matrix

        if model is None:
            logger.error("ALS model is not available.")
            return []

        # The ALS training matrix is a SciPy sparse matrix.
        # Its rows follow sorted user IDs and its columns follow
        # sorted product IDs from the interaction pivot table.
        if user_item_matrix.shape[0] == 0:
            logger.error("ALS user-item matrix is empty.")
            return []

        if user_item_matrix.shape[1] != model.item_factors.shape[0]:
            logger.error(
                "ALS matrix/model mismatch: matrix has %s items; "
                "model has %s item factors.",
                user_item_matrix.shape[1],
                model.item_factors.shape[0]
            )
            return []

        # In this project, user IDs are 1-based and ALS row
        # indices are 0-based. For user 1, use row 0.
        user_index = int(user_id) - 1

        if user_index < 0 or user_index >= user_item_matrix.shape[0]:
            logger.warning(
                "User ID %s is outside the ALS matrix.", user_id
            )
            return []

        # Get the user's sparse interaction row.
        user_items = user_item_matrix[user_index]

        recommendations = model.recommend(
            user_index,
            user_items,
            N=top_k,
            filter_already_liked_items=True
        )

        if isinstance(recommendations, tuple):
            item_ids, scores = recommendations
        else:
            item_ids = recommendations
            scores = [None] * len(item_ids)

        results = []

        # ALS item indices refer to the columns of its training
        # matrix, not directly to the product catalog IDs.
        interaction_columns = sorted(
            interactions_df["product_id"].dropna().unique().tolist()
        ) if (
            not interactions_df.empty
            and "product_id" in interactions_df.columns
        ) else list(range(1, user_item_matrix.shape[1] + 1))

        for item_index, score in zip(item_ids, scores):
            item_index = int(item_index)

            if item_index < 0 or item_index >= len(interaction_columns):
                continue

            # Interaction product IDs are 1-based; catalog IDs
            # in this API are 0-based.
            interaction_product_id = int(
                interaction_columns[item_index]
            )
            catalog_product_id = interaction_product_id - 1

            product = find_product(catalog_product_id)
            if product is None:
                continue

            product["cf_score"] = (
                float(score) if score is not None else None
            )
            results.append(product)

        return results

    except Exception:
        logger.exception("Collaborative recommendation failed.")
        return []

# ============================================================
# CONTENT-BASED EMBEDDINGS
# ============================================================


def embedding_recommend(query, user_id, top_k):
    try:
        from src.content_recommender import recommend_for_user

        result = call_function(
            recommend_for_user,
            user_id,
            query,
            top_k
        )

        return enrich_product_ids(result)

    except Exception:
        logger.exception("Embedding recommendation failed.")
        return []

# ============================================================
# HYBRID LTR
# ============================================================

def hybrid_recommendation(user_id, query, top_k):
    try:
        from src.hybrid_recommender import hybrid_recommend

        result = call_function(
            hybrid_recommend,
            user_id,
            query,
            top_k
        )

        return enrich_product_ids(result)

    except Exception:
        logger.exception(
            "Hybrid recommendation failed; using embeddings."
        )
        return embedding_recommend(query, user_id, top_k)

# ============================================================
# CONVERSATIONAL RAG
# ============================================================

def conversational_recommendation(user_id, query, top_k):
    try:
        from src.conversational_rag import recommend_for_ui

        result = call_function(
            recommend_for_ui,
            user_id,
            query,
            top_k
        )

        response_text = ""
        products = result
        alternative_products = []

        if isinstance(result, dict):
            response_text = result.get(
                "response",
                result.get("answer", result.get("message", "")),
            )
            products = result.get(
                "products", result.get("recommendations", [])
            )
            alternative_products = result.get(
                "alternative_products",
                [],
            )
        elif isinstance(result, tuple) and len(result) >= 2:
            response_text = str(result[0])
            products = result[1]

        return (
            enrich_product_ids(products),
            response_text,
            enrich_product_ids(alternative_products),
        )

    except Exception:
        logger.exception(
            "Conversational recommendation failed; using embeddings."
        )
        return embedding_recommend(query, user_id, top_k), "", []

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Hyper-Personalized AI Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "products_loaded": len(products_df)
    }

# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "products_loaded": len(products_df),
        "service": "Hyper-Personalized AI Recommendation API",
        "version": "1.0.0"
    }

# ============================================================
# POPULARITY ENDPOINT
# ============================================================

@app.post("/recommend/popularity")
def recommend_popularity(request: RecommendationRequest):
    start_time = time.perf_counter()
    products = popularity_recommend(
        request.top_k,
        request.query or "",
        request.category,
        request.brand,
        request.max_price,
    )
    products = enforce_query_relevance(
        products,
        request.query or "",
        request.top_k,
        request.category,
        request.brand,
        request.max_price,
    )
    elapsed = time.perf_counter() - start_time

    return {
        "success": True,
        "method": "Popularity",
        "algorithm": "Popularity / Interaction Frequency",
        "query": request.query,
        "user_id": request.user_id,
        "count": len(products),
        "products": products,
        "response_time_seconds": elapsed
    }

# ============================================================
# COLLABORATIVE ENDPOINT
# ============================================================

@app.post("/recommend/collaborative")
def recommend_collaborative(request: RecommendationRequest):
    start_time = time.perf_counter()
    products = collaborative_recommend(
        request.user_id,
        request.top_k
    )
    products = enforce_query_relevance(
        products,
        request.query or "",
        request.top_k,
        request.category,
        request.brand,
        request.max_price,
    )
    elapsed = time.perf_counter() - start_time

    return {
        "success": True,
        "method": "Collaborative Filtering",
        "algorithm": "Implicit ALS",
        "query": request.query,
        "user_id": request.user_id,
        "count": len(products),
        "products": products,
        "response_time_seconds": elapsed
    }

# ============================================================
# EMBEDDING ENDPOINT
# ============================================================

@app.post("/recommend/embedding")
def recommend_embedding(request: RecommendationRequest):
    start_time = time.perf_counter()
    products = embedding_recommend(
        request.query,
        request.user_id,
        request.top_k
    )
    products = enforce_query_relevance(
        products,
        request.query or "",
        request.top_k,
        request.category,
        request.brand,
        request.max_price,
    )
    elapsed = time.perf_counter() - start_time

    return {
        "success": True,
        "method": "Content-Based Embeddings",
        "algorithm": "Sentence Transformer + ChromaDB",
        "query": request.query,
        "user_id": request.user_id,
        "count": len(products),
        "products": products,
        "response_time_seconds": elapsed
    }

# ============================================================
# HYBRID ENDPOINT
# ============================================================

@app.post("/recommend/hybrid")
def recommend_hybrid(request: RecommendationRequest):
    start_time = time.perf_counter()
    products = hybrid_recommendation(
        request.user_id,
        request.query,
        request.top_k
    )
    products = enforce_query_relevance(
        products,
        request.query or "",
        request.top_k,
        request.category,
        request.brand,
        request.max_price,
    )
    elapsed = time.perf_counter() - start_time

    return {
        "success": True,
        "method": "Hybrid LTR",
        "algorithm": "ALS + Embeddings + XGBoost LTR",
        "query": request.query,
        "user_id": request.user_id,
        "count": len(products),
        "products": products,
        "response_time_seconds": elapsed
    }

# ============================================================
# CONVERSATIONAL ENDPOINT
# ============================================================

@app.post("/recommend/conversational")
def recommend_conversational(request: RecommendationRequest):
    start_time = time.perf_counter()
    products, response_text, alternative_products = conversational_recommendation(
        request.user_id,
        request.query,
        request.top_k
    )
    products = enforce_query_relevance(
        products,
        request.query or "",
        request.top_k,
        request.category,
        request.brand,
        request.max_price,
    )
    elapsed = time.perf_counter() - start_time

    return {
        "success": True,
        "method": "Conversational RAG",
        "algorithm": "RAG + Conversational Recommendation",
        "query": request.query,
        "user_id": request.user_id,
        "response": response_text,
        "count": len(products),
        "products": products,
        "alternative_count": len(alternative_products),
        "alternative_products": alternative_products,
        "response_time_seconds": elapsed
    }