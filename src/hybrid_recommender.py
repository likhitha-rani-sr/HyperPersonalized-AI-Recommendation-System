
"""
Query-aware hybrid recommendation adapter.

Reuses the existing content recommender, ALS candidate
generation, and trained Optuna LTR model.

Product IDs returned by this module are zero-based,
matching the API catalog.
"""

import logging
import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hybrid_recommendations.csv"
)

TOP_K = 10
logger = logging.getLogger(__name__)


def hybrid_recommend(user_id, query=None, top_k=TOP_K):
    """
    Generate personalized, query-aware recommendations.

    The existing content recommender handles:
    - Semantic query matching
    - ALS collaborative filtering
    - The saved Optuna LTR model
    - The existing seven LTR features

    Results use zero-based product IDs.
    """
    try:
        from src.content_recommender import recommend_for_user

        user_id = int(user_id)
        top_k = max(1, min(int(top_k), 20))
        query = str(query or "").strip()

        results = recommend_for_user(
            user_id=user_id,
            query=query,
            top_k=top_k
        )

        if results is None:
            return []

        if isinstance(results, pd.DataFrame):
            records = results.to_dict(orient="records")
        elif isinstance(results, list):
            records = results
        elif isinstance(results, dict):
            records = results.get("products", [results])
        else:
            logger.warning(
                "Unexpected recommendation result type: %s",
                type(results).__name__
            )
            return []

        safe_records = []

        for record in records:
            if not isinstance(record, dict):
                continue

            safe_record = {}

            for key, value in record.items():
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except (ValueError, TypeError):
                        pass

                if pd.isna(value) if not isinstance(
                    value, (dict, list, tuple)
                ) else False:
                    value = None

                safe_record[key] = value

            if safe_record.get("product_id") is not None:
                try:
                    safe_record["product_id"] = int(
                        safe_record["product_id"]
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Skipping result with invalid product ID."
                    )
                    continue

            safe_records.append(safe_record)

        return safe_records[:top_k]

    except Exception:
        logger.exception(
            "Query-aware hybrid recommendation failed."
        )
        return []


if __name__ == "__main__":
    sample_results = hybrid_recommend(
        user_id=1,
        query="comfortable fabric sofa",
        top_k=5
    )

    print(pd.DataFrame(sample_results).to_string(index=False))
