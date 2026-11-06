import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PRODUCT_FILE = "data/processed/processed_products.csv"
EMBEDDING_FILE = "embeddings/product_embeddings.npy"
METADATA_FILE = "embeddings/embedding_metadata.json"

RESULT_DIR = "embeddings/evaluation"
os.makedirs(RESULT_DIR, exist_ok=True)

K_VALUES = [5, 10]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("EMBEDDING OPTIMIZATION & EVALUATION")
print("=" * 60)

products = pd.read_csv(PRODUCT_FILE)
embeddings = np.load(EMBEDDING_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"\nProducts loaded       : {len(products)}")
print(f"Embeddings loaded     : {embeddings.shape}")
print(f"Embedding model       : {metadata.get('model', 'Unknown')}")


# ============================================================
# FIND CATEGORY COLUMN
# ============================================================

category_column = None

for column in products.columns:
    if column.lower() in ["category", "product_category", "product_type"]:
        category_column = column
        break

if category_column is None:
    raise ValueError(
        "Category column not found in processed_products.csv"
    )

print(f"Category column       : {category_column}")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1

normalized_embeddings = embeddings / norms


# ============================================================
# CATEGORY-BASED CONTENT RETRIEVAL EVALUATION
# ============================================================

def evaluate_embeddings(embedding_matrix, name):
    """
    Category-based proxy evaluation.

    A retrieved product is considered relevant when it belongs
    to the same category as the query product.

    The query product itself is excluded.
    """

    embedding_matrix = embedding_matrix.astype(np.float32)

    norms = np.linalg.norm(
        embedding_matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    normalized = embedding_matrix / norms

    similarity_matrix = np.matmul(
        normalized,
        normalized.T
    )

    results = []

    for k in K_VALUES:

        precision_scores = []
        recall_scores = []

        for i in range(len(products)):

            query_category = str(
                products.iloc[i][category_column]
            ).strip().lower()

            # Number of relevant products excluding query itself
            total_relevant = (
                products[category_column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(query_category)
                .sum()
                - 1
            )

            if total_relevant <= 0:
                continue

            # Remove the query product itself
            similarity_scores = similarity_matrix[i].copy()
            similarity_scores[i] = -1

            top_indices = np.argsort(
                similarity_scores
            )[-k:][::-1]

            relevant_count = 0

            for index in top_indices:

                candidate_category = str(
                    products.iloc[index][category_column]
                ).strip().lower()

                if candidate_category == query_category:
                    relevant_count += 1

            precision = relevant_count / k

            recall = relevant_count / total_relevant

            # Recall cannot exceed 1
            recall = min(recall, 1.0)

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)

        results.append({
            "method": name,
            "k": k,
            "precision": avg_precision,
            "recall": avg_recall
        })

    return results


# ============================================================
# BASELINE MODEL EVALUATION
# ============================================================

print("\n" + "-" * 60)
print("1. BASELINE EMBEDDING EVALUATION")
print("-" * 60)

baseline_results = evaluate_embeddings(
    normalized_embeddings,
    "all-MiniLM-L6-v2"
)

for result in baseline_results:

    print(
        f"K={result['k']} | "
        f"Precision={result['precision']:.4f} | "
        f"Recall={result['recall']:.4f}"
    )


# ============================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "-" * 60)
print("2. PCA DIMENSIONALITY REDUCTION")
print("-" * 60)

original_dimension = embeddings.shape[1]

target_dimension = min(128, original_dimension)

pca = PCA(
    n_components=target_dimension,
    random_state=42
)

reduced_embeddings = pca.fit_transform(
    normalized_embeddings
)

explained_variance = pca.explained_variance_ratio_.sum()

print(f"Original dimension   : {original_dimension}")
print(f"Reduced dimension    : {target_dimension}")
print(
    f"Explained variance   : "
    f"{explained_variance:.4f}"
)

pca_results = evaluate_embeddings(
    reduced_embeddings,
    f"MiniLM-PCA-{target_dimension}"
)

for result in pca_results:

    print(
        f"K={result['k']} | "
        f"Precision={result['precision']:.4f} | "
        f"Recall={result['recall']:.4f}"
    )


# ============================================================
# COMPARE RESULTS
# ============================================================

all_results = baseline_results + pca_results

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULT_DIR,
    "embedding_evaluation_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

# Save PCA model information
pca_info = {
    "original_dimension": int(original_dimension),
    "reduced_dimension": int(target_dimension),
    "explained_variance_ratio": float(explained_variance)
}

pca_info_file = os.path.join(
    RESULT_DIR,
    "pca_info.json"
)

with open(
    pca_info_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        pca_info,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING EVALUATION COMPLETED")
print("=" * 60)

print(f"\nResults saved to:")
print(results_file)

print(f"\nPCA information saved to:")
print(pca_info_file)

print("\nDone.")