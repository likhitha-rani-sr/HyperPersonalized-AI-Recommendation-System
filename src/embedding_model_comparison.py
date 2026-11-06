import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PRODUCT_FILE = "data/processed/processed_products.csv"

RESULT_DIR = "embeddings/evaluation"
os.makedirs(RESULT_DIR, exist_ok=True)

K_VALUES = [5, 10]

MODEL_1 = "all-MiniLM-L6-v2"
MODEL_2 = "all-mpnet-base-v2"


# ============================================================
# LOAD PRODUCTS
# ============================================================

print("=" * 65)
print("EMBEDDING MODEL COMPARISON")
print("=" * 65)

products = pd.read_csv(PRODUCT_FILE)

print(f"\nProducts loaded : {len(products)}")


# ============================================================
# FIND CATEGORY COLUMN
# ============================================================

category_column = None

for column in products.columns:
    if column.lower() in [
        "category",
        "product_category",
        "product_type"
    ]:
        category_column = column
        break

if category_column is None:
    raise ValueError("Category column not found.")

print(f"Category column : {category_column}")


# ============================================================
# CREATE PRODUCT TEXT
# ============================================================

def find_column(possible_names):

    for column in products.columns:
        if column.lower() in [
            name.lower() for name in possible_names
        ]:
            return column

    return None


name_col = find_column([
    "product_name",
    "productname",
    "name",
    "title"
])

brand_col = find_column([
    "brand",
    "brand_name"
])

color_col = find_column([
    "color",
    "colour"
])

material_col = find_column([
    "material"
])

style_col = find_column([
    "style"
])

price_col = find_column([
    "price",
    "product_price"
])


def get_value(row, column):

    if column is None:
        return ""

    value = row[column]

    if pd.isna(value):
        return ""

    return str(value)


product_texts = []

for _, row in products.iterrows():

    text = f"""
    Product: {get_value(row, name_col)}
    Category: {get_value(row, category_column)}
    Brand: {get_value(row, brand_col)}
    Color: {get_value(row, color_col)}
    Material: {get_value(row, material_col)}
    Style: {get_value(row, style_col)}
    Price: {get_value(row, price_col)}
    """

    product_texts.append(
        " ".join(text.split())
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_embeddings(
    embeddings,
    model_name
):

    print("\n" + "-" * 65)
    print(f"Evaluating: {model_name}")
    print("-" * 65)

    # Normalize embeddings
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    normalized = embeddings / norms

    # Cosine similarity
    similarity_matrix = np.matmul(
        normalized,
        normalized.T
    )

    results = []

    categories = (
        products[category_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for k in K_VALUES:

        precision_scores = []
        recall_scores = []

        for i in range(len(products)):

            query_category = categories.iloc[i]

            total_relevant = (
                (categories == query_category).sum() - 1
            )

            if total_relevant <= 0:
                continue

            scores = similarity_matrix[i].copy()

            # Remove itself
            scores[i] = -1

            top_indices = np.argsort(
                scores
            )[-k:][::-1]

            relevant_count = 0

            for index in top_indices:

                if categories.iloc[index] == query_category:
                    relevant_count += 1

            precision = relevant_count / k

            recall = min(
                relevant_count / total_relevant,
                1.0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = np.mean(
            precision_scores
        )

        avg_recall = np.mean(
            recall_scores
        )

        results.append({
            "model": model_name,
            "k": k,
            "precision": avg_precision,
            "recall": avg_recall
        })

        print(
            f"K={k} | "
            f"Precision={avg_precision:.4f} | "
            f"Recall={avg_recall:.4f}"
        )

    return results


# ============================================================
# MODEL 1
# ============================================================

print("\n" + "=" * 65)
print(f"Loading model 1: {MODEL_1}")
print("=" * 65)

model1 = SentenceTransformer(MODEL_1)

print("Generating embeddings...")

embeddings1 = model1.encode(
    product_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    f"Embedding shape: {embeddings1.shape}"
)

results1 = evaluate_embeddings(
    embeddings1,
    MODEL_1
)


# ============================================================
# MODEL 2
# ============================================================

print("\n" + "=" * 65)
print(f"Loading model 2: {MODEL_2}")
print("=" * 65)

model2 = SentenceTransformer(MODEL_2)

print("Generating embeddings...")

embeddings2 = model2.encode(
    product_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    f"Embedding shape: {embeddings2.shape}"
)

results2 = evaluate_embeddings(
    embeddings2,
    MODEL_2
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = results1 + results2

results_df = pd.DataFrame(
    all_results
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULT_DIR,
    "embedding_model_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE SECOND MODEL EMBEDDINGS
# ============================================================

embedding_file = os.path.join(
    RESULT_DIR,
    "mpnet_embeddings.npy"
)

np.save(
    embedding_file,
    embeddings2
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("EMBEDDING MODEL COMPARISON COMPLETED")
print("=" * 65)

print("\nResults saved:")
print(output_file)

print("\nMPNet embeddings saved:")
print(embedding_file)

print("\nWeek 3 embedding comparison completed.")