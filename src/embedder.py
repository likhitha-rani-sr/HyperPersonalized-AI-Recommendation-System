import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRODUCT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "processed_products.csv"
)

EMBEDDINGS_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

EMBEDDINGS_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "product_embeddings.npy"
)

METADATA_FILE = os.path.join(
    EMBEDDINGS_DIR,
    "embedding_metadata.json"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_column(df, possible_names):
    """
    Finds a column from a list of possible column names.
    Handles differences such as:
    Product Name
    product_name
    product name
    """

    normalized_columns = {}

    for column in df.columns:
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        normalized_columns[normalized] = column

    for name in possible_names:
        normalized_name = (
            name.strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


# ============================================================
# CREATE PRODUCT DESCRIPTION
# ============================================================

def create_product_description(row, columns):
    """
    Creates the text representation used for embeddings.

    Required Sprint 2 idea:
    style + material + category + color + price

    Additional available product information is also included
    when present.
    """

    parts = []

    # Product name
    if columns["name"]:
        value = row[columns["name"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Product: {value}")

    # Style
    if columns["style"]:
        value = row[columns["style"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Style: {value}")

    # Category
    if columns["category"]:
        value = row[columns["category"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Category: {value}")

    # Material
    if columns["material"]:
        value = row[columns["material"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Material: {value}")

    # Color
    if columns["color"]:
        value = row[columns["color"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Color: {value}")

    # Price
    if columns["price"]:
        value = row[columns["price"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Price: {value}")

    # Existing product description
    if columns["description"]:
        value = row[columns["description"]]

        if pd.notna(value):
            value = str(value).strip()

            if value and value.lower() != "unknown":
                parts.append(f"Description: {value}")

    return ". ".join(parts)


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_embeddings():

    print("=" * 60)
    print("HYPER-PERSONALIZED AI RECOMMENDER")
    print("PRODUCT EMBEDDING GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check product file
    # --------------------------------------------------------

    if not os.path.exists(PRODUCT_FILE):
        raise FileNotFoundError(
            f"\nProcessed product file not found:\n{PRODUCT_FILE}"
        )

    print("\nLoading processed product data...")
    df = pd.read_csv(PRODUCT_FILE)

    print(f"Products loaded: {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    if df.empty:
        raise ValueError("The processed product dataset is empty.")

    # --------------------------------------------------------
    # Find columns
    # --------------------------------------------------------

    columns = {
        "name": find_column(
            df,
            [
                "Product Name",
                "product_name",
                "product name",
                "name"
            ]
        ),

        "category": find_column(
            df,
            [
                "Category",
                "category"
            ]
        ),

        "brand": find_column(
            df,
            [
                "Brand",
                "brand"
            ]
        ),

        "color": find_column(
            df,
            [
                "Color",
                "color"
            ]
        ),

        "material": find_column(
            df,
            [
                "Material",
                "material"
            ]
        ),

        "style": find_column(
            df,
            [
                "Style",
                "style"
            ]
        ),

        "price": find_column(
            df,
            [
                "Price",
                "price"
            ]
        ),

        "description": find_column(
            df,
            [
                "Product Description",
                "product_description",
                "product description",
                "Description",
                "description"
            ]
        )
    }

    print("\nDetected columns:")

    for key, value in columns.items():
        print(f"  {key:12} -> {value}")

    # --------------------------------------------------------
    # Create product descriptions
    # --------------------------------------------------------

    print("\nCreating product descriptions...")

    descriptions = []

    for _, row in df.iterrows():

        description = create_product_description(
            row,
            columns
        )

        # Fallback if no expected fields were found
        if not description.strip():
            description = "Furniture product"

        descriptions.append(description)

    df["embedding_text"] = descriptions

    print("Product descriptions created.")

    # Show first few descriptions
    print("\nSample product descriptions:")

    for i, description in enumerate(descriptions[:5]):
        print(f"\nProduct {i + 1}:")
        print(description)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\nEmbeddings generated.")

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------------
    # Create embeddings directory
    # --------------------------------------------------------

    os.makedirs(
        EMBEDDINGS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    print(
        f"\nEmbeddings saved to:\n{EMBEDDINGS_FILE}"
    )

    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    metadata = []

    for index, row in df.iterrows():

        item = {
            "embedding_index": int(index),
            "product_id": int(index)
        }

        # Product name
        if columns["name"]:
            value = row[columns["name"]]

            if pd.notna(value):
                item["product_name"] = str(value)

        # Category
        if columns["category"]:
            value = row[columns["category"]]

            if pd.notna(value):
                item["category"] = str(value)

        # Brand
        if columns["brand"]:
            value = row[columns["brand"]]

            if pd.notna(value):
                item["brand"] = str(value)

        # Color
        if columns["color"]:
            value = row[columns["color"]]

            if pd.notna(value):
                item["color"] = str(value)

        # Material
        if columns["material"]:
            value = row[columns["material"]]

            if pd.notna(value):
                item["material"] = str(value)

        # Style
        if columns["style"]:
            value = row[columns["style"]]

            if pd.notna(value):
                item["style"] = str(value)

        # Price
        if columns["price"]:
            value = row[columns["price"]]

            if pd.notna(value):
                try:
                    item["price"] = float(value)
                except (ValueError, TypeError):
                    item["price"] = str(value)

        # Store embedding text
        item["embedding_text"] = descriptions[index]

        metadata.append(item)

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_output = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_products": int(len(df)),
        "normalized": True,
        "metadata": metadata
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata_output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Metadata saved to:\n{METADATA_FILE}"
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 60)

    print(f"Total products      : {len(df)}")
    print(f"Embedding dimension : {embeddings.shape[1]}")
    print(f"Embedding model     : {MODEL_NAME}")
    print(f"Embeddings file     : {EMBEDDINGS_FILE}")
    print(f"Metadata file       : {METADATA_FILE}")

    print("\nNext step:")
    print("Use vector_store.py to create the ChromaDB vector store.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_embeddings()