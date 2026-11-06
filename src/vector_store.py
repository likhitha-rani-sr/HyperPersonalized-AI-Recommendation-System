import os
import json
import numpy as np
import chromadb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "product_embeddings.npy"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "embedding_metadata.json"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db",
    "chroma_db"
)


# ============================================================
# CHROMA COLLECTION
# ============================================================

COLLECTION_NAME = "product_embeddings"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Embeddings file not found:\n{EMBEDDINGS_FILE}\n\n"
            "Run embedder.py first."
        )

    print("\nLoading product embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}\n\n"
            "Run embedder.py first."
        )

    print("\nLoading product metadata...")

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    metadata = data["metadata"]

    print(f"Metadata records: {len(metadata)}")

    return metadata


# ============================================================
# PREPARE CHROMA METADATA
# ============================================================

def prepare_chroma_metadata(item):
    """
    Chroma metadata values must be simple scalar values.
    """

    metadata = {}

    allowed_fields = [
        "product_id",
        "product_name",
        "category",
        "brand",
        "color",
        "material",
        "style",
        "price"
    ]

    for field in allowed_fields:

        if field in item:

            value = item[field]

            if value is None:
                continue

            # Convert values to Chroma-compatible types
            if isinstance(value, (str, int, float, bool)):
                metadata[field] = value

            else:
                metadata[field] = str(value)

    return metadata


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store():

    print("=" * 60)
    print("HYPER-PERSONALIZED AI RECOMMENDER")
    print("CHROMADB VECTOR STORE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    embeddings = load_embeddings()
    metadata = load_metadata()

    # --------------------------------------------------------
    # Validate data
    # --------------------------------------------------------

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Number of embeddings does not match "
            "number of metadata records."
        )

    print("\nData validation successful.")

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    os.makedirs(
        VECTOR_DB_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create Chroma client
    # --------------------------------------------------------

    print("\nCreating ChromaDB client...")

    client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR
    )

    # --------------------------------------------------------
    # Delete old collection if it exists
    # --------------------------------------------------------

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Existing collection removed.")

    except Exception:

        print("No existing collection found.")

    # --------------------------------------------------------
    # Create collection
    # --------------------------------------------------------

    print("\nCreating ChromaDB collection...")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Product embedding vector store",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    ids = []
    documents = []
    chroma_metadatas = []
    embedding_lists = []

    print("\nPreparing products for ChromaDB...")

    for index, item in enumerate(metadata):

        # Unique Chroma ID
        product_id = str(
            item.get(
                "product_id",
                index
            )
        )

        ids.append(product_id)

        # Store embedding text as document
        documents.append(
            item.get(
                "embedding_text",
                ""
            )
        )

        # Store product metadata
        chroma_metadatas.append(
            prepare_chroma_metadata(item)
        )

        # Convert NumPy embedding to list
        embedding_lists.append(
            embeddings[index].tolist()
        )

    # --------------------------------------------------------
    # Add products to ChromaDB
    # --------------------------------------------------------

    print("\nAdding products to ChromaDB...")

    collection.add(
        ids=ids,
        embeddings=embedding_lists,
        documents=documents,
        metadatas=chroma_metadatas
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    total_products = collection.count()

    print("\n" + "=" * 60)
    print("CHROMADB VECTOR STORE CREATED")
    print("=" * 60)

    print(f"Products indexed : {total_products}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Database path    : {VECTOR_DB_DIR}")

    # --------------------------------------------------------
    # Test vector search
    # --------------------------------------------------------

    print("\nTesting vector search...")

    test_embedding = embeddings[0].tolist()

    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=5
    )

    print("\nTop 5 similar products:")

    result_ids = results["ids"][0]
    result_distances = results["distances"][0]

    for i in range(len(result_ids)):

        product_id = result_ids[i]
        distance = result_distances[i]

        print(
            f"{i + 1}. Product ID: "
            f"{product_id} | "
            f"Distance: {distance:.4f}"
        )

    print("\nVector search test successful.")

    print("\nNext step:")
    print(
        "Use content_recommender.py "
        "for content-based recommendations."
    )


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_products(
    query_embedding,
    top_k=5,
    similarity_threshold=None
):
    """
    Search the ChromaDB vector store.

    Parameters:
        query_embedding:
            Embedding vector for the search query.

        top_k:
            Number of results to return.

        similarity_threshold:
            Optional minimum cosine similarity.

    Returns:
        List of matching products.
    """

    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(
            "ChromaDB vector store not found.\n"
            "Run vector_store.py first."
        )

    client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        ],
        n_results=top_k
    )

    products = []

    result_ids = results["ids"][0]
    result_metadatas = results["metadatas"][0]
    result_distances = results["distances"][0]

    for i in range(len(result_ids)):

        distance = result_distances[i]

        # Because embeddings are normalized,
        # cosine similarity can be derived from
        # cosine distance:
        similarity = 1.0 - distance

        if (
            similarity_threshold is not None
            and similarity < similarity_threshold
        ):
            continue

        product = {
            "product_id": result_ids[i],
            "similarity": float(similarity),
            "metadata": result_metadatas[i]
        }

        products.append(product)

    return products


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    create_vector_store()