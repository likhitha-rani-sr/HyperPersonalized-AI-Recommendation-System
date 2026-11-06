import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "vector_db/chroma_db"
COLLECTION_NAME = "product_embeddings"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.30


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

class SentenceTransformerEmbeddings:
    """
    Small LangChain-compatible embedding wrapper
    using the same model used in Week 3.
    """

    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding.tolist()


# ============================================================
# CREATE LANGCHAIN CHROMA STORE
# ============================================================

def create_vector_store():

    print("\nConnecting LangChain to existing ChromaDB...")

    embeddings = SentenceTransformerEmbeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    print("ChromaDB connection successful.")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Database   : {CHROMA_PATH}")

    return vector_store


# ============================================================
# RETRIEVER
# ============================================================

def create_retriever(
    vector_store,
    top_k=DEFAULT_TOP_K,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
):
    """
    Creates a LangChain retriever with:
    - top_k
    - similarity threshold
    """

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "score_threshold": similarity_threshold
        }
    )

    return retriever


# ============================================================
# SEARCH
# ============================================================

def search_products(
    query,
    top_k=DEFAULT_TOP_K,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
):

    vector_store = create_vector_store()

    retriever = create_retriever(
        vector_store,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

    documents = retriever.invoke(query)

    return documents


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(query, documents):

    print("\n" + "=" * 70)
    print("LANGCHAIN RAG RETRIEVER")
    print("=" * 70)

    print(f"\nQuery: {query}")
    print(f"Products retrieved: {len(documents)}")

    if not documents:
        print("\nNo products matched the similarity threshold.")
        return

    for i, document in enumerate(documents, start=1):

        print(f"\n{i}. Product")

        print("-" * 50)

        print("Content:")
        print(document.page_content)

        print("\nMetadata:")

        for key, value in document.metadata.items():
            print(f"  {key}: {value}")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [
        "modern black sofa",
        "IKEA wooden table"
    ]

    for query in test_queries:

        results = search_products(
            query=query,
            top_k=5,
            similarity_threshold=0.30
        )

        display_results(
            query,
            results
        )

    print("\n" + "=" * 70)
    print("LANGCHAIN VECTOR RETRIEVER TEST COMPLETED")
    print("=" * 70)