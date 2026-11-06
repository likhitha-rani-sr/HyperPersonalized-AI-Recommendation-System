import sys
import os

# Allow importing files from src/
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from rag_recommender import retrieve_products


# ============================================================
# RAG CONTEXT BUILDER
# ============================================================

def build_rag_context(
    query,
    top_k=5
):
    """
    Retrieve products for a user query
    and convert them into LLM-ready RAG context.
    """

    # --------------------------------------------------------
    # Retrieve products
    # --------------------------------------------------------

    parsed_query, results = retrieve_products(
        query,
        top_k=top_k
    )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        context = f"""
User Request:
{query}

No matching products were found.
"""

        return parsed_query, results, context

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    context_parts.append(
        "USER REQUEST:"
    )

    context_parts.append(
        query
    )

    context_parts.append(
        "\nRETRIEVED PRODUCTS:"
    )

    for index, product in enumerate(
        results,
        start=1
    ):

        product_text = f"""
Product {index}:
Product ID: {product['product_id']}
Product Name: {product['product_name']}
Category: {product['category']}
Brand: {product['brand']}
Color: {product['color']}
Material: {product['material']}
Price: ₹{product['price']}
Semantic Similarity: {product['similarity']:.4f}
Style Relevance: {product['style_relevance']:.1f}
"""

        context_parts.append(
            product_text
        )

    # --------------------------------------------------------
    # Final context
    # --------------------------------------------------------

    context = "\n".join(
        context_parts
    )

    return (
        parsed_query,
        results,
        context
    )


# ============================================================
# DISPLAY RAG CONTEXT
# ============================================================

def display_rag_context(
    query
):

    print()
    print("=" * 60)
    print("RAG CONTEXT GENERATION")
    print("=" * 60)

    print()
    print(
        f"User Query:\n{query}"
    )

    parsed_query, results, context = (
        build_rag_context(
            query
        )
    )

    print()
    print(
        "Retrieved Products:"
    )

    print(
        f"Count: {len(results)}"
    )

    print()
    print(
        "=" * 60
    )

    print(
        "GENERATED RAG CONTEXT"
    )

    print(
        "=" * 60
    )

    print(
        context
    )

    print(
        "=" * 60
    )

    return context


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "I want a modern black sofa under 30000",

        "Show me an IKEA wooden table below 10000",

        "I need a grey fabric chair between 5000 and 15000"
    ]

    for query in test_queries:

        display_rag_context(
            query
        )

    print()
    print("=" * 60)
    print(
        "RAG CONTEXT GENERATION TEST COMPLETED."
    )
    print("=" * 60)