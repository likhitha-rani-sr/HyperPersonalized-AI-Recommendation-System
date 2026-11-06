from langchain_ollama import ChatOllama
from rag_context import build_rag_context


MODEL_NAME = "phi3"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0
)


def create_product_context(products):
    """
    Convert retrieved products into a simple,
    LLM-friendly format.
    """

    lines = []

    for i, product in enumerate(products, start=1):

        product_name = str(
            product.get("product_name", "Unknown")
        )

        category = str(
            product.get("category", "Unknown")
        )

        brand = str(
            product.get("brand", "Unknown")
        )

        color = str(
            product.get("color", "Unknown")
        )

        material = str(
            product.get("material", "Unknown")
        )

        price = product.get("price", "Unknown")

        lines.append(
            f"""
Product {i}
Name: {product_name}
Category: {category}
Brand: {brand}
Color: {color}
Material: {material}
Price: ₹{price}
"""
        )

    return "\n".join(lines)


def create_prompt(query, products):

    product_context = create_product_context(products)

    return f"""
You are a furniture recommendation assistant.

USER REQUEST:
{query}

AVAILABLE PRODUCTS:
{product_context}

IMPORTANT RULES:

1. Recommend ONLY products listed above.
2. Do NOT create or invent any product.
3. Do NOT invent prices.
4. Do NOT invent brands.
5. Do NOT invent dimensions.
6. Do NOT invent features.
7. Do NOT add information that is not provided above.
8. Use the exact product names and prices provided.
9. If a product matches the user's request, recommend it.
10. If multiple products match, list them.
11. If no products match, clearly say that no matching product was found.
12. Keep the response concise.

Answer the user's request using ONLY the available products.
"""


def generate_recommendation(query):

    print("\n" + "=" * 70)
    print("RAG + OLLAMA RECOMMENDER")
    print("=" * 70)

    print("\nUser Query:")
    print(query)

    # Existing RAG retrieval
    context_text, products, query_info = build_rag_context(
        query,
        top_k=5
    )

    print("\nProducts retrieved:", len(products))

    if not products:
        print("\nNo matching products found.")
        return

    # Convert retrieved products into simple LLM context
    prompt = create_prompt(
        query,
        products
    )

    print("\nGenerating response using Phi-3...")

    response = llm.invoke(prompt)

    print("\n" + "-" * 70)
    print("AI RECOMMENDATION")
    print("-" * 70)

    print(response.content)

    print("\n" + "=" * 70)
    print("RAG + OLLAMA TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    query = "I want a modern black sofa under 30000 rupees"

    generate_recommendation(query)