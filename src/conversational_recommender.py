import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_processor import process_query
from rag_context import build_rag_context


class ConversationState:
    """
    Stores preferences from the previous user query.
    """

    def __init__(self):
        self.preferences = None

    def reset(self):
        self.preferences = None


def merge_preferences(previous, new):
    """
    Merge new preferences with previous preferences.

    Values provided in the new query replace old values.
    Values not provided in the new query remain unchanged.
    """

    if previous is None:
        return new.copy()

    merged = previous.copy()

    for key, value in new.items():

        if value is not None:
            merged[key] = value

    return merged


def build_combined_query(preferences):
    """
    Convert conversation preferences back into a query
    that can be processed by the existing RAG pipeline.
    """

    parts = []

    style = preferences.get("style")
    color = preferences.get("color")
    material = preferences.get("material")
    category = preferences.get("category")
    brand = preferences.get("brand")

    min_price = preferences.get("min_price")
    max_price = preferences.get("max_price")

    top_k = preferences.get("top_k", 5)

    # Style
    if style:
        parts.append(style)

    # Color
    if color:
        parts.append(color)

    # Material
    if material:
        parts.append(material)

    # Category
    if category:
        parts.append(category)
    else:
        parts.append("products")

    # Brand
    if brand:
        parts.append("from " + brand)

    # Price
    if min_price is not None and max_price is not None:

        parts.append(
            f"between {int(min_price)} and {int(max_price)}"
        )

    elif max_price is not None:

        parts.append(
            f"under {int(max_price)}"
        )

    elif min_price is not None:

        parts.append(
            f"above {int(min_price)}"
        )

    # Number of results
    parts.append(
        f"show top {int(top_k)}"
    )

    return " ".join(parts)


def extract_products_from_rag_result(result):
    """
    Extract the product list from the existing RAG context result.

    Your current build_rag_context() returns 3 values:
        context_text
        products
        query_info

    This function handles that structure.
    """

    if isinstance(result, tuple):

        if len(result) == 3:

            context_text = result[0]
            products = result[1]
            query_info = result[2]

            return context_text, products, query_info

        elif len(result) == 2:

            context_text = result[0]
            products = result[1]

            return context_text, products, None

    # Safety fallback
    return str(result), [], None


def generate_response(query):
    """
    Generate a natural-language recommendation response.
    """

    result = build_rag_context(
        query,
        top_k=5
    )

    context_text, products, query_info = (
        extract_products_from_rag_result(result)
    )

    if not products:

        return (
            "I could not find products matching your current "
            "preferences. Try changing the category, price range, "
            "brand, color, or material."
        )

    response = []

    response.append(
        f"I found {len(products)} products matching your request."
    )

    for index, product in enumerate(products, start=1):

        product_name = product.get(
            "product_name",
            "Unknown Product"
        )

        brand = product.get(
            "brand",
            "Unknown"
        )

        color = product.get(
            "color",
            "Unknown"
        )

        material = product.get(
            "material",
            "Unknown"
        )

        price = product.get(
            "price",
            "Unknown"
        )

        try:

            numeric_price = float(price)

            formatted_price = (
                f"₹{numeric_price:,.0f}"
            )

        except (ValueError, TypeError):

            formatted_price = f"₹{price}"

        response.append(
            f"{index}. {product_name} from {brand}, "
            f"{color} {material}, "
            f"priced at {formatted_price}."
        )

    return "\n".join(response)


def conversational_search(state, user_query):
    """
    Process a new query or a follow-up query.
    """

    new_preferences = process_query(
        user_query
    )

    # ---------------------------------------------------------
    # FIRST QUERY
    # ---------------------------------------------------------

    if state.preferences is None:

        state.preferences = new_preferences

    # ---------------------------------------------------------
    # FOLLOW-UP QUERY
    # ---------------------------------------------------------

    else:

        state.preferences = merge_preferences(
            state.preferences,
            new_preferences
        )

    # Build the complete query using the conversation state
    combined_query = build_combined_query(
        state.preferences
    )

    print("\nConversation State:")
    print(state.preferences)

    print("\nCombined Query:")
    print(combined_query)

    return generate_response(
        combined_query
    )


def run_test():

    print("\n")
    print("=" * 70)
    print("CONVERSATIONAL FOLLOW-UP TEST")
    print("=" * 70)

    state = ConversationState()

    # =========================================================
    # FIRST QUERY
    # =========================================================

    query1 = "I want black sofas under 30000"

    print("\nUser:", query1)

    response1 = conversational_search(
        state,
        query1
    )

    print("\nAI:")
    print(response1)

    # =========================================================
    # FOLLOW-UP 1
    # =========================================================

    query2 = "Only IKEA"

    print("\nUser:", query2)

    response2 = conversational_search(
        state,
        query2
    )

    print("\nAI:")
    print(response2)

    # =========================================================
    # FOLLOW-UP 2
    # =========================================================

    query3 = "Make it under 20000"

    print("\nUser:", query3)

    response3 = conversational_search(
        state,
        query3
    )

    print("\nAI:")
    print(response3)

    # =========================================================
    # FOLLOW-UP 3
    # =========================================================

    query4 = "Show only 3"

    print("\nUser:", query4)

    response4 = conversational_search(
        state,
        query4
    )

    print("\nAI:")
    print(response4)

    # =========================================================

    print("\n")
    print("=" * 70)
    print("CONVERSATIONAL FOLLOW-UP TEST COMPLETED")
    print("=" * 70)


def run_interactive():

    print("\n")
    print("=" * 70)
    print("CONVERSATIONAL AI RECOMMENDER")
    print("=" * 70)

    print("\nType 'reset' to start a new search.")
    print("Type 'exit' to stop.")

    state = ConversationState()

    while True:

        user_query = input("\nUser: ").strip()

        if not user_query:
            continue

        if user_query.lower() in [
            "exit",
            "quit"
        ]:

            print("\nConversation ended.")
            break

        if user_query.lower() in [
            "reset",
            "new search",
            "start over"
        ]:

            state.reset()

            print(
                "\nAI: Conversation reset. "
                "What would you like to find?"
            )

            continue

        try:

            response = conversational_search(
                state,
                user_query
            )

            print("\nAI:")
            print(response)

        except Exception as e:

            print("\nAI: Something went wrong.")
            print("Error:", e)


if __name__ == "__main__":

    run_test()