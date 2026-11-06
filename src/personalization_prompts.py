# ============================================================
# PERSONALIZATION PROMPTS
# HyperPersonalized AI Recommender System
# Sprint 3 - Week 5
# ============================================================


# ------------------------------------------------------------
# 1. STYLE-FOCUSED PROMPT
# ------------------------------------------------------------

STYLE_PROMPT = """
Find {style} furniture that matches the user's preferences.

User preferences:
{preferences}

Retrieved products:
{context}

Recommend only products from the retrieved products.
Do not invent product details.
"""


# ------------------------------------------------------------
# 2. BUDGET-FOCUSED PROMPT
# ------------------------------------------------------------

BUDGET_PROMPT = """
Recommend furniture within the user's budget of ₹{budget}.

User preferences:
{preferences}

Retrieved products:
{context}

Prioritize products that satisfy the budget.
Use only the products provided.
Do not invent prices or product information.
"""


# ------------------------------------------------------------
# 3. CATEGORY-FOCUSED PROMPT
# ------------------------------------------------------------

CATEGORY_PROMPT = """
Recommend {category} products based on the user's preferences.

User preferences:
{preferences}

Retrieved products:
{context}

Recommend only matching products from the retrieved products.
Do not invent products or product details.
"""


# ------------------------------------------------------------
# 4. COLOR-FOCUSED PROMPT
# ------------------------------------------------------------

COLOR_PROMPT = """
Find furniture in {color} color that matches the user's preferences.

User preferences:
{preferences}

Retrieved products:
{context}

Recommend only products present in the retrieved products.
Do not invent colors, prices, brands, or other details.
"""


# ------------------------------------------------------------
# 5. MATERIAL-FOCUSED PROMPT
# ------------------------------------------------------------

MATERIAL_PROMPT = """
Recommend furniture made from {material} based on the user's preferences.

User preferences:
{preferences}

Retrieved products:
{context}

Use only the retrieved products.
Do not invent material information.
"""


# ------------------------------------------------------------
# 6. BRAND-FOCUSED PROMPT
# ------------------------------------------------------------

BRAND_PROMPT = """
Recommend {brand} furniture that matches the user's requirements.

User preferences:
{preferences}

Retrieved products:
{context}

Recommend only products from the retrieved products.
Do not invent products or brand information.
"""


# ------------------------------------------------------------
# 7. ROOM-FOCUSED PROMPT
# ------------------------------------------------------------

ROOM_PROMPT = """
Recommend furniture suitable for a {room_type}.

User preferences:
{preferences}

Retrieved products:
{context}

Consider the user's stated preferences while recommending.
Use only the retrieved products.
Do not invent product details.
"""


# ------------------------------------------------------------
# 8. PRICE-RANGE PROMPT
# ------------------------------------------------------------

PRICE_RANGE_PROMPT = """
Recommend furniture between ₹{min_price} and ₹{max_price}.

User preferences:
{preferences}

Retrieved products:
{context}

Only recommend products that satisfy the given price range.
Do not invent prices or product information.
"""


# ------------------------------------------------------------
# 9. CROSS-SELLING PROMPT
# ------------------------------------------------------------

CROSS_SELL_PROMPT = """
The user already has or purchased:

{purchased}

Recommend complementary furniture that could go well with
the user's existing products.

User preferences:
{preferences}

Retrieved products:
{context}

Recommend only products present in the retrieved products.
Do not invent products or compatibility information.
"""


# ------------------------------------------------------------
# 10. PERSONALIZED USER-PREFERENCE PROMPT
# ------------------------------------------------------------

PERSONALIZED_PROMPT = """
Provide personalized furniture recommendations based on the
user's preferences.

User persona:
{persona}

User request:
{query}

Retrieved products:
{context}

Prioritize products that best match the user's stated
preferences.

Recommend only products from the retrieved products.
Do not invent product details.
"""


# ------------------------------------------------------------
# 11. MULTI-PREFERENCE PROMPT
# ------------------------------------------------------------

MULTI_PREFERENCE_PROMPT = """
Find furniture satisfying the user's multiple preferences.

User request:
{query}

Preferences:
Category: {category}
Color: {color}
Material: {material}
Style: {style}
Brand: {brand}
Budget: ₹{budget}

Retrieved products:
{context}

Recommend products that match as many of the user's
requirements as possible.

Use only the retrieved products.
Do not invent product details.
"""


# ------------------------------------------------------------
# 12. BEST-MATCH PROMPT
# ------------------------------------------------------------

BEST_MATCH_PROMPT = """
Identify the products that best match the user's request.

User request:
{query}

User preferences:
{preferences}

Retrieved products:
{context}

Explain briefly why each recommended product matches
the user's requirements.

Use only information available in the retrieved products.
Do not invent product details.
"""


# ------------------------------------------------------------
# 13. ALTERNATIVE PRODUCT PROMPT
# ------------------------------------------------------------

ALTERNATIVE_PROMPT = """
The user is looking for an alternative to:

{current_product}

User preferences:
{preferences}

Retrieved products:
{context}

Suggest alternative products from the retrieved products
that have similar characteristics.

Do not invent product details.
"""


# ------------------------------------------------------------
# 14. CONSTRAINT-BASED PROMPT
# ------------------------------------------------------------

CONSTRAINT_PROMPT = """
Recommend furniture satisfying all available constraints.

User request:
{query}

Constraints:
Category: {category}
Color: {color}
Material: {material}
Style: {style}
Brand: {brand}
Minimum price: ₹{min_price}
Maximum price: ₹{max_price}

Retrieved products:
{context}

Only recommend products that satisfy the constraints.
If no product satisfies all constraints, clearly state that.

Do not invent products or product information.
"""


# ------------------------------------------------------------
# 15. CONVERSATIONAL PERSONALIZATION PROMPT
# ------------------------------------------------------------

CONVERSATIONAL_PROMPT = """
Continue helping the user based on the previous conversation.

Previous conversation:
{conversation}

Current user request:
{query}

Current user preferences:
{preferences}

Retrieved products:
{context}

Maintain the user's previously stated preferences unless
the current request changes them.

Recommend only products available in the retrieved products.

Do not invent product names, prices, brands, colors,
materials, dimensions, or other product information.
"""


# ============================================================
# PROMPT COLLECTION
# ============================================================

PROMPT_TEMPLATES = {

    "style": STYLE_PROMPT,

    "budget": BUDGET_PROMPT,

    "category": CATEGORY_PROMPT,

    "color": COLOR_PROMPT,

    "material": MATERIAL_PROMPT,

    "brand": BRAND_PROMPT,

    "room": ROOM_PROMPT,

    "price_range": PRICE_RANGE_PROMPT,

    "cross_sell": CROSS_SELL_PROMPT,

    "personalized": PERSONALIZED_PROMPT,

    "multi_preference": MULTI_PREFERENCE_PROMPT,

    "best_match": BEST_MATCH_PROMPT,

    "alternative": ALTERNATIVE_PROMPT,

    "constraint": CONSTRAINT_PROMPT,

    "conversational": CONVERSATIONAL_PROMPT
}


# ============================================================
# GET PROMPT
# ============================================================

def get_prompt(prompt_type):

    if prompt_type not in PROMPT_TEMPLATES:

        raise ValueError(
            f"Unknown prompt type: {prompt_type}"
        )

    return PROMPT_TEMPLATES[prompt_type]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PERSONALIZATION PROMPT TEMPLATES")
    print("=" * 70)

    print("\nTotal prompt templates:",
          len(PROMPT_TEMPLATES))

    print("\nAvailable prompts:")

    for i, name in enumerate(
        PROMPT_TEMPLATES.keys(),
        start=1
    ):

        print(f"{i}. {name}")

    print("\nPersonalization prompt setup completed.")