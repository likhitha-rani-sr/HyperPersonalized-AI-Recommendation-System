# INTERACTION MATRIX
# Create meaningful synthetic user-product interactions

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# --------------------------------------------------
# 1. Load processed product data
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/processed_products.csv"
)


# --------------------------------------------------
# 2. Create product IDs
# --------------------------------------------------

df["product_id"] = np.arange(
    1,
    len(df) + 1
)


# --------------------------------------------------
# 3. Clean text columns
# --------------------------------------------------

df["category_clean"] = (
    df["category"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["brand_clean"] = (
    df["brand"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["material_clean"] = (
    df["material"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# --------------------------------------------------
# 4. Create 5 user preference profiles
# --------------------------------------------------

user_preferences = {

    1: {
        "category": "sofa",
        "brand": "ikea",
        "material": "fabric"
    },

    2: {
        "category": "bed",
        "brand": "godrej",
        "material": "wood"
    },

    3: {
        "category": "chair",
        "brand": "nilkamal",
        "material": "metal"
    },

    4: {
        "category": "table",
        "brand": "urban ladder",
        "material": "glass"
    },

    5: {
        "category": "wardrobe",
        "brand": "hometown",
        "material": "engineered wood"
    }
}


# --------------------------------------------------
# 5. Create interaction data
# --------------------------------------------------

interaction_rows = []


for user_id, preferences in user_preferences.items():

    for _, product in df.iterrows():

        score = 0

        # Category preference
        if product["category_clean"] == preferences["category"]:
            score += 3

        # Brand preference
        if product["brand_clean"] == preferences["brand"]:
            score += 2

        # Material preference
        if product["material_clean"] == preferences["material"]:
            score += 2

        # Create interaction only when
        # the product matches at least one preference
        if score > 0:

            interaction_rows.append({
                "user_id": user_id,
                "product_id": product["product_id"],
                "interaction": score
            })


# --------------------------------------------------
# 6. Create interaction DataFrame
# --------------------------------------------------

interaction_df = pd.DataFrame(
    interaction_rows
)


# --------------------------------------------------
# 7. Create User-Product Interaction Matrix
# --------------------------------------------------

interaction_matrix = interaction_df.pivot_table(
    index="user_id",
    columns="product_id",
    values="interaction",
    fill_value=0
)


print("Interaction Matrix:")
print(interaction_matrix)


# --------------------------------------------------
# 8. Convert to sparse matrix
# --------------------------------------------------

sparse_matrix = csr_matrix(
    interaction_matrix.values
)


print("\nSparse Matrix Shape:")
print(sparse_matrix.shape)


# --------------------------------------------------
# 9. Save interaction data
# --------------------------------------------------

interaction_df.to_csv(
    "data/processed/user_product_interactions.csv",
    index=False
)


print("\nInteraction data saved!")

print("\nInteraction Counts:")
print(
    interaction_df.groupby("user_id").size()
)

print("\nInteraction Values:")
print(
    interaction_df["interaction"].value_counts().sort_index()
)

print("\nInteraction Matrix Done!")