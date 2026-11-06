# RECOMMENDATION OUTPUT

import pandas as pd

# Import the trained ALS model and interaction matrix
from als_model import model, interaction_matrix, user_item_matrix


# --------------------------------------------------
# 1. Select User
# --------------------------------------------------

user_id = 3


# --------------------------------------------------
# 2. Check whether user exists
# --------------------------------------------------

if user_id not in interaction_matrix.index:

    print("User ID not found!")

else:

    # --------------------------------------------------
    # 3. Get user index
    # --------------------------------------------------

    user_index = list(interaction_matrix.index).index(user_id)


    # --------------------------------------------------
    # 4. Generate Top 5 Recommendations
    # --------------------------------------------------

    recommendation_ids, recommendation_scores = model.recommend(
        user_index,
        user_item_matrix[user_index],
        N=5
    )


    # --------------------------------------------------
    # 5. Load Product Data
    # --------------------------------------------------

    products = pd.read_csv(
        "data/processed/processed_products.csv"
    )


    # Create product IDs
    products["product_id"] = range(
        1,
        len(products) + 1
    )


    # --------------------------------------------------
    # 6. Create Recommendation Output
    # --------------------------------------------------

    recommendation_output = []


    for item_index, score in zip(
        recommendation_ids,
        recommendation_scores
    ):

        product_id = interaction_matrix.columns[item_index]

        # Find product details
        product = products[
            products["product_id"] == product_id
        ]


        if not product.empty:

            product = product.iloc[0]


            recommendation_output.append({

                "user_id": user_id,

                "product_id": product_id,

                "product_name": product["product_name"],

                "category": product["category"],

                "brand": product["brand"],

                "material": product["material"],

                "price": product["price"],

                "recommendation_score": score

            })


    # --------------------------------------------------
    # 7. Convert to DataFrame
    # --------------------------------------------------

    recommendation_df = pd.DataFrame(
        recommendation_output
    )


    # --------------------------------------------------
    # 8. Display Recommendations
    # --------------------------------------------------

    print("\nRecommendations for User", user_id)

    print(
        recommendation_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # 9. Save Recommendations
    # --------------------------------------------------

    recommendation_df.to_csv(
        "data/processed/recommendations.csv",
        index=False
    )


    print(
        "\nRecommendations saved successfully!"
    )