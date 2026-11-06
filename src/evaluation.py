# RECOMMENDATION SYSTEM EVALUATION
# Evaluate ALS for all users

import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


# 1. Load interaction data
df = pd.read_csv(
    "data/processed/user_product_interactions.csv"
)

print("Total interactions:", len(df))


# 2. Split data into training and testing
train_data = df.sample(
    frac=0.8,
    random_state=42
)

test_data = df.drop(
    train_data.index
)

print("Training interactions:", len(train_data))
print("Testing interactions:", len(test_data))


# 3. Create training interaction matrix
train_matrix = train_data.pivot_table(
    index="user_id",
    columns="product_id",
    values="interaction",
    fill_value=0
)

train_user_item_matrix = csr_matrix(
    train_matrix.values
)


# 4. Create and train ALS model
model = AlternatingLeastSquares(
    factors=10,
    iterations=10,
    random_state=42
)

model.fit(train_user_item_matrix)

print("\nALS Model trained!")


# 5. Evaluate all users
results = []

for user_id in train_matrix.index:

    # Get test products for this user
    actual_test_products = set(
        test_data[
            test_data["user_id"] == user_id
        ]["product_id"]
    )

    # Skip users with no test data
    if len(actual_test_products) == 0:
        continue

    # Get user index
    user_index = list(
        train_matrix.index
    ).index(user_id)

    # Generate Top 5 recommendations
    recommendation_ids, recommendation_scores = model.recommend(
        user_index,
        train_user_item_matrix[user_index],
        N=5
    )

    # Convert indexes to product IDs
    recommended_products = []

    for item_index in recommendation_ids:

        product_id = train_matrix.columns[item_index]

        recommended_products.append(
            product_id
        )

    # Find matching products
    common_products = set(
        recommended_products
    ).intersection(
        actual_test_products
    )

    # Precision@5
    precision_at_5 = (
        len(common_products)
        / len(recommended_products)
    )

    # Recall@5
    recall_at_5 = (
        len(common_products)
        / len(actual_test_products)
    )

    results.append({
        "user_id": user_id,
        "precision_at_5": precision_at_5,
        "recall_at_5": recall_at_5
    })


# 6. Convert results to DataFrame
results_df = pd.DataFrame(results)


# 7. Calculate overall metrics
overall_precision = results_df[
    "precision_at_5"
].mean()

overall_recall = results_df[
    "recall_at_5"
].mean()


# 8. Display results
print("\nEvaluation Results:")
print(results_df)

print("\nOverall Precision@5:",
      overall_precision)

print("Overall Recall@5:",
      overall_recall)