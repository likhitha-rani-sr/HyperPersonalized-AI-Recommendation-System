import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


# ============================================================
# LOAD INTERACTION DATA
# ============================================================

df = pd.read_csv(
    "data/processed/user_product_interactions.csv"
)


# ============================================================
# CREATE USER-PRODUCT MATRIX
# ============================================================

interaction_matrix = df.pivot_table(
    index="user_id",
    columns="product_id",
    values="interaction",
    fill_value=0
)


# ============================================================
# CONVERT TO SPARSE MATRIX
# ============================================================

user_item_matrix = csr_matrix(
    interaction_matrix.values
)


print("Interaction Matrix Shape:")
print(user_item_matrix.shape)


# ============================================================
# CREATE ALS MODEL
# ============================================================

model = AlternatingLeastSquares(
    factors=10,
    iterations=10,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    user_item_matrix
)


print("\nALS Model Training Done!")