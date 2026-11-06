import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


# ============================================================
# CONFIGURATION
# ============================================================

PRODUCTS_PATH = "data/processed/processed_products.csv"
INTERACTIONS_PATH = "data/processed/user_product_interactions.csv"
EMBEDDINGS_PATH = "embeddings/product_embeddings.npy"

OUTPUT_PATH = "models/ltr_dataset.csv"

STRONG_INTERACTION_THRESHOLD = 5

# Hold out this fraction of strong interactions for LTR positives.
# 20% means approximately 20% of strong interactions per user
# are kept unseen during training and used as positive examples.
HOLDOUT_FRACTION = 0.20

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("BUILDING HOLDOUT-AWARE LTR DATASET")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

products = pd.read_csv(PRODUCTS_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

print()
print(f"Products loaded      : {len(products)}")
print(f"Interactions loaded  : {len(interactions)}")
print(f"Embeddings shape     : {embeddings.shape}")


# ============================================================
# VALIDATION
# ============================================================

if len(products) != len(embeddings):
    raise ValueError(
        "Number of products and embeddings do not match."
    )

required_product_columns = [
    "product_name",
    "category",
    "brand",
    "color",
    "material",
    "price"
]

for column in required_product_columns:

    if column not in products.columns:
        raise ValueError(
            f"Missing product column: {column}"
        )


required_interaction_columns = [
    "user_id",
    "product_id",
    "interaction"
]

for column in required_interaction_columns:

    if column not in interactions.columns:
        raise ValueError(
            f"Missing interaction column: {column}"
        )


# ============================================================
# CREATE 1-BASED PRODUCT IDs
# ============================================================

# The interaction dataset uses product IDs corresponding
# to 1-based row positions in the product dataset.

products = products.copy()

products["product_id"] = np.arange(
    1,
    len(products) + 1
)


# ============================================================
# CLEAN PRODUCT TEXT COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "category",
    "brand",
    "color",
    "material"
]

for column in TEXT_COLUMNS:

    products[column] = (
        products[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# STYLE COLUMN
# ============================================================

# Some datasets do not have a separate style column.
# If it does not exist, derive a simple style signal from
# the product description/name.

if "style" not in products.columns:

    products["style"] = ""

else:

    products["style"] = (
        products["style"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# NUMERIC PRICE
# ============================================================

products["price_numeric"] = pd.to_numeric(
    products["price"],
    errors="coerce"
)

median_price = products[
    "price_numeric"
].median()

if pd.isna(median_price):
    median_price = 0.0

products["price_numeric"] = (
    products["price_numeric"]
    .fillna(median_price)
)


# ============================================================
# NORMALIZE INTERACTIONS
# ============================================================

interactions = interactions.copy()

interactions["user_id"] = pd.to_numeric(
    interactions["user_id"],
    errors="coerce"
)

interactions["product_id"] = pd.to_numeric(
    interactions["product_id"],
    errors="coerce"
)

interactions["interaction"] = pd.to_numeric(
    interactions["interaction"],
    errors="coerce"
)

interactions = interactions.dropna(
    subset=[
        "user_id",
        "product_id",
        "interaction"
    ]
)

interactions["user_id"] = (
    interactions["user_id"].astype(int)
)

interactions["product_id"] = (
    interactions["product_id"].astype(int)
)

interactions["interaction"] = (
    interactions["interaction"].astype(float)
)


# ============================================================
# KEEP VALID PRODUCT IDS
# ============================================================

valid_product_ids = set(
    products["product_id"]
)

interactions = interactions[
    interactions["product_id"].isin(
        valid_product_ids
    )
].copy()


# ============================================================
# USER LIST
# ============================================================

user_ids = sorted(
    interactions["user_id"].unique()
)

product_ids = products[
    "product_id"
].tolist()


# ============================================================
# PRODUCT INDEX
# ============================================================

product_to_index = {
    product_id: index
    for index, product_id
    in enumerate(product_ids)
}


# ============================================================
# HOLDOUT SPLIT
# ============================================================

print()
print("=" * 60)
print("CREATING HOLDOUT SPLIT")
print("=" * 60)

rng = np.random.default_rng(
    RANDOM_STATE
)

training_interactions = []
heldout_interactions = []

for user_id in user_ids:

    user_data = interactions[
        interactions["user_id"] == user_id
    ].copy()

    # Strong interactions are our positive-label proxy.
    strong = user_data[
        user_data["interaction"]
        >= STRONG_INTERACTION_THRESHOLD
    ].copy()

    weak = user_data[
        user_data["interaction"]
        < STRONG_INTERACTION_THRESHOLD
    ].copy()

    # --------------------------------------------------------
    # Hold out strong interactions
    # --------------------------------------------------------

    if len(strong) >= 2:

        holdout_count = max(
            1,
            int(
                np.floor(
                    len(strong)
                    * HOLDOUT_FRACTION
                )
            )
        )

        # Never hold out all strong interactions.
        holdout_count = min(
            holdout_count,
            len(strong) - 1
        )

        selected_indices = rng.choice(
            strong.index.to_numpy(),
            size=holdout_count,
            replace=False
        )

        user_heldout = strong.loc[
            selected_indices
        ].copy()

        user_training_strong = strong.drop(
            selected_indices
        ).copy()

    else:

        # If a user has fewer than 2 strong
        # interactions, keep all for training.
        user_heldout = strong.iloc[0:0].copy()

        user_training_strong = strong.copy()

    # --------------------------------------------------------
    # Training interactions
    # --------------------------------------------------------

    user_training = pd.concat(
        [
            weak,
            user_training_strong
        ],
        ignore_index=True
    )

    training_interactions.append(
        user_training
    )

    heldout_interactions.append(
        user_heldout
    )


training_interactions = pd.concat(
    training_interactions,
    ignore_index=True
)

heldout_interactions = pd.concat(
    heldout_interactions,
    ignore_index=True
)


print()
print(
    f"Training interactions : "
    f"{len(training_interactions)}"
)

print(
    f"Held-out interactions : "
    f"{len(heldout_interactions)}"
)

print(
    f"Training strong      : "
    f"{sum(training_interactions['interaction'] >= STRONG_INTERACTION_THRESHOLD)}"
)

print(
    f"Held-out strong      : "
    f"{sum(heldout_interactions['interaction'] >= STRONG_INTERACTION_THRESHOLD)}"
)


# ============================================================
# BUILD TRAINING INTERACTION MATRIX
# ============================================================

print()
print(
    "Building training interaction matrix..."
)

user_to_index = {
    user_id: index
    for index, user_id
    in enumerate(user_ids)
}

rows = []
cols = []
data = []

for _, row in training_interactions.iterrows():

    user_id = int(
        row["user_id"]
    )

    product_id = int(
        row["product_id"]
    )

    interaction = float(
        row["interaction"]
    )

    rows.append(
        user_to_index[user_id]
    )

    cols.append(
        product_to_index[product_id]
    )

    data.append(
        interaction
    )


interaction_matrix = csr_matrix(
    (
        data,
        (rows, cols)
    ),
    shape=(
        len(user_ids),
        len(product_ids)
    )
)


print(
    f"Interaction matrix : "
    f"{interaction_matrix.shape}"
)


# ============================================================
# TRAIN ALS ON TRAINING DATA ONLY
# ============================================================

print()
print("Training ALS on training interactions...")

als_model = AlternatingLeastSquares(
    factors=10,
    iterations=10,
    random_state=RANDOM_STATE
)

als_model.fit(
    interaction_matrix
)

print("ALS training completed.")


# ============================================================
# NORMALIZED CF SCORE
# ============================================================

def get_cf_scores(user_id):

    user_index = user_to_index[
        user_id
    ]

    user_factor = (
        als_model.user_factors[
            user_index
        ]
    )

    scores = np.dot(
        als_model.item_factors,
        user_factor
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    min_score = scores.min()
    max_score = scores.max()

    if max_score > min_score:

        scores = (
            scores - min_score
        ) / (
            max_score - min_score
        )

    else:

        scores = np.zeros_like(
            scores
        )

    return scores


# ============================================================
# CONTENT SIMILARITY
# ============================================================

def get_content_scores(
    user_id
):

    user_training = training_interactions[
        training_interactions["user_id"]
        == user_id
    ]

    strong_training = user_training[
        user_training["interaction"]
        >= STRONG_INTERACTION_THRESHOLD
    ]

    strong_product_ids = (
        strong_training[
            "product_id"
        ]
        .astype(int)
        .tolist()
    )

    strong_product_ids = [
        product_id
        for product_id
        in strong_product_ids
        if product_id
        in product_to_index
    ]

    scores = np.zeros(
        len(products),
        dtype=float
    )

    if not strong_product_ids:
        return scores

    strong_indices = [
        product_to_index[
            product_id
        ]
        for product_id
        in strong_product_ids
    ]

    strong_embeddings = (
        embeddings[
            strong_indices
        ]
    )

    # Normalize strong embeddings.
    strong_norms = np.linalg.norm(
        strong_embeddings,
        axis=1,
        keepdims=True
    )

    strong_norms[
        strong_norms == 0
    ] = 1

    normalized_strong = (
        strong_embeddings
        / strong_norms
    )

    # Normalize all product embeddings.
    product_norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    product_norms[
        product_norms == 0
    ] = 1

    normalized_products = (
        embeddings
        / product_norms
    )

    similarities = np.dot(
        normalized_products,
        normalized_strong.T
    )

    scores = similarities.max(
        axis=1
    )

    return scores


# ============================================================
# USER PREFERENCE PROFILE
# ============================================================

def get_user_preferences(
    user_id
):

    user_training = training_interactions[
        training_interactions["user_id"]
        == user_id
    ]

    strong_training = user_training[
        user_training["interaction"]
        >= STRONG_INTERACTION_THRESHOLD
    ]

    strong_product_ids = (
        strong_training[
            "product_id"
        ]
        .astype(int)
        .tolist()
    )

    strong_product_ids = [
        product_id
        for product_id
        in strong_product_ids
        if product_id
        in product_to_index
    ]

    if not strong_product_ids:

        return {
            "categories": set(),
            "brands": set(),
            "materials": set(),
            "styles": set(),
            "avg_price": float(
                products[
                    "price_numeric"
                ].mean()
            )
        }

    preferred_products = products[
        products["product_id"].isin(
            strong_product_ids
        )
    ]

    categories = set(
        preferred_products[
            "category"
        ]
        .dropna()
        .astype(str)
    )

    brands = set(
        preferred_products[
            "brand"
        ]
        .dropna()
        .astype(str)
    )

    materials = set(
        preferred_products[
            "material"
        ]
        .dropna()
        .astype(str)
    )

    styles = set(
        preferred_products[
            "style"
        ]
        .dropna()
        .astype(str)
    )

    avg_price = float(
        preferred_products[
            "price_numeric"
        ].mean()
    )

    return {
        "categories": categories,
        "brands": brands,
        "materials": materials,
        "styles": styles,
        "avg_price": avg_price
    }


# ============================================================
# CREATE LTR DATASET
# ============================================================

print()
print("=" * 60)
print("CREATING HOLDOUT-AWARE LTR FEATURES")
print("=" * 60)

all_rows = []


for user_id in user_ids:

    print(
        f"Creating features for user "
        f"{user_id} ..."
    )

    # --------------------------------------------------------
    # User training interactions
    # --------------------------------------------------------

    user_training = training_interactions[
        training_interactions["user_id"]
        == user_id
    ]

    training_product_ids = set(
        user_training[
            "product_id"
        ].astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # Held-out positives
    # --------------------------------------------------------

    user_heldout = heldout_interactions[
        heldout_interactions["user_id"]
        == user_id
    ]

    heldout_positive_ids = set(
        user_heldout[
            "product_id"
        ].astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # LTR candidates are products that were NOT used
    # as training interactions.
    #
    # This means the model learns to rank unseen products.
    # --------------------------------------------------------

    candidate_products = products[
        ~products["product_id"].isin(
            training_product_ids
        )
    ].copy()

    if candidate_products.empty:
        continue

    # --------------------------------------------------------
    # User preferences from training data
    # --------------------------------------------------------

    preferences = get_user_preferences(
        user_id
    )

    # --------------------------------------------------------
    # CF scores
    # --------------------------------------------------------

    cf_scores = get_cf_scores(
        user_id
    )

    # --------------------------------------------------------
    # Content scores
    # --------------------------------------------------------

    content_scores = get_content_scores(
        user_id
    )

    # --------------------------------------------------------
    # Create feature rows
    # --------------------------------------------------------

    for _, product in candidate_products.iterrows():

        product_id = int(
            product["product_id"]
        )

        product_index = product_to_index[
            product_id
        ]

        # ----------------------------------------------------
        # CF score
        # ----------------------------------------------------

        cf_score = float(
            cf_scores[
                product_index
            ]
        )

        # ----------------------------------------------------
        # Content similarity
        # ----------------------------------------------------

        content_similarity = float(
            content_scores[
                product_index
            ]
        )

        # ----------------------------------------------------
        # Price delta
        # ----------------------------------------------------

        average_price = (
            preferences["avg_price"]
        )

        product_price = float(
            product["price_numeric"]
        )

        if average_price > 0:

            price_delta = abs(
                product_price
                - average_price
            ) / average_price

        else:

            price_delta = 0.0

        # ----------------------------------------------------
        # Category match
        # ----------------------------------------------------

        category_match = int(
            product["category"]
            in preferences["categories"]
        )

        # ----------------------------------------------------
        # Brand match
        # ----------------------------------------------------

        brand_match = int(
            product["brand"]
            in preferences["brands"]
        )

        # ----------------------------------------------------
        # Material match
        # ----------------------------------------------------

        material_match = int(
            product["material"]
            in preferences["materials"]
        )

        # ----------------------------------------------------
        # Style match
        # ----------------------------------------------------

        style_match = int(
            product["style"]
            in preferences["styles"]
        )

        # ----------------------------------------------------
        # LABEL
        #
        # Held-out strong interaction = positive.
        # Other unseen candidates = negative.
        # ----------------------------------------------------

        label = int(
            product_id
            in heldout_positive_ids
        )

        all_rows.append({
            "user_id": user_id,
            "product_id": product_id,

            "cf_score": cf_score,
            "content_similarity":
                content_similarity,
            "price_delta":
                price_delta,

            "category_match":
                category_match,
            "brand_match":
                brand_match,
            "material_match":
                material_match,
            "style_match":
                style_match,

            "label": label
        })


# ============================================================
# DATAFRAME
# ============================================================

ltr_df = pd.DataFrame(
    all_rows
)


# ============================================================
# VALIDATION
# ============================================================

if ltr_df.empty:

    raise ValueError(
        "LTR dataset is empty."
    )


FEATURE_COLUMNS = [
    "cf_score",
    "content_similarity",
    "price_delta",
    "category_match",
    "brand_match",
    "material_match",
    "style_match"
]


REQUIRED_COLUMNS = [
    "user_id",
    "product_id"
] + FEATURE_COLUMNS + [
    "label"
]


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in ltr_df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: "
        f"{missing_columns}"
    )


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

for column in FEATURE_COLUMNS:

    ltr_df[column] = pd.to_numeric(
        ltr_df[column],
        errors="coerce"
    )

    ltr_df[column] = (
        ltr_df[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


ltr_df["label"] = (
    ltr_df["label"]
    .astype(int)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

ltr_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

positive_count = int(
    ltr_df["label"].sum()
)

negative_count = int(
    len(ltr_df)
    - positive_count
)

print()
print("=" * 60)
print("HOLDOUT-AWARE LTR DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print()
print(
    f"Rows                : "
    f"{len(ltr_df)}"
)

print(
    f"Columns             : "
    f"{len(ltr_df.columns)}"
)

print(
    f"Positive labels     : "
    f"{positive_count}"
)

print(
    f"Negative labels     : "
    f"{negative_count}"
)

print(
    f"Users               : "
    f"{ltr_df['user_id'].nunique()}"
)

print(
    f"Products            : "
    f"{ltr_df['product_id'].nunique()}"
)

print()
print("Feature columns:")

for feature in FEATURE_COLUMNS:

    print(
        f" - {feature}"
    )

print()
print(
    "Label definition:"
)

print(
    " - 1 = held-out strong interaction"
)

print(
    " - 0 = unseen candidate without "
    "held-out strong interaction"
)

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)