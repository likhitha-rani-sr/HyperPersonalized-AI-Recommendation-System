import os
import time
import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from src.query_processor import process_query
except ImportError:
    from query_processor import process_query


def _safe_print(*values):
    print(
        *(
            str(value).encode("ascii", "backslashreplace").decode("ascii")
            for value in values
        )
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PRODUCT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "processed_products.csv"
)

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "product_embeddings.npy"
)


# ============================================================
# LOAD PRODUCTS
# ============================================================

print("=" * 60)
print("CONVERSATIONAL RAG PRODUCT RETRIEVER")
print("=" * 60)

products_df = pd.read_csv(
    PRODUCT_FILE
)

print(
    f"Products loaded : {len(products_df)}"
)

print(
    f"Columns         : {list(products_df.columns)}"
)

products_df = products_df.reset_index(
    drop=True
)

products_df["product_id"] = np.arange(
    1,
    len(products_df) + 1
)

products_df["price"] = pd.to_numeric(
    products_df["price"].replace({"Unknown": pd.NA, "unknown": pd.NA, "": pd.NA}),
    errors="coerce"
)
products_df = products_df.dropna(subset=["price"]).copy()
products_df = products_df[products_df["price"] > 0].copy()


# ============================================================
# CLEAN TEXT COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "product_name",
    "category",
    "brand",
    "color",
    "material",
    "description"
]

for column in TEXT_COLUMNS:

    if column in products_df.columns:

        products_df[column] = (
            products_df[column]
            .fillna("")
            .astype(str)
        )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print()
print("Loading pre-computed product embeddings...")

if not os.path.exists(
    EMBEDDING_FILE
):

    raise FileNotFoundError(
        f"Embedding file not found: "
        f"{EMBEDDING_FILE}"
    )

product_embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"Embedding shape : "
    f"{product_embeddings.shape}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded."
)


# ============================================================
# CONVERSATION STATE
# ============================================================

conversation_history = []

current_preferences = {
    "category": None,
    "color": None,
    "material": None,
    "style": None,
    "brand": None,
    "excluded_brands": [],
    "min_price": None,
    "max_price": None,
    "top_k": 5
}


# ============================================================
# GREETING
# ============================================================

def is_greeting(query):

    text = str(
        query
    ).lower().strip()

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "good night"
    }

    return text in greetings


# ============================================================
# NORMALIZE QUERY
# ============================================================

def normalize_query_wording(query):

    query = str(
        query
    ).lower()

    replacements = {

        r"\bwooden\b":
            "wood",
        r"\bwodden\b":
            "wood",
        r"\bsofas\b":
            "sofa",
        r"\bchairs\b":
            "chair",
        r"\btables\b":
            "table",
        r"\bbeds\b":
            "bed",
        r"\bwardrobes\b":
            "wardrobe",
        r"\bwardrope\b":
            "wardrobe",
        r"\bwardroop\b":
            "wardrobe",
        r"\bwaardroop\b":
            "wardrobe",
        r"\bwardroom\b":
            "wardrobe",
        r"\bwardroob\b":
            "wardrobe",
        r"\bmetel\b":
            "metal",
        r"\bmetalic\b":
            "metal",
        r"\blamps\b":
            "lamp",
        r"\bdesks\b":
            "desk",
        r"\bcabinets\b":
            "cabinet",
        r"\bbookshelves\b":
            "bookshelf"
    }

    for pattern, replacement in replacements.items():

        query = re.sub(
            pattern,
            replacement,
            query
        )

    return query


# ============================================================
# RESET PREFERENCES
# ============================================================

def clear_preferences():

    global current_preferences

    current_preferences = {
        "category": None,
        "color": None,
        "material": None,
        "style": None,
        "brand": None,
        "excluded_brands": [],
        "min_price": None,
        "max_price": None,
        "top_k": 5
    }


# ============================================================
# DETECT NEW PRODUCT REQUEST
# ============================================================

def contains_product_category(query):

    query = normalize_query_wording(
        query
    )

    categories = [
        "sofa",
        "chair",
        "table",
        "bed",
        "wardrobe",
        "lamp",
        "desk",
        "bookshelf",
        "cabinet"
    ]

    for category in categories:

        if re.search(
            r"\b"
            + re.escape(category)
            + r"\b",
            query
        ):

            return True

    return False


# ============================================================
# UPDATE PREFERENCES
# ============================================================

def update_preferences(
    query_info,
    new_product_request=False
):

    global current_preferences

    # --------------------------------------------------------
    # If the user starts a NEW product category,
    # clear unrelated old preferences.
    # --------------------------------------------------------

    if new_product_request:

        old_category = current_preferences.get(
            "category"
        )

        new_category = query_info.get(
            "category"
        )

        # A different product category means
        # start a fresh recommendation request.
        if (
            new_category
            and old_category
            and new_category != old_category
        ):

            old_top_k = current_preferences.get(
                "top_k",
                5
            )

            current_preferences = {
                "category": None,
                "color": None,
                "material": None,
                "style": None,
                "brand": None,
                "excluded_brands": [],
                "min_price": None,
                "max_price": None,
                "top_k": old_top_k
            }

    # --------------------------------------------------------
    # Store values explicitly mentioned by the user
    # --------------------------------------------------------

    excluded_brands = query_info.get(
        "excluded_brands",
        []
    )

    if excluded_brands:
        current_preferences["excluded_brands"] = list(
            dict.fromkeys(
                current_preferences.get("excluded_brands", [])
                + excluded_brands
            )
        )

        if current_preferences.get("brand") in excluded_brands:
            current_preferences["brand"] = None

    elif query_info.get("brand") is not None:
        current_preferences["excluded_brands"] = []

    for key in [
        "category",
        "color",
        "material",
        "style",
        "brand",
        "min_price",
        "max_price",
        "top_k"
    ]:

        value = query_info.get(
            key
        )

        if value is not None:

            current_preferences[key] = value


def is_category_only_retry(query_info, query):

    category = query_info.get("category")
    if not category or category != current_preferences.get("category"):
        return False

    if str(query).strip().casefold() not in {
        category,
        f"{category}s",
    }:
        return False

    if any(
        query_info.get(key)
        for key in (
            "color",
            "material",
            "style",
            "brand",
            "excluded_brands",
            "min_price",
            "max_price",
        )
    ):
        return False

    if not conversation_history:
        return False

    previous_answer = str(
        conversation_history[-1].get("assistant", "")
    ).casefold()
    return "no exact matches were found" in previous_answer


def is_new_general_furniture_request(query_info, query):

    normalized_query = normalize_query_wording(query)
    return (
        query_info.get("category") is None
        and re.search(r"\bfurniture\b", normalized_query)
        and re.search(
            r"\b(?:recommend|find|show|suggest|looking for|i want|i need)\b",
            normalized_query,
        )
    )


# ============================================================
# BUILD COMBINED QUERY
# ============================================================

def build_combined_query(
    current_query
):

    query = normalize_query_wording(
        current_query
    ).strip()

    parts = [
        query
    ]

    for key in [
        "category",
        "color",
        "material",
        "style",
        "brand"
    ]:

        value = current_preferences.get(
            key
        )

        if value:

            value_text = str(
                value
            ).lower()

            if value_text not in query:

                parts.append(
                    value_text
                )

    return " ".join(
        parts
    )


# ============================================================
# TEXT MATCH HELPER
# ============================================================

def text_contains(
    series,
    value
):

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            str(value).lower(),
            na=False,
            regex=False
        )
    )


# ============================================================
# STRUCTURED FILTERING
# ============================================================

def filter_products(
    query_info
):

    df = products_df.copy()

    category = query_info.get(
        "category"
    )

    color = query_info.get(
        "color"
    )

    material = query_info.get(
        "material"
    )

    brand = query_info.get(
        "brand"
    )

    min_price = query_info.get(
        "min_price"
    )

    max_price = query_info.get(
        "max_price"
    )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if category:

        category_value = str(
            category
        ).lower()

        category_aliases = {
            "sofa": [
                "sofa"
            ],
            "chair": [
                "chair"
            ],
            "table": [
                "table"
            ],
            "bed": [
                "bed"
            ],
            "wardrobe": [
                "wardrobe"
            ],
            "lamp": [
                "lamp"
            ],
            "desk": [
                "desk"
            ],
            "bookshelf": [
                "bookshelf"
            ],
            "cabinet": [
                "cabinet"
            ]
        }

        possible_categories = (
            category_aliases.get(
                category_value,
                [category_value]
            )
        )

        category_mask = pd.Series(
            False,
            index=df.index
        )

        for value in possible_categories:

            category_mask = (
                category_mask
                | text_contains(
                    df["category"],
                    value
                )
            )

        df = df[
            category_mask
        ]

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    if color:

        color_value = str(
            color
        ).lower()

        # gray / grey treated as same
        if color_value in {
            "gray",
            "grey"
        }:

            color_mask = (
                text_contains(
                    df["color"],
                    "grey"
                )
                |
                text_contains(
                    df["color"],
                    "gray"
                )
            )

            df = df[
                color_mask
            ]

        else:

            df = df[
                text_contains(
                    df["color"],
                    color_value
                )
            ]

    # --------------------------------------------------------
    # MATERIAL
    # --------------------------------------------------------

    if material:

        material_value = str(
            material
        ).lower()

        if material_value in {
            "wood",
            "wooden"
        }:

            # Match both Wood and Engineered Wood
            df = df[
                text_contains(
                    df["material"],
                    "wood"
                )
            ]

        else:

            df = df[
                text_contains(
                    df["material"],
                    material_value
                )
            ]

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    if brand:

        brand_value = str(
            brand
        ).lower()

        # Normalize Home Centre
        if brand_value == "home centre":
            brand_value = "homecentre"

        df = df[
            text_contains(
                df["brand"],
                brand_value
            )
        ]

    # Excluded brands apply to the current query and remain in chat memory.
    excluded_brands = query_info.get(
        "excluded_brands",
        []
    )

    if excluded_brands:

        normalized_brands = (
            df["brand"].fillna("").astype(str)
            + " "
            + df["product_name"].fillna("").astype(str)
        ).str.lower().str.replace(
            r"\s+",
            "",
            regex=True
        )

        for excluded_brand in excluded_brands:

            excluded_value = re.sub(
                r"\s+",
                "",
                str(excluded_brand).lower()
            )

            df = df[
                ~normalized_brands.loc[df.index].str.contains(
                    excluded_value,
                    regex=False,
                    na=False
                )
            ]

    # --------------------------------------------------------
    # MIN PRICE
    # --------------------------------------------------------

    if min_price is not None:

        df = df[
            df["price"]
            >= float(min_price)
        ]

    # --------------------------------------------------------
    # MAX PRICE
    # --------------------------------------------------------

    if max_price is not None:

        df = df[
            df["price"]
            <= float(max_price)
        ]

    return df


def _matches_requested_value(
    product,
    constraint,
    requested_value,
):

    column = {
        "category": "category",
        "color": "color",
        "material": "material",
        "brand": "brand",
    }[constraint]

    actual_value = str(
        product.get(column, "")
    ).strip().casefold()
    requested_value = str(
        requested_value
    ).strip().casefold()

    if constraint == "brand":
        actual_value = re.sub(r"\s+", "", actual_value)
        requested_value = re.sub(r"\s+", "", requested_value)

    return requested_value in actual_value


def find_alternative_products(
    query_info,
    query,
    top_k,
):

    alternatives_by_index = {}

    for constraint in ("material", "category", "color", "brand"):
        requested_value = query_info.get(constraint)
        if not requested_value:
            continue

        relaxed_query_info = query_info.copy()
        relaxed_query_info[constraint] = None
        candidates = filter_products(relaxed_query_info)

        for index, product in candidates.iterrows():
            if _matches_requested_value(
                product,
                constraint,
                requested_value,
            ):
                continue

            if index in alternatives_by_index:
                continue

            alternative = product.copy()
            actual_value = str(
                product.get(
                    {
                        "category": "category",
                        "color": "color",
                        "material": "material",
                        "brand": "brand",
                    }[constraint],
                    "",
                )
            ).strip()
            catalog_value = actual_value or "not specified"
            alternative["alternative_reason"] = (
                f"Does not satisfy the requested {constraint} "
                f"'{requested_value}'; the catalog lists "
                f"'{catalog_value}'."
            )
            alternatives_by_index[index] = alternative

    if not alternatives_by_index:
        return products_df.iloc[0:0].copy()

    alternatives = pd.DataFrame.from_dict(
        alternatives_by_index,
        orient="index",
    )
    return rank_products(
        alternatives,
        query,
    ).head(
        int(top_k)
    ).copy()


# ============================================================
# SEMANTIC RANKING
# ============================================================

def rank_products(
    df,
    query
):

    if df.empty:

        return df

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )

    indices = df.index.to_numpy()

    candidate_embeddings = (
        product_embeddings[
            indices
        ]
    )

    similarities = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )[0]

    result = df.copy()

    result[
        "similarity"
    ] = similarities

    result = result.sort_values(
        "similarity",
        ascending=False
    )

    return result


def diversify_products(
    ranked_products,
    top_k,
    max_per_brand=2
):

    if (
        ranked_products.empty
        or "brand" not in ranked_products.columns
    ):

        return ranked_products.head(
            top_k
        ).copy()

    selected_indices = []
    deferred_indices = []
    brand_counts = {}

    for index, product in ranked_products.iterrows():

        brand = product.get(
            "brand"
        )

        brand_key = (
            str(brand).strip().casefold()
            if pd.notna(brand) and str(brand).strip()
            else "unknown"
        )

        if brand_counts.get(
            brand_key,
            0
        ) < max_per_brand:

            selected_indices.append(
                index
            )
            brand_counts[brand_key] = (
                brand_counts.get(
                    brand_key,
                    0
                )
                + 1
            )

        else:

            deferred_indices.append(
                index
            )

        if len(selected_indices) >= top_k:
            break

    if len(selected_indices) < top_k:

        selected_indices.extend(
            deferred_indices[
                :top_k - len(selected_indices)
            ]
        )

    return ranked_products.loc[
        selected_indices
    ].copy()


# ============================================================
# FAST GROUNDED RESPONSE
# ============================================================

def generate_fast_response(
    products,
    query_info
):

    if products.empty:
        constraints = []
        category = query_info.get("category")
        color = query_info.get("color")
        material = query_info.get("material")
        brand = query_info.get("brand")
        excluded_brands = query_info.get("excluded_brands") or []
        min_price = query_info.get("min_price")
        max_price = query_info.get("max_price")

        if category:
            constraints.append(str(category))

        if color:
            constraints.append(f"{color} color")

        if material:
            constraints.append(f"{material} material")

        if brand:
            constraints.append(f"from {brand}")

        if excluded_brands:
            brands = ", ".join(
                str(brand)
                for brand in excluded_brands
            )
            constraints.append(f"excluding {brands}")

        if min_price is not None:
            constraints.append(
                f"priced at or above ₹{float(min_price):,.0f}"
            )

        if max_price is not None:
            constraints.append(
                f"priced at or below ₹{float(max_price):,.0f}"
            )

        if constraints:
            requirements = ", ".join(constraints)
            return (
                "I couldn't find any products matching all your "
                f"requirements ({requirements}). Try relaxing one "
                "of these filters."
            )

        return (
            "I couldn't find furniture matching "
            "your current requirements. Try changing "
            "the budget, brand, color, material, "
            "or category."
        )

    first = products.iloc[0]

    name = first[
        "product_name"
    ]

    brand = first[
        "brand"
    ]

    price = first[
        "price"
    ]

    category = query_info.get(
        "category"
    )

    color = query_info.get(
        "color"
    )

    max_price = query_info.get(
        "max_price"
    )

    if category and color:

        response = (
            f"The {name} from {brand} "
            f"matches your {color} "
            f"{category} request and is "
            f"priced at ₹{price:.0f}."
        )

    elif category:

        response = (
            f"The {name} from {brand} "
            f"matches your {category} "
            f"request and is priced at "
            f"₹{price:.0f}."
        )

    else:

        response = (
            f"The {name} from {brand} "
            f"is priced at ₹{price:.0f} "
            f"and matches your requirements."
        )

    if max_price is not None:

        if float(price) <= float(
            max_price
        ):

            response = (
                response.rstrip(".")
                + f", within your "
                f"₹{float(max_price):.0f} budget."
            )

    return response


# ============================================================
# DISPLAY PRODUCTS
# ============================================================

def display_products(
    products
):

    _safe_print(
        f"Products: {len(products)}"
    )

    for _, row in products.iterrows():

        _safe_print(
            f"{row['product_name']} | "
            f"{row['brand']} | "
            f"{row['category']} | "
            f"{row['color']} | "
            f"{row['material']} | "
            f"₹ {row['price']}"
        )


# ============================================================
# RECOMMEND
# ============================================================

def recommend(
    query
):

    start_time = time.time()

    _safe_print()
    _safe_print(
        f"Current query: {query}"
    )

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if is_greeting(
        query
    ):

        answer = (
            "Hello! 👋 I can help you find "
            "furniture based on category, "
            "color, brand, material, style, "
            "and budget."
        )

        _safe_print(answer)

        return {
            "answer": answer,
            "products": [],
            "preferences":
                current_preferences.copy(),
            "response_time":
                round(
                    time.time()
                    - start_time,
                    2
                )
        }

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    normalized_query = (
        normalize_query_wording(
            query
        )
    )

    # --------------------------------------------------------
    # PROCESS CURRENT QUERY
    # --------------------------------------------------------

    query_info = process_query(
        normalized_query
    )

    # Wooden → wood
    if (
        "wood" in normalized_query
        and query_info.get(
            "material"
        ) is None
    ):

        query_info[
            "material"
        ] = "wood"

    if is_category_only_retry(query_info, query):
        clear_preferences()

    if is_new_general_furniture_request(query_info, query):
        remembered_budget = current_preferences.get("max_price")
        references_budget = re.search(
            r"\b(?:my|the)\s+budget\b",
            normalized_query,
        )
        clear_preferences()
        if references_budget and remembered_budget is not None:
            current_preferences["max_price"] = remembered_budget

    # --------------------------------------------------------
    # DETERMINE IF THIS IS A NEW PRODUCT REQUEST
    # --------------------------------------------------------

    new_product_request = (
        query_info.get(
            "category"
        ) is not None
    )

    # --------------------------------------------------------
    # UPDATE MEMORY
    # --------------------------------------------------------

    update_preferences(
        query_info,
        new_product_request
    )

    # --------------------------------------------------------
    # BUILD FINAL QUERY
    # --------------------------------------------------------

    combined_query = (
        build_combined_query(
            query
        )
    )

    _safe_print(
        f"Combined RAG query: "
        f"{combined_query}"
    )

    _safe_print(
        "Preferences:"
    )

    for key, value in (
        current_preferences.items()
    ):

        if value is not None:

            _safe_print(
                f"  {key}: {value}"
            )

    # --------------------------------------------------------
    # FINAL QUERY INFORMATION
    # --------------------------------------------------------

    final_query_info = (
        process_query(
            combined_query
        )
    )

    # Wooden → wood
    if (
        current_preferences.get(
            "material"
        ) == "wood"
        and final_query_info.get(
            "material"
        ) is None
    ):

        final_query_info[
            "material"
        ] = "wood"

    # --------------------------------------------------------
    # RESTORE MEMORY VALUES
    # --------------------------------------------------------

    for key in [
        "category",
        "color",
        "material",
        "style",
        "brand",
        "excluded_brands",
        "min_price",
        "max_price",
        "top_k"
    ]:

        if (
            final_query_info.get(key) is None
            or (
                key == "excluded_brands"
                and not final_query_info.get(key)
            )
        ):

            final_query_info[
                key
            ] = current_preferences.get(
                key
            )

    # --------------------------------------------------------
    # STRUCTURED FILTER
    # --------------------------------------------------------

    filtered_products = (
        filter_products(
            final_query_info
        )
    )

    _safe_print()
    _safe_print(
        "Products after structured "
        "filtering:",
        len(filtered_products)
    )

    # --------------------------------------------------------
    # SEMANTIC RANKING
    # --------------------------------------------------------

    if not filtered_products.empty:

        ranked_products = (
            rank_products(
                filtered_products,
                combined_query
            )
        )

        top_k = (
            final_query_info.get(
                "top_k",
                5
            )
        )

        if top_k is None:
            top_k = 5

        if final_query_info.get(
            "brand"
        ):

            products = (
                ranked_products
                .head(
                    int(top_k)
                )
                .copy()
            )

        else:

            products = diversify_products(
                ranked_products,
                int(top_k)
            )

    else:

        products = (
            filtered_products.copy()
        )

    alternative_products = products_df.iloc[0:0].copy()
    if products.empty:
        top_k = final_query_info.get("top_k") or 5
        alternative_products = find_alternative_products(
            final_query_info,
            combined_query,
            top_k,
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    answer = generate_fast_response(
        products,
        final_query_info
    )
    if not alternative_products.empty:
        requested_constraints = [
            f"{key} '{final_query_info[key]}'"
            for key in ("category", "color", "material", "brand")
            if final_query_info.get(key)
        ]
        if final_query_info.get("min_price") is not None:
            requested_constraints.append(
                "priced at or above "
                f"₹{float(final_query_info['min_price']):,.0f}"
            )
        if final_query_info.get("max_price") is not None:
            requested_constraints.append(
                "priced at or below "
                f"₹{float(final_query_info['max_price']):,.0f}"
            )
        requested_text = ", ".join(requested_constraints) or "requirements"
        answer = (
            f"No exact matches were found for your requested "
            f"{requested_text}.\n\n### Alternative suggestions"
        )
    elif products.empty:
        requested_constraints = [
            f"{key} '{final_query_info[key]}'"
            for key in ("category", "color", "material", "brand")
            if final_query_info.get(key)
        ]
        if final_query_info.get("min_price") is not None:
            requested_constraints.append(
                "priced at or above "
                f"₹{float(final_query_info['min_price']):,.0f}"
            )
        if final_query_info.get("max_price") is not None:
            requested_constraints.append(
                "priced at or below "
                f"₹{float(final_query_info['max_price']):,.0f}"
            )
        requested_text = ", ".join(requested_constraints) or "requirements"
        answer = (
            f"No exact matches were found for your requested "
            f"{requested_text}. No suitable alternatives remain "
            f"that satisfy your other active constraints. "
            f"{answer}"
        )

    response_time = round(
        time.time()
        - start_time,
        2
    )

    _safe_print(
        f"Response time: "
        f"{response_time} seconds"
    )

    _safe_print(
        "AI:",
        answer
    )

    display_products(
        products
    )
    if not alternative_products.empty:
        _safe_print("Alternative suggestions:")
        display_products(alternative_products)

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    conversation_history.append(
        {
            "user": query,
            "assistant": answer
        }
    )

    return {
        "answer": answer,
        "products":
            products.to_dict(
                orient="records"
            ),
        "alternative_products":
            alternative_products.to_dict(
                orient="records"
            ),
        "preferences":
            current_preferences.copy(),
        "response_time":
            response_time
    }


# ============================================================
# STREAMLIT UI
# ============================================================

def recommend_for_ui(
    query
):

    return recommend(
        query
    )


# ============================================================
# RESTORE CONVERSATION
# ============================================================

def restore_conversation(
    messages,
    preferences=None
):

    global conversation_history
    global current_preferences

    conversation_history = []

    for message in messages:
        role = message.get("role")

        if role == "user":
            conversation_history.append(
                {
                    "user": message.get("content", ""),
                    "assistant": "",
                }
            )
        elif role == "assistant" and conversation_history:
            conversation_history[-1]["assistant"] = message.get(
                "content",
                "",
            )

    if preferences:

        category = preferences.get("category")
        if isinstance(category, str) and category.casefold() in {
            "dining table",
            "dining tables",
        }:
            category = "table"

        current_preferences = {
            "category":
                category,

            "color":
                preferences.get(
                    "color"
                ),

            "material":
                preferences.get(
                    "material"
                ),

            "style":
                preferences.get(
                    "style"
                ),

            "brand":
                preferences.get(
                    "brand"
                ),

            "excluded_brands":
                preferences.get(
                    "excluded_brands",
                    []
                ),

            "min_price":
                preferences.get(
                    "min_price"
                ),

            "max_price":
                preferences.get(
                    "max_price"
                ),

            "top_k":
                preferences.get(
                    "top_k",
                    5
                )
        }
    else:
        clear_preferences()


# ============================================================
# RESET CONVERSATION
# ============================================================

def reset_conversation():

    global conversation_history
    global current_preferences

    conversation_history = []

    current_preferences = {
        "category": None,
        "color": None,
        "material": None,
        "style": None,
        "brand": None,
        "excluded_brands": [],
        "min_price": None,
        "max_price": None,
        "top_k": 5
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "CONVERSATIONAL RAG FURNITURE RECOMMENDER"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    print()
    print(
        "TEST 1: Greeting"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "hi"
    )

    # --------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------

    print()
    print(
        "TEST 2: Modern Grey Chair"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "I want a modern grey chair under 15000"
    )

    # --------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------

    print()
    print(
        "TEST 3: Wooden Table"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "I want a wooden table under 10000"
    )

    # --------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------

    print()
    print(
        "TEST 4: Black Sofa"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "I want black sofas under 30000"
    )

    # --------------------------------------------------------
    # TEST 5
    # --------------------------------------------------------

    print()
    print(
        "TEST 5: Luxury Bed"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "I want a luxury bed above 20000"
    )

    # --------------------------------------------------------
    # TEST 6
    # --------------------------------------------------------

    print()
    print(
        "TEST 6: IKEA Table"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "I want an IKEA table under 20000"
    )

    # --------------------------------------------------------
    # TEST 7
    # --------------------------------------------------------

    print()
    print(
        "TEST 7: Multi-Turn Conversation"
    )

    print("-" * 60)

    reset_conversation()

    recommend(
        "I want black sofas under 30000"
    )

    recommend(
        "Only IKEA"
    )

    recommend(
        "Make it under 20000"
    )

    recommend(
        "Show only 3"
    )

    print()
    print("=" * 60)
    print(
        "TESTING COMPLETED"
    )
    print("=" * 60)