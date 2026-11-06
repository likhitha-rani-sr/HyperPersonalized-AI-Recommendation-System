import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from query_processor import process_query


# ============================================================
# CONFIGURATION
# ============================================================

PRODUCTS_PATH = "data/processed/processed_products.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5

SIMILARITY_THRESHOLD = 0.10


# ============================================================
# LOAD PRODUCTS
# ============================================================

products = pd.read_csv(
    PRODUCTS_PATH
)

print("=" * 60)
print("RAG PRODUCT RETRIEVER")
print("=" * 60)

print(
    f"Products loaded : {len(products)}"
)

print(
    f"Columns         : {list(products.columns)}"
)


# ============================================================
# PRODUCT IDs
# ============================================================

products = products.copy()

products["product_id"] = np.arange(
    1,
    len(products) + 1
)

products["price_numeric"] = pd.to_numeric(
    products["price"],
    errors="coerce"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ============================================================
# BUILD PRODUCT TEXT
# ============================================================

def build_product_text(row):

    parts = [

        row.get(
            "product_name",
            ""
        ),

        row.get(
            "category",
            ""
        ),

        row.get(
            "brand",
            ""
        ),

        row.get(
            "color",
            ""
        ),

        row.get(
            "material",
            ""
        ),

        row.get(
            "description",
            ""
        )
    ]

    return " ".join(
        clean_text(x)
        for x in parts
        if not pd.isna(x)
    )


products["product_text"] = products.apply(
    build_product_text,
    axis=1
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print(
    "Loading embedding model..."
)

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded."
)


# ============================================================
# CREATE PRODUCT EMBEDDINGS
# ============================================================

print()
print(
    "Creating product embeddings..."
)

product_embeddings = model.encode(
    products["product_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    f"Embedding shape : "
    f"{product_embeddings.shape}"
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_category(value):

    value = clean_text(value)

    return value


# ============================================================
# MATERIAL MATCHING
# ============================================================

def material_matches(
    product_material,
    requested_material
):

    product_material = clean_text(
        product_material
    )

    requested_material = clean_text(
        requested_material
    )

    if not requested_material:
        return True

    # Wood and wooden are equivalent
    if requested_material in [
        "wood",
        "wooden"
    ]:

        return (
            "wood" in product_material
            or
            "wooden" in product_material
        )

    return requested_material in product_material


# ============================================================
# STRUCTURED FILTER
# ============================================================

def apply_structured_filters(
    data,
    parsed_query
):

    filtered = data.copy()

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category = parsed_query.get(
        "category"
    )

    if category:

        filtered = filtered[
            filtered["category"]
            .apply(clean_text)
            ==
            clean_text(category)
        ]

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    brand = parsed_query.get(
        "brand"
    )

    if brand:

        filtered = filtered[
            filtered["brand"]
            .apply(clean_text)
            .str.contains(
                clean_text(brand),
                na=False
            )
        ]

    excluded_brands = parsed_query.get(
        "excluded_brands",
        []
    )

    if excluded_brands:

        normalized_brands = (
            filtered["brand"].fillna("").astype(str)
            + " "
            + filtered["product_name"].fillna("").astype(str)
        ).str.lower().str.replace(
            r"\s+",
            "",
            regex=True
        )

        for excluded_brand in excluded_brands:

            normalized_exclusion = (
                clean_text(excluded_brand)
                .replace(" ", "")
            )

            filtered = filtered[
                ~normalized_brands.str.contains(
                    normalized_exclusion,
                    regex=False,
                    na=False
                )
            ]

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    color = parsed_query.get(
        "color"
    )

    if color:

        filtered = filtered[
            filtered["color"]
            .apply(clean_text)
            .str.contains(
                clean_text(color),
                na=False
            )
        ]

    # --------------------------------------------------------
    # MATERIAL
    # --------------------------------------------------------

    material = parsed_query.get(
        "material"
    )

    if material:

        filtered = filtered[
            filtered["material"]
            .apply(
                lambda x:
                material_matches(
                    x,
                    material
                )
            )
        ]

    # --------------------------------------------------------
    # MINIMUM PRICE
    # --------------------------------------------------------

    min_price = parsed_query.get(
        "min_price"
    )

    if min_price is not None:

        filtered = filtered[
            filtered["price_numeric"]
            >= float(min_price)
        ]

    # --------------------------------------------------------
    # MAXIMUM PRICE
    # --------------------------------------------------------

    max_price = parsed_query.get(
        "max_price"
    )

    if max_price is not None:

        filtered = filtered[
            filtered["price_numeric"]
            <= float(max_price)
        ]

    return filtered


# ============================================================
# STYLE RELEVANCE
# ============================================================

def style_matches(
    row,
    requested_style
):

    if not requested_style:
        return 0.0

    requested_style = clean_text(
        requested_style
    )

    product_text = clean_text(
        row["product_text"]
    )

    style_variants = {

        "modern": [
            "modern",
            "contemporary"
        ],

        "contemporary": [
            "contemporary",
            "modern"
        ],

        "luxury": [
            "luxury",
            "premium"
        ],

        "classic": [
            "classic",
            "traditional"
        ],

        "traditional": [
            "traditional",
            "classic"
        ],

        "minimal": [
            "minimal",
            "minimalist",
            "simple"
        ],

        "minimalist": [
            "minimalist",
            "minimal",
            "simple"
        ],

        "vintage": [
            "vintage",
            "classic"
        ],

        "simple": [
            "simple",
            "minimal",
            "minimalist"
        ]
    }

    variants = style_variants.get(
        requested_style,
        [requested_style]
    )

    for variant in variants:

        if variant in product_text:
            return 1.0

    return 0.0


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_embedding,
    candidate_embeddings
):

    return np.dot(
        candidate_embeddings,
        query_embedding
    )


# ============================================================
# RAG PRODUCT RETRIEVAL
# ============================================================

def retrieve_products(
    query,
    top_k=None
):

    # --------------------------------------------------------
    # STEP 1: QUERY UNDERSTANDING
    # --------------------------------------------------------

    parsed_query = process_query(
        query
    )

    if top_k is None:

        top_k = parsed_query.get(
            "top_k",
            DEFAULT_TOP_K
        )

    print()
    print(
        "Natural Language Query:"
    )

    print(query)

    print()
    print(
        "Query Understanding:"
    )

    print(parsed_query)

    # --------------------------------------------------------
    # STEP 2: STRUCTURED FILTERING
    # --------------------------------------------------------

    filtered = apply_structured_filters(
        products,
        parsed_query
    )

    print()
    print(
        f"Products after structured "
        f"filtering: {len(filtered)}"
    )

    # --------------------------------------------------------
    # NO PRODUCTS AFTER FILTERING
    # --------------------------------------------------------

    if filtered.empty:

        return (
            parsed_query,
            []
        )

    # --------------------------------------------------------
    # STEP 3: QUERY EMBEDDING
    # --------------------------------------------------------

    semantic_query = parsed_query.get(
        "normalized_query",
        str(query).lower()
    )

    for excluded_brand in parsed_query.get(
        "excluded_brands",
        []
    ):

        brand_variants = {
            excluded_brand,
            excluded_brand.replace("homecentre", "home centre")
        }

        for brand_variant in brand_variants:

            exclusion_pattern = (
                r"\b(?:not(?:\s+from)?|except|excluding|exclude|without|"
                r"other\s+(?:than|that)|anything\s+but)\s+(?:the\s+)?"
                + re.escape(brand_variant)
                + r"\b"
            )

            semantic_query = re.sub(
                exclusion_pattern,
                " ",
                semantic_query
            )

    semantic_query = re.sub(
        r"\bother\s+brands?\b|\bbrand\b",
        " ",
        semantic_query
    )

    for field in (
        "category",
        "color",
        "material",
        "style",
        "brand"
    ):

        value = parsed_query.get(field)

        if value and not re.search(
            r"\b" + re.escape(str(value)) + r"\b",
            semantic_query
        ):
            semantic_query += f" {value}"

    semantic_query = re.sub(
        r"\s+",
        " ",
        semantic_query
    ).strip()

    query_embedding = model.encode(
        semantic_query or str(query),
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # STEP 4: GET EMBEDDINGS
    # --------------------------------------------------------

    candidate_indices = (
        filtered.index.to_numpy()
    )

    candidate_embeddings = (
        product_embeddings[
            candidate_indices
        ]
    )

    # --------------------------------------------------------
    # STEP 5: SEMANTIC SIMILARITY
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )

    filtered = filtered.copy()

    filtered["similarity"] = similarities

    # --------------------------------------------------------
    # STEP 6: STYLE RELEVANCE
    # --------------------------------------------------------

    requested_style = parsed_query.get(
        "style"
    )

    filtered["style_relevance"] = (
        filtered.apply(
            lambda row:
            style_matches(
                row,
                requested_style
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # STEP 7: RANKING
    # --------------------------------------------------------

    if requested_style:

        # Products matching requested style
        # are ranked before other semantic matches.
        filtered = filtered.sort_values(
            by=[
                "style_relevance",
                "similarity"
            ],
            ascending=[
                False,
                False
            ]
        )

    else:

        filtered = filtered.sort_values(
            by="similarity",
            ascending=False
        )

    # --------------------------------------------------------
    # STEP 8: SIMILARITY THRESHOLD
    # --------------------------------------------------------

    filtered = filtered[
        filtered["similarity"]
        >= SIMILARITY_THRESHOLD
    ]

    # --------------------------------------------------------
    # STEP 9: TOP K
    # --------------------------------------------------------

    filtered = filtered.head(
        top_k
    )

    # --------------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------------

    results = []

    for _, row in filtered.iterrows():

        results.append({

            "product_id":
                int(row["product_id"]),

            "product_name":
                row["product_name"],

            "category":
                row["category"],

            "brand":
                row["brand"],

            "color":
                row["color"],

            "material":
                row["material"],

            "price":
                row["price"],

            "similarity":
                float(row["similarity"]),

            "style_relevance":
                float(row["style_relevance"])
        })

    return (
        parsed_query,
        results
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    parsed_query,
    results
):

    print()
    print(
        "Extracted Filters"
    )

    print("-" * 60)

    print(
        f"Category : "
        f"{parsed_query.get('category')}"
    )

    print(
        f"Color    : "
        f"{parsed_query.get('color')}"
    )

    print(
        f"Material : "
        f"{parsed_query.get('material')}"
    )

    print(
        f"Style    : "
        f"{parsed_query.get('style')}"
    )

    print(
        f"Brand    : "
        f"{parsed_query.get('brand')}"
    )

    print(
        f"Min Price: "
        f"{parsed_query.get('min_price')}"
    )

    print(
        f"Max Price: "
        f"{parsed_query.get('max_price')}"
    )

    print(
        f"Top K    : "
        f"{parsed_query.get('top_k')}"
    )

    print()
    print(
        "Retrieved Products"
    )

    print("-" * 60)

    if not results:

        print(
            "No matching products found."
        )

        return

    for rank, product in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"{rank}. "
            f"{product['product_name']}"
        )

        print(
            f"   Product ID     : "
            f"{product['product_id']}"
        )

        print(
            f"   Category       : "
            f"{product['category']}"
        )

        print(
            f"   Brand          : "
            f"{product['brand']}"
        )

        print(
            f"   Color          : "
            f"{product['color']}"
        )

        print(
            f"   Material       : "
            f"{product['material']}"
        )

        print(
            f"   Price          : "
            f"{product['price']}"
        )

        print(
            f"   Similarity     : "
            f"{product['similarity']:.4f}"
        )

        print(
            f"   Style Relevance: "
            f"{product['style_relevance']:.1f}"
        )


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "I want a modern black sofa under 30000",

        "Show me an IKEA wooden table below 10000",

        "I need a grey fabric chair between 5000 and 15000",

        "Give me a luxury bed above 20000",

        "Show top 10 modern wardrobes"
    ]

    print()
    print("=" * 60)
    print(
        "TESTING RAG PRODUCT RETRIEVAL"
    )
    print("=" * 60)

    for query in test_queries:

        print()
        print("=" * 60)

        parsed_query, results = (
            retrieve_products(
                query
            )
        )

        display_results(
            parsed_query,
            results
        )

    print()
    print("=" * 60)
    print(
        "RAG PRODUCT RETRIEVAL TEST COMPLETED."
    )
    print("=" * 60)