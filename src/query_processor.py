import re


# ============================================================
# QUERY PROCESSOR
# Sprint 3 - Week 5
# ============================================================


# ------------------------------------------------------------
# Category aliases
# ------------------------------------------------------------

CATEGORY_ALIASES = {
    "sofa": "sofa",
    "sofas": "sofa",

    "chair": "chair",
    "chairs": "chair",

    "table": "table",
    "tables": "table",

    "bed": "bed",
    "beds": "bed",

    "wardrobe": "wardrobe",
    "wardrobes": "wardrobe",
    "wardrope": "wardrobe",
    "wardroop": "wardrobe",
    "waardroop": "wardrobe",
    "wardroom": "wardrobe",
    "wardroob": "wardrobe",

    "lamp": "lamp",
    "lamps": "lamp",

    "desk": "desk",
    "desks": "desk",

    "bookshelf": "bookshelf",
    "bookshelves": "bookshelf",

    "cabinet": "cabinet",
    "cabinets": "cabinet",

    "dining table": "table",
    "dining tables": "table",

    "office chair": "chair",
    "office chairs": "chair",
    "ofice chair": "chair",
    "ofice chairs": "chair"
}


# ------------------------------------------------------------
# Known colors
# ------------------------------------------------------------

COLORS = [
    "black",
    "white",
    "grey",
    "gray",
    "brown",
    "blue",
    "red",
    "green",
    "yellow",
    "beige",
    "cream",
    "orange",
    "pink",
    "purple"
]


# ------------------------------------------------------------
# Known materials
# ------------------------------------------------------------

MATERIALS = [
    "engineered wood",
    "wood",
    "wooden",
    "wodden",
    "metal",
    "metel",
    "metalic",
    "glass",
    "plastic",
    "fabric",
    "velvet",
    "leather",
    "marble",
    "steel"
]


# ------------------------------------------------------------
# Known styles
# ------------------------------------------------------------

STYLES = [
    "modern",
    "classic",
    "luxury",
    "minimal",
    "minimalist",
    "traditional",
    "contemporary",
    "vintage",
    "simple"
]


# ------------------------------------------------------------
# Known brands
# ------------------------------------------------------------

BRANDS = [
    "ikea",
    "godrej",
    "durian",
    "urban ladder",
    "urban living",
    "hometown",
    "nilkamal",
    "wakefit",
    "homecentre",
    "home centre"
]


# ============================================================
# NORMALIZE QUERY
# ============================================================

def normalize_query(query):

    if query is None:
        return ""

    query = str(query).lower().strip()

    typo_replacements = {
        r"\bwardrope\b": "wardrobe",
        r"\bwardroop\b": "wardrobe",
        r"\bwaardroop\b": "wardrobe",
        r"\bwardroom\b": "wardrobe",
        r"\bwardroob\b": "wardrobe",
        r"\bmetel\b": "metal",
        r"\bmetalic\b": "metal",
        r"\bwodden\b": "wood",
        r"\bwooden\b": "wood",
    }

    for pattern, replacement in typo_replacements.items():
        query = re.sub(pattern, replacement, query)

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


# ============================================================
# EXTRACT CATEGORY
# ============================================================

def extract_category(query):

    query = normalize_query(query)

    # Check longer phrases first
    categories = sorted(
        CATEGORY_ALIASES.keys(),
        key=len,
        reverse=True
    )

    for category in categories:

        pattern = (
            r"\b"
            + re.escape(category)
            + r"\b"
        )

        if re.search(
            pattern,
            query
        ):

            return CATEGORY_ALIASES[
                category
            ]

    return None


# ============================================================
# EXTRACT COLOR
# ============================================================

def extract_color(query):

    query = normalize_query(query)

    for color in COLORS:

        if re.search(
            r"\b"
            + re.escape(color)
            + r"\b",
            query
        ):

            if color == "gray":
                return "grey"

            return color

    return None


# ============================================================
# EXTRACT MATERIAL
# ============================================================

def extract_material(query):

    query = normalize_query(query)

    sorted_materials = sorted(
        MATERIALS,
        key=len,
        reverse=True
    )

    for material in sorted_materials:

        if re.search(
            r"\b"
            + re.escape(material)
            + r"\b",
            query
        ):

            if material in {"wooden", "wodden"}:
                return "wood"

            if material in {"metel", "metalic"}:
                return "metal"

            return material

    return None


# ============================================================
# EXTRACT STYLE
# ============================================================

def extract_style(query):

    query = normalize_query(query)

    for style in STYLES:

        if re.search(
            r"\b"
            + re.escape(style)
            + r"\b",
            query
        ):

            if style == "minimalist":
                return "minimal"

            return style

    return None


# ============================================================
# EXTRACT BRAND
# ============================================================

def extract_brand(query):

    query = normalize_query(query)

    sorted_brands = sorted(
        BRANDS,
        key=len,
        reverse=True
    )

    for brand in sorted_brands:

        if re.search(
            r"\b"
            + re.escape(brand)
            + r"\b",
            query
        ):

            if brand == "home centre":
                return "homecentre"

            return brand

    return None


def extract_excluded_brands(query):

    query = normalize_query(query)
    excluded_brands = []

    exclusion_pattern = (
        r"\b(?:not(?:\s+from)?|except|excluding|exclude|without|"
        r"other\s+(?:than|that)|anything\s+but)\s+(?:the\s+)?"
    )

    for brand in sorted(BRANDS, key=len, reverse=True):

        pattern = (
            exclusion_pattern
            + re.escape(brand)
            + r"\b"
        )

        if re.search(pattern, query):
            normalized_brand = (
                "homecentre"
                if brand == "home centre"
                else brand
            )

            if normalized_brand not in excluded_brands:
                excluded_brands.append(normalized_brand)

    return excluded_brands


# ============================================================
# EXTRACT MAXIMUM PRICE
# ============================================================

def extract_max_price(query):

    query = normalize_query(query)

    query = query.replace(
        ",",
        ""
    )

    patterns = [

        r"(?:under|below|less than|upto|up to|within|maximum|max|budget)"
        r"\s*(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)",

        r"(?:₹|rs\.?|inr)"
        r"\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except ValueError:
                pass

    return None


# ============================================================
# EXTRACT MINIMUM PRICE
# ============================================================

def extract_min_price(query):

    query = normalize_query(query)

    query = query.replace(
        ",",
        ""
    )

    patterns = [

        r"(?:above|over|more than|starting from|minimum|min)"
        r"\s*(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except ValueError:
                pass

    return None


# ============================================================
# EXTRACT PRICE RANGE
# ============================================================

def extract_price_range(query):

    query = normalize_query(query)

    query = query.replace(
        ",",
        ""
    )

    patterns = [

        r"between\s*"
        r"(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*(?:and|to|-)\s*"
        r"(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)",

        r"(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*(?:to|-)\s*"
        r"(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            try:

                min_price = float(
                    match.group(1)
                )

                max_price = float(
                    match.group(2)
                )

                return (
                    min_price,
                    max_price
                )

            except ValueError:
                pass

    return None


# ============================================================
# EXTRACT TOP K
# ============================================================

def extract_top_k(query):

    query = normalize_query(query)

    patterns = [

        # Examples:
        # top 5
        # top 10
        r"\btop\s+(\d+)",

        # Examples:
        # show 5
        # show 10
        r"\bshow\s+(\d+)",

        # Examples:
        # show only 3
        # show me only 3
        # show just 3
        r"\bshow(?:\s+me)?\s+(?:only|just)\s+(\d+)",

        # Examples:
        # give me 5
        # give me 10
        r"\bgive\s+me\s+(\d+)",

        # Examples:
        # give me only 3
        # give me just 3
        r"\bgive\s+me\s+(?:only|just)\s+(\d+)",

        # Examples:
        # only 3
        # just 3
        r"\b(?:only|just)\s+(\d+)",

        # Examples:
        # first 5
        r"\bfirst\s+(\d+)",

        # Examples:
        # 5 recommendations
        # 10 recommendations
        r"\b(\d+)\s+recommendations?",

        # Examples:
        # 3 products
        # 5 products
        r"\b(\d+)\s+products?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            try:

                value = int(
                    match.group(1)
                )

                if value > 0:

                    return min(
                        value,
                        20
                    )

            except ValueError:
                pass

    return 5


# ============================================================
# PROCESS COMPLETE QUERY
# ============================================================

def process_query(query):

    normalized_query = normalize_query(
        query
    )

    price_range = extract_price_range(
        normalized_query
    )

    if price_range is not None:

        min_price = price_range[0]
        max_price = price_range[1]

    else:

        min_price = extract_min_price(
            normalized_query
        )

        max_price = extract_max_price(
            normalized_query
        )

    excluded_brands = extract_excluded_brands(
        normalized_query
    )
    brand = extract_brand(normalized_query)
    if brand in excluded_brands:
        brand = None

    result = {

        "original_query":
            query,

        "normalized_query":
            normalized_query,

        "category":
            extract_category(
                normalized_query
            ),

        "color":
            extract_color(
                normalized_query
            ),

        "material":
            extract_material(
                normalized_query
            ),

        "style":
            extract_style(
                normalized_query
            ),

        "brand":
            brand,

        "excluded_brands":
            excluded_brands,

        "min_price":
            min_price,

        "max_price":
            max_price,

        "top_k":
            extract_top_k(
                normalized_query
            )
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_query_result(result):

    print()

    print(
        "=" * 65
    )

    print(
        "QUERY PROCESSING RESULT"
    )

    print(
        "=" * 65
    )

    print(
        "Original query :",
        result["original_query"]
    )

    print(
        "Normalized     :",
        result["normalized_query"]
    )

    print(
        "Category       :",
        result["category"]
    )

    print(
        "Color          :",
        result["color"]
    )

    print(
        "Material       :",
        result["material"]
    )

    print(
        "Style          :",
        result["style"]
    )

    print(
        "Brand          :",
        result["brand"]
    )

    print(
        "Minimum price  :",
        result["min_price"]
    )

    print(
        "Maximum price  :",
        result["max_price"]
    )

    print(
        "Top K          :",
        result["top_k"]
    )

    print(
        "=" * 65
    )


# ============================================================
# TEST QUERIES
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "I want a modern black sofa under 30000",

        "Show me an IKEA wooden table below 10000",

        "I need a grey fabric chair between 5000 and 15000",

        "Give me a luxury bed above 20000",

        "Show top 10 modern wardrobes",

        "I want a black sofa from IKEA",

        "Give me 3 wooden beds",

        "Show me modern sofas",

        "I need white chairs",

        "Give me IKEA tables under 15000",

        # New TOP-K tests
        "Show only 3",

        "Show me only 3",

        "Only 3",

        "Give me only 3",

        "Give me just 3",

        "Show just 3",

        "Show 3 products"
    ]

    print()

    print(
        "=" * 65
    )

    print(
        "NATURAL LANGUAGE QUERY PROCESSOR"
    )

    print(
        "=" * 65
    )

    for query in test_queries:

        result = process_query(
            query
        )

        print_query_result(
            result
        )

    print()

    print(
        "Query processor test completed."
    )
    