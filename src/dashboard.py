from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

# The A/B experiment is explicitly simulated for the Sprint 3 demonstration.
try:
    from src.ab_testing import run_ab_test
except ImportError:
    try:
        from ab_testing import run_ab_test  # supports running from src/
    except ImportError:
        run_ab_test = None

try:
    from src.chat_panel import render_chat_panel
except ImportError:
    from chat_panel import render_chat_panel


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hyper-Personalized AI Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIGURATION
# ============================================================
API_URL = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent.parent
PRODUCT_FILE = BASE_DIR / "data" / "processed" / "processed_products.csv"

METHOD_ENDPOINTS = {
    "Popularity": "/recommend/popularity",
    "Collaborative Filtering": "/recommend/collaborative",
    "Content-Based Embeddings": "/recommend/embedding",
    "Hybrid LTR": "/recommend/hybrid",
    "Conversational RAG": "/recommend/conversational",
}

METHOD_DESCRIPTIONS = {
    "Popularity": "Recommends products using the popularity baseline.",
    "Collaborative Filtering": "Uses user–product interaction behaviour and collaborative signals.",
    "Content-Based Embeddings": "Uses semantic product embeddings to match the requirement.",
    "Hybrid LTR": "Combines recommendation signals with learning-to-rank.",
    "Conversational RAG": "Retrieves relevant products and returns a conversational recommendation.",
}

EXAMPLES = [
    "Modern wooden table",
    "Comfortable fabric sofa",
    "Black office chair",
    "Wood furniture under ₹15000",
]

# ============================================================
# STYLING — retains the original dashboard's visual structure
# ============================================================
st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    #MainMenu, footer { visibility: hidden; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .search-label {
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }
    .product-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.1rem;
        margin-bottom: 0.6rem;
        background: var(--secondary-background-color);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .product-name {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .product-description {
        color: #6b7280;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    .method-box {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: var(--secondary-background-color);
        border: 1px solid #e5e7eb;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .muted { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE AND CALLBACKS
# ============================================================
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "run_recommendation_search" not in st.session_state:
    st.session_state["run_recommendation_search"] = False
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""


def choose_example(example: str) -> None:
    """Streamlit runs this callback before rerendering the text input."""
    st.session_state["search_query"] = example
    st.session_state["run_recommendation_search"] = True


# ============================================================
# API AND DATA HELPERS
# ============================================================
@st.cache_data(ttl=10, show_spinner=False)
def check_api_connection() -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        response.raise_for_status()
        payload = response.json()
        return True, payload if isinstance(payload, dict) else {}
    except (requests.RequestException, ValueError):
        return False, {}


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    if not PRODUCT_FILE.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(PRODUCT_FILE)
        if "product_id" not in frame.columns:
            frame.insert(0, "product_id", range(1, len(frame) + 1))
        return frame
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def call_recommendation_api(
    method: str,
    query: str,
    top_k: int,
    user_id: int,
    category: str | None = None,
    brand: str | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    endpoint = METHOD_ENDPOINTS[method]
    payload = {
        "query": query,
        "top_k": int(top_k),
        "user_id": int(user_id),
    }
    # Include optional filters only when entered.
    if category:
        payload["category"] = category
    if brand:
        payload["brand"] = brand
    if max_price is not None and max_price > 0:
        payload["max_price"] = float(max_price)

    started = time.perf_counter()
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=30 if method == "Conversational RAG" else 15,
    )
    elapsed = time.perf_counter() - started

    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text[:2000]}

    if not response.ok:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        return {
            "success": False,
            "error": detail,
            "status_code": response.status_code,
            "response_time_seconds": elapsed,
        }

    if not isinstance(data, dict):
        return {
            "success": False,
            "error": "The API response was not a JSON object.",
            "status_code": response.status_code,
            "response_time_seconds": elapsed,
        }

    data.setdefault("success", True)
    data.setdefault("status_code", response.status_code)
    data.setdefault("response_time_seconds", elapsed)
    return data


def get_products_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Support common response keys used by recommender APIs."""
    for key in ("products", "recommendations", "items", "results"):
        value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def format_price(price: Any) -> str:
    """Format a product price, showing a placeholder for missing or invalid values."""
    numeric_price = pd.to_numeric(price, errors="coerce")
    if pd.isna(numeric_price):
        return "—"
    return f"₹{numeric_price:,.0f}"


def display_product(product: dict[str, Any], index: int) -> None:
    name = html.escape(str(product.get("product_name") or product.get("name") or "Product"))
    description = html.escape(str(product.get("description") or "No description available."))
    category = str(product.get("category") or "—").replace("_", " ").title()
    brand = str(product.get("brand") or "—")
    color = str(product.get("color") or "—").replace("_", " ").title()
    material = str(product.get("material") or "—").replace("_", " ").title()
    price = product.get("price")
    stock = product.get("stock")

    st.markdown(
        f"""
        <div class="product-card">
            <div class="product-name">{index}. 🪑 {name}</div>
            <div class="product-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(5)
    details = [
        ("Category", category),
        ("Brand", brand),
        ("Color", color),
        ("Material", material),
        ("Price", format_price(price)),
    ]
    for column, (label, value) in zip(columns, details):
        with column:
            st.caption(label)
            st.write(value)

    if stock is not None:
        st.caption(f"Stock: {stock}")
    st.divider()


# ============================================================
# LOAD STATUS
# ============================================================
products_df = load_products()
api_connected, health_data = check_api_connection()


# ============================================================
# SIDEBAR — original settings and controls
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Recommendation Settings")
    method = st.selectbox(
        "Choose recommendation method",
        list(METHOD_ENDPOINTS.keys()),
        index=0,
    )
    top_k = st.slider("Number of products", min_value=3, max_value=10, value=5)
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)

    st.divider()
    st.markdown("### 🔎 Filters")
    category_filter = st.text_input("Category filter (optional)", "")
    brand_filter = st.text_input("Brand filter (optional)", "")
    max_price_filter = st.number_input(
        "Maximum price (0 = no limit)",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

    st.divider()
    st.markdown("### 🔌 System Status")
    if api_connected:
        st.success("● API Connected")
        st.caption(f"Products loaded by API: {health_data.get('products_loaded', 'Unknown')}")
    else:
        st.error("● API Not Connected")
        st.caption("Start FastAPI on port 8000.")

    st.divider()
    st.markdown("### 📊 Dataset")
    st.metric("Local products", len(products_df) if not products_df.empty else "Unavailable")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">🤖 Hyper-Personalized AI Recommendation System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Discover products using preferences, behaviour, '
    'semantic similarity and AI-powered ranking.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# RECOMMENDATION SEARCH AND CONVERSATIONAL RECOMMENDER
# ============================================================
search_column, chat_column = st.columns([1.8, 1], gap="large")

with search_column:
    st.markdown(
        '<div class="search-label">🔎 What are you looking for?</div>',
        unsafe_allow_html=True,
    )
    st.text_input(
        "Your requirement",
        placeholder="Example: modern wooden table, black office chair, comfortable fabric sofa under ₹15000...",
        key="search_query",
        label_visibility="collapsed",
    )

    st.markdown("### 💡 Try an example")
    example_columns = st.columns(4)
    for index, example in enumerate(EXAMPLES):
        example_columns[index].button(
            example,
            key=f"example_{index}",
            use_container_width=True,
            on_click=choose_example,
            args=(example,),
        )

    st.markdown(
        f"""
        <div class="method-box">
            <strong>{html.escape(method)}</strong><br>
            <span class="muted">{html.escape(METHOD_DESCRIPTIONS[method])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    manual_search_clicked = st.button(
        "✨ Get Personalized Recommendations",
        type="primary",
        use_container_width=True,
    )
    search_clicked = manual_search_clicked or st.session_state["run_recommendation_search"]
    st.session_state["run_recommendation_search"] = False

with chat_column:
    render_chat_panel()


# ============================================================
# RECOMMENDATIONS
# ============================================================
if search_clicked:
    clean_query = st.session_state["search_query"].strip()

    if not clean_query:
        st.warning("Please describe what you are looking for.")
    elif not api_connected:
        st.error("The recommendation API is not connected. Start FastAPI on port 8000 first.")
    else:
        try:
            with st.spinner("Finding products that match your requirement..."):
                result = call_recommendation_api(
                    method=method,
                    query=clean_query,
                    top_k=top_k,
                    user_id=int(user_id),
                    category=category_filter.strip() or None,
                    brand=brand_filter.strip() or None,
                    max_price=max_price_filter if max_price_filter > 0 else None,
                )
            st.session_state["last_result"] = result
            st.session_state["last_query"] = clean_query
        except requests.exceptions.Timeout:
            st.error("The recommendation service took too long to respond.")
            st.session_state["last_result"] = None
        except requests.exceptions.RequestException as error:
            st.error(f"Could not connect to FastAPI: {error}")
            st.session_state["last_result"] = None
        except Exception as error:
            st.error(f"Recommendation request failed: {error}")
            st.session_state["last_result"] = None

result = st.session_state.get("last_result")
clean_query = st.session_state.get("last_query", "")

if isinstance(result, dict):
    if not result.get("success", False):
        st.error(f"Recommendation request failed (HTTP {result.get('status_code', 'unknown')}).")
        with st.expander("Technical details"):
            st.write(result.get("error", "Unknown error"))
    else:
        recommendation_products = get_products_from_result(result)
        response_text = result.get("response") or result.get("answer") or ""
        response_time = float(result.get("response_time_seconds") or 0)

        st.divider()
        st.markdown("## ✨ Personalized Recommendations")
        if clean_query:
            st.markdown(f"Results for **{clean_query}**")

        if method == "Conversational RAG" and response_text:
            st.markdown("### 💬 Conversational response")
            st.info(str(response_text))

        st.caption(
            f"Found {len(recommendation_products)} recommendation"
            f"{'' if len(recommendation_products) == 1 else 's'}"
        )

        if recommendation_products:
            for index, product in enumerate(recommendation_products, start=1):
                display_product(product, index)
        else:
            st.warning(
                "The API returned no products. Check the selected endpoint's response "
                "and the FastAPI terminal for backend errors."
            )

        st.markdown("## ⚡ Recommendation Performance")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Method", method)
        metric_columns[1].metric("Response Time", f"{response_time:.3f}s")
        metric_columns[2].metric("Recommendations", len(recommendation_products))
        metric_columns[3].metric(
            "Algorithm",
            str(result.get("algorithm") or result.get("method") or "AI Recommendation"),
        )

        with st.expander("📌 Product Summary"):
            if recommendation_products:
                summary_rows = []
                for product in recommendation_products:
                    summary_rows.append({
                        "Product": product.get("product_name") or product.get("name") or "Product",
                        "Category": product.get("category") or "—",
                        "Brand": product.get("brand") or "—",
                        "Material": product.get("material") or "—",
                        "Price": format_price(product.get("price")),
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.write("No products were returned.")

        with st.expander("View technical details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**API method:**", result.get("method", method))
                st.write("**Algorithm:**", result.get("algorithm", "N/A"))
                st.write("**User ID:**", result.get("user_id", user_id))
            with col2:
                st.write("**Query:**", result.get("query", clean_query))
                st.write("**Results:**", result.get("count", len(recommendation_products)))
                st.write("**Response time:**", f"{response_time:.3f} seconds")


# ============================================================
# A/B TESTING — simulated metrics clearly labelled
# ============================================================
st.divider()
st.markdown("## 🧪 A/B Testing")
st.caption(
    "This assignment experiment uses simulated user-level metrics. "
    "It demonstrates the comparison workflow and is not evidence of real customer behaviour."
)

if run_ab_test is None:
    st.warning("Could not import src.ab_testing.run_ab_test. Check that src/ab_testing.py exists.")
else:
    if st.button("Run simulated A/B experiment", key="run_ab_test"):
        try:
            with st.spinner("Running simulated experiment..."):
                ab_results = run_ab_test()
            summary = ab_results.get("summary", {})
            control = summary.get("control", {})
            treatment = summary.get("treatment", {})

            left, right = st.columns(2)
            with left:
                st.subheader("Control — Popularity")
                st.metric("CTR proxy", f"{control.get('ctr', 0):.3f}")
                st.metric("Add-to-cart likelihood", f"{control.get('add_to_cart_rate', 0):.3f}")
            with right:
                st.subheader("Treatment — Hybrid/RAG")
                st.metric("CTR proxy", f"{treatment.get('ctr', 0):.3f}")
                st.metric("Add-to-cart likelihood", f"{treatment.get('add_to_cart_rate', 0):.3f}")

            lift_cols = st.columns(2)
            ctr_lift = summary.get("ctr_relative_lift_percent")
            cart_lift = summary.get("add_to_cart_relative_lift_percent")
            lift_cols[0].metric(
                "CTR proxy lift",
                f"{ctr_lift:+.2f}%" if ctr_lift is not None else "N/A",
            )
            lift_cols[1].metric(
                "Add-to-cart lift",
                f"{cart_lift:+.2f}%" if cart_lift is not None else "N/A",
            )
            ctr_p_value = summary.get("ctr_p_value")
            cart_p_value = summary.get("add_to_cart_p_value")
            ctr_p_value_text = (
                f"{ctr_p_value:.4f}" if ctr_p_value is not None else "N/A"
            )
            cart_p_value_text = (
                f"{cart_p_value:.4f}" if cart_p_value is not None else "N/A"
            )
            st.caption(
                f"Simulated participants: {summary.get('n_users_per_group', 0)} per group. "
                f"Welch t-test p-values — CTR: {ctr_p_value_text}; "
                f"add-to-cart: {cart_p_value_text}."
            )
            if ab_results.get("csv_path"):
                st.caption(f"Results saved to: {ab_results['csv_path']}")
            if isinstance(ab_results.get("observations"), pd.DataFrame):
                with st.expander("View simulated participant-level results"):
                    st.dataframe(ab_results["observations"], use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"A/B experiment failed: {error}")

st.divider()
st.markdown("### 📚 API Documentation")
st.write(f"Interactive OpenAPI documentation: {API_URL}/docs")
