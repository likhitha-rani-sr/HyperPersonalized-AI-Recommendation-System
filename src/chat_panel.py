from __future__ import annotations

import importlib
import html
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any

import streamlit as st


def _load_backend() -> Any:
    try:
        return importlib.import_module("src.conversational_rag")
    except ModuleNotFoundError as error:
        if error.name != "src":
            raise
        return importlib.import_module("conversational_rag")


@st.cache_resource
def _backend_future() -> Future[Any]:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat-recommender")
    return executor.submit(_load_backend)


def _backend() -> Any:
    module = _backend_future().result()
    st.session_state["_chat_backend_loaded"] = True
    return module


def _is_greeting(text: str) -> bool:
    return text.lower().strip() in {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
    }


def _conversation_title(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = str(message.get("content", "New Conversation"))
            return text[:35] + "..." if len(text) > 35 else text
    return "New Conversation"


def _save_conversation() -> None:
    messages = st.session_state["chat_messages"]
    if not messages:
        return

    conversation_id = st.session_state["active_chat_conversation"]
    if conversation_id is None:
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        st.session_state["active_chat_conversation"] = conversation_id

    preferences = st.session_state.get("chat_preferences", {})
    if st.session_state.get("_chat_backend_loaded"):
        preferences = dict(_backend().current_preferences)
        st.session_state["chat_preferences"] = preferences

    st.session_state["chat_conversations"][conversation_id] = {
        "title": _conversation_title(messages),
        "messages": list(messages),
        "preferences": preferences,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_new_conversation() -> None:
    _save_conversation()
    if st.session_state.get("_chat_backend_loaded"):
        _backend().reset_conversation()
    st.session_state["chat_messages"] = []
    st.session_state["active_chat_conversation"] = None
    st.session_state["chat_preferences"] = {}


def _load_conversation(conversation_id: str) -> None:
    conversation = st.session_state["chat_conversations"].get(conversation_id)
    if conversation is None:
        return

    messages = list(conversation.get("messages", []))
    preferences = dict(conversation.get("preferences", {}))
    _backend().restore_conversation(messages, preferences)
    st.session_state["chat_messages"] = messages
    st.session_state["active_chat_conversation"] = conversation_id
    st.session_state["chat_preferences"] = preferences


def _render_message_products(
    products: list[dict[str, Any]],
    title: str = "Recommended products",
) -> None:
    if not products:
        return
    if title:
        st.markdown(f"**{title}**")
    for index, product in enumerate(products, start=1):
        name = html.escape(
            str(product.get("product_name") or product.get("name") or "Product")
        )
        brand = html.escape(str(product.get("brand") or ""))
        raw_price = product.get("price")
        price = html.escape(str(raw_price if raw_price is not None else "Price unavailable"))
        details = [
            str(product.get(field)).strip()
            for field in ("color", "material", "dimensions")
            if product.get(field)
        ]
        variant = html.escape(" · ".join(details))
        alternative_reason = product.get("alternative_reason")
        reason = (
            f"  \n{html.escape(str(alternative_reason))}"
            if alternative_reason
            else ""
        )
        st.markdown(
            f"**{index}. {name}**  \n"
            f"{brand + ' · ' if brand else ''}"
            f"{variant + ' · ' if variant else ''}₹ {price}"
            f"{reason}"
        )


def render_chat_panel() -> None:
    st.markdown("### 💬 Conversational recommender")
    st.caption("Describe what you need and refine your preferences as you chat.")
    backend_future = _backend_future()
    if not backend_future.done():
        st.caption("Warming up the recommendation engine in the background…")

    for key, initial_value in (
        ("chat_messages", []),
        ("chat_conversations", {}),
        ("active_chat_conversation", None),
        ("chat_preferences", {}),
    ):
        if key not in st.session_state:
            st.session_state[key] = initial_value

    if st.button("➕ New conversation", key="new_chat_conversation", use_container_width=True):
        _start_new_conversation()

    with st.expander("Conversation history", expanded=False):
        conversations = list(st.session_state["chat_conversations"].items())
        conversations.reverse()
        if not conversations:
            st.caption("No previous conversations yet.")
        for conversation_id, conversation in conversations:
            if st.button(
                conversation.get("title", "Conversation"),
                key=f"chat_history_{conversation_id}",
                use_container_width=True,
            ):
                _load_conversation(conversation_id)

    for message in st.session_state["chat_messages"]:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(str(message.get("content", "")))
            if role == "assistant":
                products = message.get("products", [])
                if isinstance(products, list):
                    _render_message_products(
                        [product for product in products if isinstance(product, dict)]
                    )
                alternative_products = message.get("alternative_products", [])
                if isinstance(alternative_products, list):
                    _render_message_products(
                        [
                            product
                            for product in alternative_products
                            if isinstance(product, dict)
                        ],
                        title="",
                    )

    with st.form("conversational_recommender_form", clear_on_submit=True):
        user_input = st.text_input(
            "Message",
            placeholder="e.g. A comfortable blue sofa under ₹25000",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)

    if not submitted:
        return
    user_input = user_input.strip()
    if not user_input:
        st.warning("Enter a message to get recommendations.")
        return

    st.session_state["chat_messages"].append(
        {"role": "user", "content": user_input}
    )
    started = time.perf_counter()
    answer = ""
    products: list[dict[str, Any]] = []
    alternative_products: list[dict[str, Any]] = []

    if _is_greeting(user_input):
        answer = (
            "Hello! I can help you find furniture based on category, color, brand, "
            "material, style, and budget."
        )
    else:
        try:
            with st.spinner("Finding recommendations…"):
                result = _backend().recommend_for_ui(user_input)
            if result is None:
                answer = "I couldn't process that request. Please try another furniture query."
            else:
                answer = str(result.get("answer") or result.get("response") or "")
                result_products = result.get("products", [])
                if isinstance(result_products, list):
                    products = [
                        product for product in result_products if isinstance(product, dict)
                    ]
                result_alternatives = result.get("alternative_products", [])
                if isinstance(result_alternatives, list):
                    alternative_products = [
                        product
                        for product in result_alternatives
                        if isinstance(product, dict)
                    ]
                if not answer:
                    answer = "I couldn't find a suitable response for that request."
        except Exception as error:
            answer = "Sorry, something went wrong while processing your request."
            st.error(f"Conversational recommendation failed: {error}")

    elapsed = time.perf_counter() - started
    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "products": products,
            "alternative_products": alternative_products,
        }
    )
    st.session_state["chat_preferences"] = (
        dict(_backend().current_preferences)
        if st.session_state.get("_chat_backend_loaded")
        else {}
    )
    st.caption(f"Response time: {elapsed:.2f}s")
    _save_conversation()
    st.rerun()
