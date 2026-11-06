import streamlit as st

try:
    from src.chat_panel import render_chat_panel
except ImportError:
    from chat_panel import render_chat_panel


st.set_page_config(
    page_title="AI Furniture Recommender",
    page_icon="🛋️",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu, footer { visibility: hidden; }
    .main-title { font-size: 2.2rem; font-weight: 750; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🛋️ AI Furniture Recommender</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Conversational RAG-based personalized furniture recommendations</div>',
    unsafe_allow_html=True,
)
st.caption("Powered by ChromaDB + Sentence Transformers + Ollama Phi-3")
render_chat_panel()
