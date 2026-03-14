import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import uuid

# Inject Streamlit secrets into environment before importing modules that use them
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.chatbot import get_reply

st.set_page_config(
    page_title="Toyota Canarias — Sofia",
    page_icon="🚗",
    layout="wide",
)

# ── shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --toyota-red: #EB0A1E;
        --toyota-dark: #1a1a1a;
        --toyota-light-red: #ff4d5e;
    }
    .stApp { background-color: #0f0f0f; }
    .block-container { padding-top: 2rem; }
    h1 { color: #EB0A1E !important; font-weight: 800 !important; letter-spacing: 1px; }
    .stCaption p { color: #aaaaaa !important; }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1e1e1e;
        border-left: 3px solid #EB0A1E;
        border-radius: 8px;
        padding: 0.5rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #1a0a0a;
        border-left: 3px solid #555;
        border-radius: 8px;
        padding: 0.5rem;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #EB0A1E !important;
        border-radius: 8px !important;
    }
    .online-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background-color: #1e1e1e; border: 1px solid #EB0A1E;
        border-radius: 20px; padding: 6px 16px;
        color: #ffffff; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 1rem;
    }
    .online-dot {
        width: 10px; height: 10px; background-color: #22c55e;
        border-radius: 50%; display: inline-block;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%   { opacity: 1; transform: scale(1); }
        50%  { opacity: 0.5; transform: scale(1.3); }
        100% { opacity: 1; transform: scale(1); }
    }
    hr { border-color: #333 !important; }
    [data-testid="stSidebar"] { background-color: #1a1a1a !important; border-right: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ── sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#EB0A1E;font-size:1.2rem;margin-bottom:0.5rem'>🚗 Toyota Canarias</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#333;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        options=["💬 Chat with Sofia", "📊 Dealership Intelligence"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#333;margin:1rem 0 0.5rem 0'>", unsafe_allow_html=True)
    st.caption("Toyota Canarias AI Platform")

# ── route pages ───────────────────────────────────────────────────────────────
if page == "📊 Dealership Intelligence":
    from app.dashboard import render_dashboard
    render_dashboard()

else:
    # ── Chat with Sofia ───────────────────────────────────────────────────────
    st.title("Toyota Canarias")
    st.caption("Your personal Toyota advisor for the Canary Islands")

    st.markdown("""
    <div class="online-badge">
        <span class="online-dot"></span>
        Sofia is online
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola! I'm Sofia, your Toyota Canarias advisor. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask Sofia anything about Toyota..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sofia is thinking..."):
                reply = get_reply(prompt, st.session_state.history, st.session_state.session_id)
            st.write(reply)

        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.session_state.messages.append({"role": "assistant", "content": reply})
