import os
import torch
import pandas as pd
import streamlit as st
from datetime import datetime
from transformers import pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GPT-Style AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_MODEL = "gpt2"
CUSTOM_MODEL_PATH = "models/fine_tuned_gpt2"
OUTPUT_PATH = "outputs/chat_history.csv"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# ============================================================
# CSS - GPT STYLE DARK UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #101b3d 0%, #020617 45%, #000000 100%);
        color: #e5e7eb;
    }

    header, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 9rem;
    }

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 900;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #60a5fa, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 24px;
    }

    .chat-scroll {
        height: 520px;
        overflow-y: auto;
        padding: 20px 10px 20px 10px;
        border-radius: 22px;
        background: transparent;
        scroll-behavior: smooth;
    }

    .chat-scroll::-webkit-scrollbar {
        width: 8px;
    }

    .chat-scroll::-webkit-scrollbar-track {
        background: #020617;
    }

    .chat-scroll::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }

    .user-message {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        margin-bottom: 16px;
        max-width: 75%;
        width: fit-content;
        font-size: 16px;
        line-height: 1.7;
        box-shadow: 0 12px 30px rgba(37,99,235,0.3);
    }

    .ai-message {
        background: rgba(2,6,23,0.96);
        border: 1px solid rgba(56,189,248,0.18);
        color: #e2e8f0;
        padding: 16px 18px;
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        margin-bottom: 16px;
        max-width: 75%;
        width: fit-content;
        font-size: 16px;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    .settings-box {
        padding-top: 12px;
    }

    .stTextArea textarea {
        background: #020617 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96,165,250,0.35) !important;
        border-radius: 18px !important;
        font-size: 16px !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background: #1f2937 !important;
        color: white !important;
        border-radius: 14px !important;
        border: 1px solid rgba(96,165,250,0.25) !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 800 !important;
        width: 100%;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(124,58,237,0.35);
    }

    label, p, span {
        color: #cbd5e1 !important;
    }

    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: min(1100px, 92%);
        background: linear-gradient(180deg, rgba(2,6,23,0.4), #020617 35%);
        padding: 14px 20px 18px 20px;
        z-index: 999;
    }

    .empty-chat {
        text-align: center;
        color: #64748b;
        padding-top: 170px;
        font-size: 17px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_name):
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device
    )


# ============================================================
# PROMPT TEMPLATE
# ============================================================

def build_prompt(user_input, prompt_style):
    if prompt_style == "Short Answer":
        return f"Answer briefly and clearly.\n\nQuestion: {user_input}\n\nAnswer:"
    elif prompt_style == "Educational":
        return f"Explain this in simple educational words.\n\nQuestion: {user_input}\n\nAnswer:"
    elif prompt_style == "Creative":
        return f"Write a creative response.\n\nPrompt: {user_input}\n\nResponse:"
    else:
        return f"Answer professionally and clearly.\n\nQuestion: {user_input}\n\nAnswer:"


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    generator,
    user_input,
    prompt_style,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty
):
    prompt = build_prompt(user_input, prompt_style)

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        do_sample=True,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    response = output[0]["generated_text"].strip()

    stop_words = ["User:", "Question:", "Prompt:"]
    for stop_word in stop_words:
        if stop_word in response:
            response = response.split(stop_word)[0].strip()

    return response


# ============================================================
# SAVE CHAT
# ============================================================

def save_chat(model_name, prompt, response):
    os.makedirs("outputs", exist_ok=True)

    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "prompt": prompt,
        "response": response
    }])

    if os.path.exists(OUTPUT_PATH):
        row.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# SEND MESSAGE CALLBACK
# ============================================================

def send_message():
    user_text = st.session_state.user_input.strip()

    if not user_text:
        return

    if st.session_state.model_option == "Custom Fine-Tuned Waiter Model":
        if os.path.exists(CUSTOM_MODEL_PATH):
            selected_model = CUSTOM_MODEL_PATH
            model_label = "Custom Fine-Tuned Waiter Model"
        else:
            selected_model = BASE_MODEL
            model_label = "Simple GPT-2"
    else:
        selected_model = BASE_MODEL
        model_label = "Simple GPT-2"

    generator = load_model(selected_model)

    response = generate_response(
        generator=generator,
        user_input=user_text,
        prompt_style=st.session_state.prompt_style,
        max_new_tokens=st.session_state.max_new_tokens,
        temperature=st.session_state.temperature,
        top_k=st.session_state.top_k,
        top_p=st.session_state.top_p,
        repetition_penalty=st.session_state.repetition_penalty
    )

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": response})

    save_chat(model_label, user_text, response)

    st.session_state.user_input = ""


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">GPT-Style AI Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Professional text generation and conversational AI system powered by Hugging Face Transformers.</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

chat_col, settings_col = st.columns([2.5, 1])

with settings_col:
    st.markdown('<div class="settings-box">', unsafe_allow_html=True)

    st.selectbox(
        "Select Model",
        ["Simple GPT-2", "Custom Fine-Tuned Waiter Model"],
        key="model_option"
    )

    st.selectbox(
        "Prompt Style",
        ["Professional", "Educational", "Creative", "Short Answer"],
        key="prompt_style"
    )

    st.slider("Max New Tokens", 20, 250, 100, key="max_new_tokens")
    st.slider("Temperature", 0.1, 1.5, 0.7, key="temperature")
    st.slider("Top-K", 10, 100, 40, key="top_k")
    st.slider("Top-P", 0.1, 1.0, 0.9, key="top_p")
    st.slider("Repetition Penalty", 1.0, 2.0, 1.2, key="repetition_penalty")

    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.user_input = ""
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


with chat_col:
    st.markdown('<div class="chat-scroll" id="chat-scroll">', unsafe_allow_html=True)

    if len(st.session_state.messages) == 0:
        st.markdown(
            '<div class="empty-chat">Start a new conversation...</div>',
            unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-message">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ai-message">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FIXED INPUT BAR
# ============================================================

st.markdown('<div class="fixed-input">', unsafe_allow_html=True)

input_col, button_col = st.columns([5, 1])

with input_col:
    st.text_area(
        "Message",
        key="user_input",
        height=90,
        placeholder="Ask anything...",
        label_visibility="collapsed"
    )

with button_col:
    st.button("Send", on_click=send_message)

st.markdown('</div>', unsafe_allow_html=True)