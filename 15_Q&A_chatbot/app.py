# ============================================================
# PROFESSIONAL STREAMLIT UI
# Domain Specific Q&A Chatbot using TF-IDF + Cosine Similarity
# ============================================================

import os
import re
import pickle
import html
import textwrap
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Domain Specific Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 8rem;
}

.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 8px;
}

.main-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 32px;
}

.custom-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
}

.card-title {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 8px;
}

.card-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

.meta-box {
    background: #020617;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 14px;
    margin-top: 15px;
    color: #cbd5e1;
    line-height: 1.6;
    font-size: 14px;
}

.match-card, .history-card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 14px;
}

.match-title, .history-title {
    color: #38bdf8;
    font-weight: 800;
    margin-bottom: 10px;
}

[data-testid="stChatMessage"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 14px;
}

[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 72%;
    z-index: 999;
    background: rgba(11, 17, 32, 0.96);
    padding: 10px 14px;
    border-radius: 30px;
    border: 1px solid #1f2937;
    box-shadow: 0 10px 40px rgba(0,0,0,0.45);
}

[data-testid="stChatInput"] > div {
    display: flex;
    align-items: center;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f8fafc !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    min-height: 28px !important;
    max-height: 160px !important;
    padding-left: 10px !important;
    padding-top: 12px !important;
    padding-bottom: 12px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    min-height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    border: none !important;
}

.stSlider label {
    color: #e5e7eb !important;
}

div[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
}

.stButton button {
    background: #2563eb !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
}

@media (max-width: 768px) {
    [data-testid="stChatInput"] {
        width: 94%;
        bottom: 12px;
    }

    .main-title {
        font-size: 32px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTION FOR SAFE HTML
# ============================================================

def safe_text(text):
    return html.escape(str(text))


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# DATASET CLEANING
# ============================================================

def clean_dataset(df):
    df = df.copy()

    required_columns = ["domain", "question", "answer"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    df = df[required_columns]
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df["question"] = df["question"].apply(preprocess_text)
    df = df[df["question"] != ""]
    df = df.reset_index(drop=True)

    return df


# ============================================================
# TF-IDF
# ============================================================

def create_tfidf_features(df):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        stop_words="english"
    )

    question_vectors = vectorizer.fit_transform(df["question"])

    return vectorizer, question_vectors


# ============================================================
# LOAD SAVED FILES
# ============================================================

@st.cache_resource
def load_saved_files():
    df = pd.read_csv(os.path.join("models", "clean_qa_dataset.csv"))

    with open(os.path.join("models", "tfidf_vectorizer.pkl"), "rb") as file:
        vectorizer = pickle.load(file)

    with open(os.path.join("models", "question_vectors.pkl"), "rb") as file:
        question_vectors = pickle.load(file)

    return df, vectorizer, question_vectors


# ============================================================
# MATCHING FUNCTIONS
# ============================================================

def get_top_matches(user_query, df, vectorizer, question_vectors, top_k=3):
    clean_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([clean_query])

    similarity_scores = cosine_similarity(
        query_vector,
        question_vectors
    ).flatten()

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    top_matches = []

    for index in top_indices:
        top_matches.append({
            "matched_question": df.iloc[index]["question"],
            "answer": df.iloc[index]["answer"],
            "similarity_score": round(float(similarity_scores[index]), 3)
        })

    return top_matches


def chatbot_response(user_query, df, vectorizer, question_vectors, threshold=0.30, top_k=3):
    top_matches = get_top_matches(
        user_query=user_query,
        df=df,
        vectorizer=vectorizer,
        question_vectors=question_vectors,
        top_k=top_k
    )

    best_match = top_matches[0]

    if best_match["similarity_score"] < threshold:
        return {
            "answer": "Sorry, I don't understand your question. Please ask something related to AI, Machine Learning, Deep Learning, NLP, or Data Science.",
            "matched_question": None,
            "similarity_score": best_match["similarity_score"],
            "top_matches": top_matches
        }

    return {
        "answer": best_match["answer"],
        "matched_question": best_match["matched_question"],
        "similarity_score": best_match["similarity_score"],
        "top_matches": top_matches
    }


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "top_matches" not in st.session_state:
    st.session_state.top_matches = []


# ============================================================
# MAIN HEADING
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Domain Specific Q&A Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">AI and Data Science Educational Assistant using TF-IDF and Cosine Similarity</div>',
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS
# ============================================================

with st.expander("⚙️ Project Settings", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05
        )

    with col2:
        top_k = st.slider(
            "Top Matches",
            min_value=1,
            max_value=5,
            value=3
        )


# ============================================================
# LOAD DATA
# ============================================================

try:
    df, vectorizer, question_vectors = load_saved_files()

except Exception as e:
    st.error("Model files not found or dataset format is incorrect.")
    st.code(str(e))
    st.stop()


# ============================================================
# WELCOME CARD
# ============================================================

if len(st.session_state.messages) == 0:
    st.markdown(
        textwrap.dedent("""
        <div class="custom-card">
            <div class="card-title">Start a Conversation</div>
            <div class="card-text">
                Ask any question related to Artificial Intelligence, Machine Learning,
                Deep Learning, NLP, or Data Science.
                <br><br>
                Example: <b>What is machine learning?</b>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and message.get("matched_question") is not None:
            st.markdown(
                textwrap.dedent(f"""
                <div class="meta-box">
                    <b>Matched Question:</b> {safe_text(message["matched_question"])}<br>
                    <b>Similarity Score:</b> {safe_text(message["similarity_score"])}
                </div>
                """),
                unsafe_allow_html=True
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    placeholder="Ask anything about AI, Machine Learning, NLP, Deep Learning, or Data Science..."
)


# ============================================================
# HANDLE USER QUERY
# ============================================================

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    response = chatbot_response(
        user_query=user_query,
        df=df,
        vectorizer=vectorizer,
        question_vectors=question_vectors,
        threshold=threshold,
        top_k=top_k
    )

    st.session_state.top_matches = response["top_matches"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "matched_question": response["matched_question"],
        "similarity_score": response["similarity_score"]
    })

    st.rerun()


# ============================================================
# TOP MATCHED QUESTIONS
# ============================================================

if len(st.session_state.top_matches) > 0:
    with st.expander("🔎 Top Matched Questions", expanded=False):
        for i, match in enumerate(st.session_state.top_matches, start=1):
            st.markdown(
                textwrap.dedent(f"""
                <div class="match-card">
                    <div class="match-title">Match {i}</div>
                    <b>Question:</b> {safe_text(match["matched_question"])}<br>
                    <b>Similarity Score:</b> {safe_text(match["similarity_score"])}<br><br>
                    <b>Answer:</b> {safe_text(match["answer"])}
                </div>
                """),
                unsafe_allow_html=True
            )


# ============================================================
# CHAT HISTORY
# ============================================================

if len(st.session_state.messages) > 0:
    with st.expander("📜 Chat History", expanded=False):
        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                user_msg = st.session_state.messages[i]["content"]
                bot_msg = st.session_state.messages[i + 1]["content"]
                score = st.session_state.messages[i + 1].get("similarity_score", 0)

                st.markdown(
                    textwrap.dedent(f"""
                    <div class="history-card">
                        <div class="history-title">Conversation {(i // 2) + 1}</div>
                        <b>User:</b> {safe_text(user_msg)}<br>
                        <b>Bot:</b> {safe_text(bot_msg)}<br>
                        <b>Similarity Score:</b> {safe_text(score)}
                    </div>
                    """),
                    unsafe_allow_html=True
                )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.top_matches = []
            st.rerun()