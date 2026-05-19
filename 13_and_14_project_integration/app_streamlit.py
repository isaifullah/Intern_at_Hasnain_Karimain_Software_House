import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import spacy
from transformers import pipeline

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hybrid AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIMPLE CLEAN CSS
# ============================================================

st.markdown("""
<style>

#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background-color: #212121;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
    max-width: 900px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #171717;
}

/* Fixed Chat Input */
div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    left: calc(50% + 130px);
    transform: translateX(-50%);
    width: min(820px, calc(100vw - 360px));
    z-index: 9999;
}

div[data-testid="stChatInput"] textarea {
    background-color: #2f2f2f !important;
    color: white !important;
    border-radius: 20px !important;
    border: 1px solid #444 !important;
}

@media (max-width: 900px) {

    div[data-testid="stChatInput"] {
        left: 50%;
        width: 92vw;
    }
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    nlp = spacy.load("en_core_web_sm")

    MODEL_DIR = "models"

    intent_model = joblib.load(
        os.path.join(MODEL_DIR, "best_model.pkl")
    )

    tfidf_vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    return (
        nlp,
        intent_model,
        tfidf_vectorizer,
        label_encoder,
        zero_shot_classifier
    )


(
    nlp,
    intent_model,
    tfidf_vectorizer,
    label_encoder,
    zero_shot_classifier
) = load_models()

# ============================================================
# RESPONSES
# ============================================================

responses = {

    "greeting":
        "Hello! How can I help you today?",

    "goodbye":
        "Goodbye! Have a great day.",

    "thanks":
        "You're welcome. Let me know if you need anything else.",

    "food_order":
        "Sure, I can help you with food ordering. What would you like to order?",

    "weather_query":
        "Please tell me your city so I can help with the weather information.",

    "password_reset":
        "You can reset your password using the Forgot Password option.",

    "payment_issue":
        "Please explain your payment issue.",

    "order_status":
        "Please provide your order ID or tracking number.",

    "technical_support":
        "Please explain the technical issue you are facing.",

    "service_info":
        "We provide support, payments, order tracking, returns, and technical assistance.",

    "business_hours":
        "Our support team is available from 9 AM to 6 PM, Monday to Saturday.",

    "cancellation":
        "Please confirm if you want to cancel your subscription.",

    "return_request":
        "Please provide your order number for return processing.",

    "unknown_intent":
        "Sorry, I could not clearly understand your request."
}

candidate_labels = list(responses.keys())

# ============================================================
# NLP FUNCTIONS
# ============================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)

    cleaned_words = []

    for token in doc:

        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.text) > 2
        ):

            cleaned_words.append(token.lemma_)

    return " ".join(cleaned_words)

# ============================================================
# RULE-BASED INTENT
# ============================================================

def rule_based_intent(user_input):

    text = user_input.lower().strip()

    greetings = [
        "hi", "hello", "hey", "salam", "assalamualaikum"
    ]

    thanks = [
        "ok", "okay", "thanks", "thank you"
    ]

    goodbye = [
        "bye", "goodbye"
    ]

    weather_words = [
        "weather",
        "wether",
        "temperature",
        "temprature",
        "temp",
        "hot",
        "cold",
        "rain",
        "forecast",
        "humidity",
        "degree",
        "degrees"
    ]

    food_words = [
        "pizza",
        "burger",
        "biryani",
        "food",
        "meal",
        "drink"
    ]

    if text in greetings:
        return "greeting", 1.0

    if text in thanks:
        return "thanks", 1.0

    if text in goodbye:
        return "goodbye", 1.0

    if any(word in text for word in weather_words):
        return "weather_query", 1.0

    if any(word in text for word in food_words):
        return "food_order", 1.0

    if "password" in text:
        return "password_reset", 1.0

    if any(word in text for word in ["payment", "billing", "card", "pay"]):
        return "payment_issue", 1.0

    if any(word in text for word in ["package", "parcel", "tracking", "order status"]):
        return "order_status", 1.0

    if any(word in text for word in ["technical", "support", "error", "problem", "issue"]):
        return "technical_support", 1.0

    if any(word in text for word in ["cancel", "unsubscribe"]):
        return "cancellation", 1.0

    if any(word in text for word in ["return", "refund"]):
        return "return_request", 1.0

    if any(word in text for word in ["service", "services"]):
        return "service_info", 1.0

    if any(word in text for word in ["hours", "open", "working hours"]):
        return "business_hours", 1.0

    return None, 0.0

# ============================================================
# SVM INTENT
# ============================================================

def predict_intent_svm(user_input, threshold=0.50):

    clean_input = preprocess_text(user_input)

    input_vector = tfidf_vectorizer.transform([clean_input])

    predicted_label = intent_model.predict(input_vector)[0]

    intent = label_encoder.inverse_transform(
        [predicted_label]
    )[0]

    probabilities = intent_model.predict_proba(
        input_vector
    )[0]

    confidence = float(np.max(probabilities))

    if confidence < threshold:
        return None, confidence

    return intent, confidence

# ============================================================
# TRANSFORMER FALLBACK
# ============================================================

def predict_intent_transformer(user_input, threshold=0.45):

    result = zero_shot_classifier(
        user_input,
        candidate_labels=candidate_labels
    )

    intent = result["labels"][0]

    confidence = float(result["scores"][0])

    if confidence < threshold:
        return "unknown_intent", confidence

    return intent, confidence

# ============================================================
# HYBRID DETECTION
# ============================================================

def detect_intent(user_input):

    rule_intent, rule_confidence = rule_based_intent(
        user_input
    )

    if rule_intent is not None:

        return (
            rule_intent,
            rule_confidence * 100,
            "Rule Based"
        )

    transformer_intent, transformer_confidence = (
        predict_intent_transformer(user_input)
    )

    if transformer_intent != "unknown_intent":

        return (
            transformer_intent,
            transformer_confidence * 100,
            "Transformer"
        )

    svm_intent, svm_confidence = predict_intent_svm(
        user_input
    )

    if svm_intent is not None:

        return (
            svm_intent,
            svm_confidence * 100,
            "SVM"
        )

    return "unknown_intent", 0.0, "Unknown"

# ============================================================
# CITY DETECTION
# ============================================================

def is_possible_city(user_input):

    text = user_input.lower().strip()

    blocked_words = [
        "yes",
        "no",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "hi",
        "hello",
        "bye"
    ]

    if text in blocked_words:
        return False

    if re.search(r"\d", text):
        return False

    return len(text.split()) <= 3 and len(text) >= 2

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

# ============================================================
# CHATBOT ENGINE
# ============================================================

def chatbot_response(user_input):

    if (
        st.session_state.last_intent == "weather_query"
        and is_possible_city(user_input)
    ):

        response = (
            f"Thank you. I will check the weather "
            f"information for {user_input.title()}."
        )

        intent = "weather_query_followup"

        confidence = 100.0

        model_used = "Context Memory"

        st.session_state.last_intent = None

    else:

        intent, confidence, model_used = detect_intent(
            user_input
        )

        response = responses.get(
            intent,
            responses["unknown_intent"]
        )

        if intent in [
            "weather_query",
            "order_status",
            "return_request",
            "cancellation",
            "food_order"
        ]:

            st.session_state.last_intent = intent

        else:

            st.session_state.last_intent = None

    return (
        response,
        intent,
        round(confidence, 2),
        model_used
    )

# ============================================================
# ADD MESSAGE
# ============================================================

def add_user_message(user_text):

    response, intent, confidence, model_used = (
        chatbot_response(user_text)
    )

    st.session_state.messages.append({
        "role": "user",
        "content": user_text
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "intent": intent,
        "confidence": confidence,
        "model": model_used
    })

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Hybrid AI")

    st.caption("Context-aware assistant")

    st.markdown("---")

    if st.button(
        "New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_intent = None

        st.rerun()

    st.markdown("---")

    st.subheader("History")

    if not st.session_state.messages:

        st.info("No chat history yet.")

    else:

        user_messages = [

            msg for msg in st.session_state.messages

            if msg["role"] == "user"
        ]

        for i, msg in enumerate(
            reversed(user_messages[-5:]),
            1
        ):

            st.write(f"{i}. {msg['content']}")

        history_df = pd.DataFrame(
            st.session_state.messages
        )

        st.download_button(
            "Download History",
            data=history_df.to_csv(index=False),
            file_name="chat_history.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================
# MAIN UI
# ============================================================

st.title("Hybrid AI Chatbot")

if not st.session_state.messages:

    st.info(
        "Ask me anything about weather, order tracking, "
        "payments, password reset, food ordering, "
        "or technical support."
    )

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        if msg["role"] == "assistant":

            st.caption(
                f"Intent: {msg['intent']} | "
                f"Confidence: {msg['confidence']}% | "
                f"Model: {msg['model']}"
            )

# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Message Hybrid AI Chatbot..."
)

if user_input:

    add_user_message(user_input)

    st.rerun()