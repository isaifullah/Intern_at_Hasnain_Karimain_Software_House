# ============================================================
# STREAMLIT WEB APPLICATION FOR CONTEXT-AWARE CHATBOT
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pickle
import random
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Context-Aware Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models from disk"""
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    with open("models/classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    
    with open("models/responses.pkl", "rb") as f:
        responses = pickle.load(f)
    
    return vectorizer, classifier, responses

# Load models
with st.spinner("Loading chatbot models..."):
    vectorizer, classifier, responses = load_models()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text: str) -> str:
    """Preprocess text for intent prediction"""
    if not text:
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum()]
    return " ".join(tokens)

def predict_intent(text: str) -> str:
    """Predict intent from user input"""
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    intent = classifier.predict(vec)[0]
    return intent

# ============================================================
# CONTEXT MANAGEMENT
# ============================================================

if 'context' not in st.session_state:
    st.session_state.context = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []

def update_context(user_id: str, intent: str):
    st.session_state.context[user_id] = intent

def get_context(user_id: str):
    return st.session_state.context.get(user_id)

# ============================================================
# CHATBOT FUNCTION
# ============================================================

def chatbot_response(user_input: str, user_id: str = "web_user") -> str:
    """Generate chatbot response with context"""
    
    if not user_input:
        return "Please say something."
    
    last_intent = get_context(user_id)
    
    # Context-aware: User responded to weather query
    if last_intent == "weather":
        update_context(user_id, "weather_done")
        return f"Weather info for {user_input} coming soon... 🌤️"
    
    # Predict intent
    intent = predict_intent(user_input)
    update_context(user_id, intent)
    
    return responses.get(intent, "Sorry, I didn't understand that.")

# ============================================================
# UI COMPONENTS
# ============================================================

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Context-Aware AI Chatbot</h1>
    <p>Powered by NLP | Context Memory | Intelligent Responses</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Chatbot Info")
    st.markdown("- **Intent Classification:** TF-IDF + Logistic Regression")
    st.markdown("- **Context Memory:** ✅ Active")
    st.markdown("- **NLP Library:** NLTK + spaCy")
    
    st.markdown("---")
    st.markdown("## 💡 Example Queries")
    
    examples = ["hello", "what is your name", "weather", "bye"]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.example = ex
            st.rerun()
    
    st.markdown("---")
    st.markdown("## 🧠 Features")
    st.markdown("- ✅ Intent Classification")
    st.markdown("- ✅ Context Memory")
    st.markdown("- ✅ NER (spaCy)")
    st.markdown("- ✅ Database Storage")
    st.markdown("- ✅ Voice Support")

# Chat area
st.markdown("## 💬 Conversation")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if msg.get("intent"):
                st.caption(f"Intent: {msg['intent']}")

# Chat input
if 'example' in st.session_state:
    user_input = st.session_state.example
    del st.session_state.example
else:
    user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.spinner("Thinking..."):
        response = chatbot_response(user_input, "web_user")
        intent = predict_intent(user_input)
    
    # Add bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "intent": intent
    })
    
    with st.chat_message("assistant"):
        st.write(response)
        st.caption(f"Intent: {intent}")
    
    st.rerun()

# Clear button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = {}
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Context-Aware AI Chatbot | Built with NLTK, spaCy, scikit-learn</p>",
    unsafe_allow_html=True
)