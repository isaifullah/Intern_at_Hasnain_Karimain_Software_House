# ============================================================
# PRETRAINED TRANSFORMER CONTEXT-AWARE AI CHATBOT
# Professional Customer Support Assistant with Voice Chat
# File: app.py
# ============================================================

import os
import re
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

# Voice libraries
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nexus Support AI | Customer Service Chatbot",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 10px 0 0;
        font-size: 1.1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e32 0%, #16162a 100%);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #2d2d44;
        text-align: center;
        margin-bottom: 15px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        word-break: break-word;
    }

    .metric-label {
        color: #888;
        font-size: 12px;
        margin-top: 5px;
    }

    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }

    hr {
        border-color: #2d2d44;
        margin: 20px 0;
    }

    .info-box {
        background: linear-gradient(135deg, #1e1e32 0%, #16162a 100%);
        border-radius: 15px;
        padding: 15px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        color: #e0e0e0;
    }

    .badge {
        background: rgba(102,126,234,0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #a0a0ff;
        display: inline-block;
        margin-right: 8px;
        margin-top: 5px;
    }
    
    /* Input row with mic button */
    .input-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .mic-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
        font-size: 20px;
    }
    
    .mic-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_PATH = "chat_history.db"


def init_database():
    """Initialize SQLite database for conversation storage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            predicted_intent TEXT,
            confidence REAL,
            model_used TEXT,
            input_mode TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(chat_history)")
    columns = [column[1] for column in cursor.fetchall()]

    if "model_used" not in columns:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN model_used TEXT")

    if "input_mode" not in columns:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN input_mode TEXT")

    conn.commit()
    conn.close()


def save_chat(user_id, user_message, bot_response, intent, confidence, model_used, input_mode="text"):
    """Save conversation to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history
        (user_id, user_message, bot_response, predicted_intent, confidence, model_used, input_mode, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, user_message, bot_response, intent, float(confidence), model_used, input_mode,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def load_chat_history(limit=10):
    """Load recent chat history."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT user_message, bot_response, predicted_intent, confidence, model_used, input_mode, timestamp
        FROM chat_history ORDER BY id DESC LIMIT ?
    """
    history = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return history


init_database()


# ============================================================
# INTENTS AND RESPONSES
# ============================================================

candidate_labels = [
    "account_help", "business_hours", "cancellation", "order_status",
    "password_reset", "payment_update", "return_request", "service_info", "technical_support"
]

responses = {
    "greeting": "Hello! How can I help you today?",
    "small_talk": "You're welcome. Let me know if you need anything else.",
    "account_help": "Sure, I can help with your account. Please tell me what issue you are facing.",
    "business_hours": "Our support team is available from 9 AM to 6 PM, Monday to Saturday.",
    "cancellation": "I can help you cancel your subscription. Please confirm if you want to continue.",
    "order_status": "I can help you track your order. Please provide your order ID or tracking number.",
    "password_reset": "You can reset your password using the Forgot Password option.",
    "payment_update": "You can update your payment method from your account billing settings.",
    "return_request": "I can help you start a return request. Please share your order number.",
    "service_info": "We provide account support, order tracking, returns, payment help, cancellation support.",
    "technical_support": "I can help with technical support. Please describe the issue."
}


# ============================================================
# LOAD PRETRAINED TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_transformer_model():
    with st.spinner("🚀 Loading AI Model..."):
        model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return model


zero_shot_classifier = load_transformer_model()


# ============================================================
# VOICE FUNCTIONS
# ============================================================

@st.cache_resource
def load_tts_engine():
    if not VOICE_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        return engine
    except Exception:
        return None


tts_engine = load_tts_engine()


def listen_to_user():
    """Capture voice input and convert to text."""
    if not VOICE_AVAILABLE:
        return None, "Voice libraries not installed."

    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        text = recognizer.recognize_google(audio)
        return text, None
    except sr.WaitTimeoutError:
        return None, "No voice detected. Please try again."
    except sr.UnknownValueError:
        return None, "Could not understand audio."
    except sr.RequestError:
        return None, "Speech recognition service unavailable."
    except Exception as e:
        return None, str(e)


def speak_response(text):
    """Convert text to speech."""
    if tts_engine is None:
        return False
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
        return True
    except Exception:
        return False


# ============================================================
# RULE-BASED INTENT DETECTION
# ============================================================

def rule_based_intent(user_input):
    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
    thanks = ["ok", "okay", "thanks", "thank you", "alright", "fine"]

    if text in greetings:
        return "greeting", 1.0
    if text in thanks:
        return "small_talk", 1.0
    if any(w in text for w in ["cancel", "cancellation", "unsubscribe"]):
        return "cancellation", 1.0
    if any(w in text for w in ["forgot password", "forget password", "reset password", "password"]):
        return "password_reset", 1.0
    if any(w in text for w in ["order", "package", "tracking", "shipment"]):
        return "order_status", 1.0
    if any(w in text for w in ["return", "refund", "replace"]):
        return "return_request", 1.0
    if any(w in text for w in ["payment", "billing", "card"]):
        return "payment_update", 1.0
    if any(w in text for w in ["technical", "error", "issue", "problem", "not working"]):
        return "technical_support", 1.0
    if any(w in text for w in ["service", "offer", "provide"]):
        return "service_info", 1.0
    if any(w in text for w in ["open", "hours", "working hours", "time"]):
        return "business_hours", 1.0

    return None, 0.0


def predict_intent_transformer(user_input):
    result = zero_shot_classifier(user_input, candidate_labels=candidate_labels)
    return result["labels"][0], float(result["scores"][0])


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False


# ============================================================
# CHATBOT ENGINE
# ============================================================

def chatbot_response(user_input):
    if not user_input.strip():
        return {"response": "Please type your message.", "intent": "empty", "confidence": 0.0, "model_used": "Rule Based"}

    last_intent = st.session_state.last_intent

    # Context follow-ups
    if last_intent == "order_status" and re.search(r"\d+", user_input):
        st.session_state.last_intent = None
        return {"response": f"Thank you. I will check order status for ID: {user_input}", "intent": "order_status_followup", "confidence": 1.0, "model_used": "Context Memory"}

    if last_intent == "return_request" and re.search(r"\d+", user_input):
        st.session_state.last_intent = None
        return {"response": f"Thank you. I will start return process for order ID: {user_input}", "intent": "return_request_followup", "confidence": 1.0, "model_used": "Context Memory"}

    if last_intent == "cancellation" and user_input.lower().strip() in ["yes", "confirm", "sure"]:
        st.session_state.last_intent = None
        return {"response": "Your cancellation request has been confirmed.", "intent": "cancellation_confirmed", "confidence": 1.0, "model_used": "Context Memory"}

    # Intent prediction
    rule_intent, rule_confidence = rule_based_intent(user_input)
    if rule_intent is not None:
        intent, confidence, model_used = rule_intent, rule_confidence, "Rule Based"
    else:
        intent, confidence = predict_intent_transformer(user_input)
        model_used = "Pretrained Transformer"

    response = responses.get(intent, "I understand, but need more detail to help you.")

    if intent in ["order_status", "return_request", "cancellation"]:
        st.session_state.last_intent = intent
    else:
        st.session_state.last_intent = None

    return {"response": response, "intent": intent, "confidence": round(confidence, 3), "model_used": model_used}


def process_message(user_input, input_mode="text"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    result = chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": result["response"], "intent": result["intent"], "confidence": result["confidence"], "model_used": result["model_used"]})
    
    save_chat("streamlit_user", user_input, result["response"], result["intent"], result["confidence"], result["model_used"], input_mode)
    
    if st.session_state.voice_enabled and input_mode == "voice":
        speak_response(result["response"])
    
    return result


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🎧 Nexus Support AI</h1>
    <p>Intelligent Customer Support Assistant | Text + Voice | Powered by Transformer AI</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

left_col, right_col = st.columns([2.5, 1])


# ============================================================
# LEFT COLUMN - CHAT INTERFACE
# ============================================================

with left_col:
    st.markdown("### 💬 Conversation")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 15px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 18px; border-radius: 20px 20px 5px 20px; max-width: 70%;">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 15px;">
                <div style="background: #1e1e32; color: #e0e0e0; padding: 12px 18px; border-radius: 20px 20px 20px 5px; max-width: 70%; border: 1px solid #2d2d44;">
                    {message['content']}
                    <div style="margin-top: 8px;">
                        <span class="badge">🎯 {message.get('intent', 'unknown')}</span>
                        <span class="badge">📊 {message.get('confidence', 0)}</span>
                        <span class="badge">🤖 {message.get('model_used', 'unknown')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # CHAT INPUT WITH MIC BUTTON INSIDE (LIKE CHATGPT)
    st.markdown("---")
    st.markdown("### ✍️ Your Message")
    
    # Create two columns: 5 parts for input, 1 part for mic button
    col_input, col_mic = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input("", placeholder="Type your message here...", key="user_text_input", label_visibility="collapsed")
    
    with col_mic:
        voice_clicked = st.button("🎤", key="mic_button", help="Click to speak", use_container_width=True)
    
    # Send button
    send_clicked = st.button("📤 Send", key="send_button", use_container_width=False)
    
    # Process text input
    if send_clicked and user_input:
        with st.spinner("🤔 Thinking..."):
            process_message(user_input, input_mode="text")
        st.rerun()
    
    # Process voice input
    if voice_clicked:
        with st.spinner("🎤 Listening... Please speak..."):
            voice_text, voice_error = listen_to_user()
        
        if voice_error:
            st.error(voice_error)
        elif voice_text:
            st.success(f"🎤 You said: {voice_text}")
            with st.spinner("🤔 Thinking..."):
                process_message(voice_text, input_mode="voice")
            st.rerun()


# ============================================================
# RIGHT COLUMN - DASHBOARD
# ============================================================

with right_col:
    st.markdown("### 🟢 System Status")
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">✅ Active</div>
        <div class="metric-label">Pretrained Transformer</div>
        <div style="margin-top: 10px; font-size: 11px; color: #666;">facebook/bart-large-mnli</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎙️ Voice Settings")
    st.session_state.voice_enabled = st.toggle("🔊 Auto Voice Response", value=st.session_state.voice_enabled)

    if VOICE_AVAILABLE and tts_engine is not None:
        st.success("✅ Voice system ready")
        if st.button("🔊 Test Voice", use_container_width=True):
            speak_response("Hello! This is a test of the voice system.")
            st.success("Voice test sent!")
    else:
        st.warning("⚠️ Voice system not available")
        st.caption("Install: pip install speechrecognition pyttsx3 pyaudio")

    st.markdown("### 🧠 Active Context")
    if st.session_state.last_intent:
        st.markdown(f"""
        <div class="info-box">
            <strong>Waiting for:</strong><br>
            <code style="color: #a0a0ff;">{st.session_state.last_intent}</code>
            <br><small>Bot expecting follow-up info</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <em>No active context</em>
            <br><small>Ready for new conversation</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Last Prediction")
    assistant_msgs = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
    if assistant_msgs:
        last = assistant_msgs[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{last.get('intent', '—')}</div>
            <div class="metric-label">Intent</div>
            <div><span class="badge">Confidence: {last.get('confidence', 0)}</span><span class="badge">Model: {last.get('model_used', '—')}</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">—</div>
            <div class="metric-label">No prediction yet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎮 Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_intent = None
            st.rerun()
    with col2:
        if st.button("🧹 Clear Context", use_container_width=True):
            st.session_state.last_intent = None
            st.rerun()

    st.markdown("### ⚡ Quick Actions")
    quick_queries = ["🔑 Forgot password", "❌ Cancel subscription", "📦 Order status", "💰 Update payment"]
    for q in quick_queries:
        if st.button(q, key=q, use_container_width=True):
            process_message(q.replace("🔑 ", "").replace("❌ ", "").replace("📦 ", "").replace("💰 ", ""), input_mode="quick")
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 Powered by Hugging Face Transformers | 🎙️ Voice Chat | 🎨 Built with Streamlit | 💾 SQLite Database</p>
    <p style="font-size: 12px;">Nexus Support AI | Click 🎤 button to speak | Type to chat</p>
</div>
""", unsafe_allow_html=True)