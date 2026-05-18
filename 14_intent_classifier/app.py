# ============================================================
# INTENT CLASSIFICATION SYSTEM - COMPLETE DARK THEME
# ZERO WHITE BACKGROUNDS ANYWHERE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time
import os
import re
import joblib
import spacy

# Set page config
st.set_page_config(
    page_title="Intent Classification System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - COMPLETE DARK THEME - NO WHITE ANYWHERE
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 50px 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 30px;
        margin-bottom: 40px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: white !important;
        display: inline-block;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s ease-in;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .intent-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        opacity: 0.9;
        margin-bottom: 15px;
        font-weight: 500;
    }
    
    .intent-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 25px;
        word-break: break-word;
    }
    
    .confidence-label {
        font-size: 0.9rem;
        margin-bottom: 10px;
        opacity: 0.9;
    }
    
    .confidence-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    /* Confidence bar */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        height: 12px;
        margin: 20px 0;
        overflow: hidden;
    }
    
    .confidence-bar {
        background: #4ade80;
        height: 100%;
        border-radius: 15px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling - DARK THEME */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #4a4a6a;
        font-size: 1rem;
        padding: 15px;
        transition: all 0.3s ease;
        font-family: inherit;
        background: #1a1a2e !important;
        color: #ffffff !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #8888aa !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        outline: none;
        background: #1e1e35 !important;
    }
    
    /* History items - DARK THEME */
    .history-item {
        background: #1a1a2e !important;
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        border: 1px solid #2a2a4a;
    }
    
    .history-item:hover {
        background: #1e1e35 !important;
        transform: translateX(5px);
        border-color: #764ba2;
    }
    
    .history-text {
        font-size: 0.95rem;
        color: #ffffff !important;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .history-meta {
        font-size: 0.75rem;
        color: #aaaacc !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    
    .history-meta span {
        color: #aaaacc !important;
    }
    
    .history-meta strong {
        color: #ffffff !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 24px;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 30px 0;
    }
    
    /* Info box */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        color: white;
        font-weight: 500;
    }
    
    /* All text white */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: white !important;
    }
    
    .stText, .stMarkdown, label {
        color: white !important;
    }
    
    .stButton button p {
        color: white !important;
    }
    
    /* Remove white background from all containers */
    .css-1r6slb0, .css-1v3fvcr, .stPlotlyChart, .element-container, .stMarkdown {
        background: transparent !important;
    }
    
    /* Block containers */
    .block-container {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Load spaCy
@st.cache_resource
def load_spacy():
    try:
        return spacy.load('en_core_web_sm')
    except:
        os.system('python -m spacy download en_core_web_sm')
        return spacy.load('en_core_web_sm')

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    return model, vectorizer, label_encoder

# Preprocessing function
def preprocess_text(text, nlp):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    doc = nlp(text)
    cleaned_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 2:
            cleaned_words.append(token.lemma_)
    
    return " ".join(cleaned_words)

# Prediction function
def predict_intent(text, model, vectorizer, label_encoder, nlp):
    if not text or text.strip() == "":
        return "empty_input", 0.0
    
    clean_input = preprocess_text(text, nlp)
    input_vector = vectorizer.transform([clean_input])
    predicted_encoded = model.predict(input_vector)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_encoded])[0]
    
    probabilities = model.predict_proba(input_vector)[0]
    confidence = np.max(probabilities)
    
    if confidence < 0.50:
        predicted_intent = "unknown_intent"
    
    return predicted_intent, confidence * 100

# Main function
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="main-title">🎯 Intent Classification System</div>
        <div class="subtitle">Advanced Machine Learning for Intelligent Intent Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check models
    if not os.path.exists('models/best_model.pkl'):
        st.error("❌ Models not found! Please train the model first.")
        return
    
    # Load everything
    with st.spinner("🚀 Loading AI models..."):
        nlp = load_spacy()
        model, vectorizer, label_encoder = load_models()
    
    # Initialize history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Intent Classifier", "📈 Performance Analytics", "📜 Prediction History"])
    
    # Tab 1: Classifier
    with tab1:
        col_left, col_right = st.columns([1.2, 0.8], gap="large")
        
        with col_left:
            st.markdown('<p class="section-header">✍️ Enter Your Text</p>', unsafe_allow_html=True)
            
            user_input = st.text_area(
                "",
                height=120,
                placeholder="Example: 'I want to order a large pizza' or 'What's the weather like today?'",
                key="live_input",
                label_visibility="collapsed"
            )
            
            classify_button = st.button("🔍 Classify Intent", use_container_width=True, type="primary")
            
            st.markdown("---")
            st.markdown("#### ⚡ Quick Examples")
            
            examples = [
                ("👋 Greeting", "Hello, how are you?"),
                ("🍕 Food Order", "I want to order pizza"),
                ("🌤️ Weather", "What's the weather today?"),
                ("📦 Track Order", "Where is my order?"),
                ("❌ Cancel", "Cancel my subscription"),
                ("🙏 Gratitude", "Thank you so much")
            ]
            
            cols = st.columns(3)
            for idx, (label, text) in enumerate(examples):
                with cols[idx % 3]:
                    if st.button(label, key=f"ex_{idx}", use_container_width=True):
                        st.session_state.live_input = text
                        st.rerun()
        
        with col_right:
            st.markdown('<p class="section-header">🎯 Classification Result</p>', unsafe_allow_html=True)
            
            if classify_button and user_input:
                with st.spinner("Analyzing..."):
                    time.sleep(0.2)
                    intent, confidence = predict_intent(user_input, model, vectorizer, label_encoder, nlp)
                
                display_intent = intent.replace('_', ' ').title()
                
                if confidence >= 80:
                    bar_color = "#4ade80"
                    conf_level = "Very High"
                    emoji = "🎯"
                elif confidence >= 60:
                    bar_color = "#facc15"
                    conf_level = "High"
                    emoji = "👍"
                elif confidence >= 40:
                    bar_color = "#fb923c"
                    conf_level = "Moderate"
                    emoji = "🤔"
                else:
                    bar_color = "#ef4444"
                    conf_level = "Low"
                    emoji = "⚠️"
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.prediction_history.insert(0, {
                    'text': user_input[:60] + "..." if len(user_input) > 60 else user_input,
                    'intent': display_intent,
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'confidence_level': conf_level,
                    'emoji': emoji
                })
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="intent-label">PREDICTED INTENT</div>
                    <div class="intent-value">{emoji} {display_intent}</div>
                    <div class="confidence-label">CONFIDENCE SCORE</div>
                    <div class="confidence-value">{confidence:.1f}%</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: {confidence}%; background: {bar_color};"></div>
                    </div>
                    <div style="margin-top: 15px; font-size: 0.85rem;">Confidence Level: {conf_level}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card" style="opacity: 0.7;">
                    <div class="intent-label">PREDICTED INTENT</div>
                    <div class="intent-value">—</div>
                    <div class="confidence-label">CONFIDENCE SCORE</div>
                    <div class="confidence-value">0%</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: 0%;"></div>
                    </div>
                    <div style="margin-top: 15px;">Enter text and click classify</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 2: Analytics - COMPLETELY DARK, NO WHITE
    with tab2:
        st.markdown('<p class="section-header">📊 Model Performance</p>', unsafe_allow_html=True)
        
        metrics_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [0.985, 0.98, 0.98, 0.98]
        })
        
        # Create chart with DARK background
        fig = px.bar(
            metrics_data, 
            x='Metric', 
            y='Score',
            text=[f'{x:.1%}' for x in metrics_data['Score']],
            color='Metric',
            color_discrete_sequence=['#667eea', '#764ba2', '#667eea', '#764ba2']
        )
        fig.update_traces(textposition='outside', marker_line_width=0)
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_tickformat='.0%',
            yaxis_range=[0, 1],
            # DARK THEME FOR CHART
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='white', size=12),
            title_font=dict(color='white'),
            xaxis=dict(
                title='Metric',
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='#2a2a4a'
            ),
            yaxis=dict(
                title='Score',
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='#2a2a4a'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison chart - DARK BACKGROUND
        st.markdown('<p class="section-header">🤖 Model Comparison</p>', unsafe_allow_html=True)
        
        comparison_data = pd.DataFrame({
            'Model': ['Logistic Regression', 'Naive Bayes', 'Linear SVM', 'Random Forest'],
            'Accuracy': [96.4, 92.8, 98.5, 97.1]
        })
        
        fig2 = px.bar(
            comparison_data,
            x='Model',
            y='Accuracy',
            text=[f'{x:.1f}%' for x in comparison_data['Accuracy']],
            color='Model',
            color_discrete_sequence=['#c3cfe2', '#c3cfe2', '#667eea', '#c3cfe2']
        )
        fig2.update_traces(textposition='auto', marker_line_width=0)
        fig2.update_layout(
            height=400,
            showlegend=False,
            yaxis_range=[85, 100],
            # DARK THEME FOR CHART
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font=dict(color='white', size=12),
            title_font=dict(color='white'),
            xaxis=dict(
                title='Model',
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='#2a2a4a'
            ),
            yaxis=dict(
                title='Accuracy (%)',
                title_font=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='#2a2a4a'
            )
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Technical Specifications - DARK CARDS
        st.markdown('<p class="section-header">🔧 Technical Specifications</p>', unsafe_allow_html=True)
        
        spec_col1, spec_col2, spec_col3 = st.columns(3)
        
        with spec_col1:
            st.markdown("""
            <div style="background: #1a1a2e; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #2a2a4a;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🔤</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #667eea;">TF-IDF</div>
                <div style="font-size: 0.8rem; color: #aaaacc;">Feature Extraction with 5000 features</div>
                <div style="font-size: 0.7rem; color: #8888aa; margin-top: 8px;">1-2 gram range</div>
            </div>
            """, unsafe_allow_html=True)
        
        with spec_col2:
            st.markdown("""
            <div style="background: #1a1a2e; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #2a2a4a;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📚</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #667eea;">spaCy</div>
                <div style="font-size: 0.8rem; color: #aaaacc;">NLP Preprocessing Pipeline</div>
                <div style="font-size: 0.7rem; color: #8888aa; margin-top: 8px;">Lemmatization & Stopword removal</div>
            </div>
            """, unsafe_allow_html=True)
        
        with spec_col3:
            st.markdown("""
            <div style="background: #1a1a2e; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #2a2a4a;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🎯</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #667eea;">50% Threshold</div>
                <div style="font-size: 0.8rem; color: #aaaacc;">Confidence Threshold for Unknown</div>
                <div style="font-size: 0.7rem; color: #8888aa; margin-top: 8px;">Unknown intent handling</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: History
    with tab3:
        st.markdown('<p class="section-header">📜 Recent Predictions</p>', unsafe_allow_html=True)
        
        if not st.session_state.prediction_history:
            st.markdown("""
            <div class="info-box">
                🤖 No predictions yet. Go to the Intent Classifier tab to make your first prediction!
            </div>
            """, unsafe_allow_html=True)
        else:
            col_btn1, col_btn2 = st.columns([1, 5])
            with col_btn1:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()
            
            st.markdown("---")
            
            for item in st.session_state.prediction_history:
                if item['confidence'] >= 80:
                    badge_color = "#4ade80"
                    badge_bg = "#1a3a2a"
                elif item['confidence'] >= 60:
                    badge_color = "#facc15"
                    badge_bg = "#3a3a1a"
                else:
                    badge_color = "#ef4444"
                    badge_bg = "#3a1a1a"
                
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-text">
                        <strong>📝 Text:</strong> "{item['text']}"
                    </div>
                    <div class="history-meta">
                        <span>{item['emoji']} <strong>Intent:</strong> {item['intent']}</span>
                        <span style="background: {badge_bg}; color: {badge_color}; padding: 4px 12px; border-radius: 20px; font-weight: 600;">
                            {item['confidence']:.1f}% ({item['confidence_level']})
                        </span>
                        <span>⏰ {item['timestamp']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.5); font-size: 0.8rem;">
        🚀 Powered by Linear SVM | TF-IDF | spaCy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()