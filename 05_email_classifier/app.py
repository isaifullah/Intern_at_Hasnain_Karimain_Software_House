"""
Email Classification System - Streamlit Web Application
Professional UI with Modern Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Dark Modern Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    .main-header p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        color: white;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102,126,234,0.3);
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .spam { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .promotion { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); color: white; }
    .social { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .important { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    .uncertain { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }
    
    .prediction-box h2 {
        font-size: 3rem;
        margin: 0;
        font-weight: bold;
    }
    .prediction-box p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Threshold row - professional compact */
    .threshold-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255,255,255,0.05);
        border-radius: 40px;
        padding: 0.3rem 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .threshold-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
    }
    .threshold-value {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.2rem 0.8rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
    .threshold-slider-container {
        flex: 1;
        margin: 0 1rem;
    }
    .threshold-range-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.6rem;
        color: rgba(255,255,255,0.4);
        margin-top: 0.2rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: rgba(255,255,255,0.6);
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

@st.cache_resource
def load_models():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# ============================================
# TEXT PREPROCESSING
# ============================================
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

download_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'
    elif tag.startswith('V'): return 'v'
    elif tag.startswith('N'): return 'n'
    elif tag.startswith('R'): return 'r'
    return 'n'

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    if not tokens:
        return ""
    
    try:
        pos_tags = pos_tag(tokens)
        cleaned = []
        for word, tag in pos_tags:
            if word not in stop_words and len(word) > 2:
                lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                cleaned.append(lemma)
    except:
        cleaned = [lemmatizer.lemmatize(word) for word in tokens 
                  if word not in stop_words and len(word) > 2]
    
    final_words = []
    for word in cleaned:
        if not final_words or final_words[-1] != word:
            final_words.append(word)
    
    return ' '.join(final_words)

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_email(text, model, vectorizer, label_encoder, threshold=0.65):
    if not text or len(text.strip()) < 10:
        return {
            'category': 'ERROR',
            'confidence': 0.0,
            'display': '❌ ERROR',
            'message': 'Please enter at least 10 characters'
        }
    
    cleaned = preprocess_text(text)
    
    if len(cleaned.split()) < 2:
        return {
            'category': 'UNCERTAIN',
            'confidence': 0.0,
            'display': '⚠️ UNCERTAIN',
            'message': 'Text quality too low'
        }
    
    X_input = vectorizer.transform([cleaned])
    probs = model.predict_proba(X_input)[0]
    confidence = max(probs)
    pred = np.argmax(probs)
    
    if confidence < threshold:
        category = "UNCERTAIN"
        message = f"Low confidence ({confidence:.1%})"
    else:
        category = label_encoder.inverse_transform([pred])[0]
        message = None
    
    emoji_map = {'Spam': '🚫', 'Promotion': '📢', 'Social': '👥', 'Important': '⭐', 'UNCERTAIN': '⚠️', 'ERROR': '❌'}
    color_map = {'Spam': 'spam', 'Promotion': 'promotion', 'Social': 'social', 'Important': 'important', 'UNCERTAIN': 'uncertain', 'ERROR': 'uncertain'}
    
    return {
        'category': category,
        'display': f"{emoji_map.get(category, '')} {category}",
        'box_class': color_map.get(category, 'uncertain'),
        'confidence': confidence,
        'message': message,
        'probabilities': {label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)}
    }

# ============================================
# MAIN APP
# ============================================
st.markdown("""
<div class="main-header">
    <h1>📧 Email Classification System</h1>
    <p>AI-Powered Email Categorization | Spam • Promotion • Social • Important</p>
</div>
""", unsafe_allow_html=True)

# Load models
model, vectorizer, label_encoder = load_models()

if model is None:
    st.error("❌ Failed to load models. Please run training first.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Processing", "📊 Analytics"])

# ============================================
# TAB 1: SINGLE EMAIL
# ============================================
with tab1:
    # Professional threshold row
    col_thresh1, col_thresh2, col_thresh3 = st.columns([1, 3, 1])
    with col_thresh2:
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.65,
            step=0.05,
            key="threshold_single",
            help="Higher threshold = fewer but more accurate predictions"
        )
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: -0.8rem;">
            <span style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">Less Strict (50%)</span>
            <span style="font-size: 0.7rem; background: linear-gradient(135deg, #667eea, #764ba2); padding: 0.1rem 0.5rem; border-radius: 20px;">Current: {int(threshold*100)}%</span>
            <span style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">More Strict (95%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ✍️ Enter Email Content")
    
    email_text = st.text_area(
        "",
        height=180,
        placeholder="Paste your email content here...\n\nExample: Congratulations! You've won $1,000,000! Click here to claim your prize now!!!",
        key="email_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        analyze_btn = st.button("🔍 Classify", use_container_width=True, type="primary")
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state.email_input = ""
        st.rerun()
    
    if analyze_btn and email_text:
        with st.spinner("Analyzing email content..."):
            result = predict_email(email_text, model, vectorizer, label_encoder, threshold)
            
            st.markdown(f"""
            <div class="prediction-box {result['box_class']}">
                <h2>{result['display']}</h2>
                <p>{'✨ Important work or personal email' if result['category'] == 'Important' else '📢 Marketing or promotional content' if result['category'] == 'Promotion' else '👥 Social notification' if result['category'] == 'Social' else '🚫 Unsolicited spam email' if result['category'] == 'Spam' else '⚠️ Low confidence - please review'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if result['message']:
                st.warning(result['message'])
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <p style="margin:0; font-size:0.8rem; opacity:0.7;">Confidence Score</p>
                    <p style="margin:0; font-size:2rem; font-weight:bold;">{result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                status_color = "#43e97b" if result['confidence'] >= threshold else "#fda085"
                status_text = "Above Threshold" if result['confidence'] >= threshold else "Below Threshold"
                st.markdown(f"""
                <div class="metric-card">
                    <p style="margin:0; font-size:0.8rem; opacity:0.7;">Threshold Status</p>
                    <p style="margin:0; font-size:1.3rem; font-weight:bold; color:{status_color};">{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                title={'text': "Confidence Gauge", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#667eea"},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, threshold*100], 'color': "rgba(245,87,108,0.3)"},
                        {'range': [threshold*100, 100], 'color': "rgba(67,233,123,0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 Probability Distribution")
            prob_df = pd.DataFrame({
                'Category': list(result['probabilities'].keys()),
                'Probability': list(result['probabilities'].values())
            })
            
            colors = {'Spam': '#f5576c', 'Promotion': '#fda085', 'Social': '#4facfe', 'Important': '#43e97b'}
            bar_colors = [colors.get(cat, '#667eea') for cat in prob_df['Category']]
            
            fig = px.bar(prob_df, x='Category', y='Probability', 
                         color='Category', color_discrete_sequence=bar_colors,
                         text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'))
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': 'white'},
                xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
                yaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analyze_btn and not email_text:
        st.warning("⚠️ Please enter email content to classify")

# ============================================
# TAB 2: BATCH PROCESSING
# ============================================
with tab2:
    # Professional threshold row
    col_thresh1, col_thresh2, col_thresh3 = st.columns([1, 3, 1])
    with col_thresh2:
        threshold_batch = st.slider(
            "Confidence Threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.65,
            step=0.05,
            key="threshold_batch",
            help="Higher threshold = fewer but more accurate predictions"
        )
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: -0.8rem;">
            <span style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">Less Strict (50%)</span>
            <span style="font-size: 0.7rem; background: linear-gradient(135deg, #667eea, #764ba2); padding: 0.1rem 0.5rem; border-radius: 20px;">Current: {int(threshold_batch*100)}%</span>
            <span style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">More Strict (95%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Batch Email Classification")
    st.markdown("Upload a CSV file with a **'text'** column containing emails to classify")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows")
        
        with st.expander("📋 Preview Data"):
            st.dataframe(df.head())
        
        if 'text' not in df.columns:
            st.error("❌ CSV must have a 'text' column")
            available = df.columns.tolist()
            st.info(f"Available columns: {', '.join(available)}")
        else:
            if st.button("🚀 Start Batch Processing", use_container_width=True):
                with st.spinner(f"Processing {len(df)} emails..."):
                    results = []
                    progress_bar = st.progress(0)
                    for idx, row in df.iterrows():
                        result = predict_email(str(row['text']), model, vectorizer, label_encoder, threshold_batch)
                        results.append(result)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    df['prediction'] = [r['display'] for r in results]
                    df['confidence'] = [r['confidence'] for r in results]
                    
                    st.success("✅ Processing complete!")
                    
                    st.markdown("#### 📊 Results Preview")
                    st.dataframe(df[['text', 'prediction', 'confidence']].head(20))
                    
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    href = f'<a href="data:file/csv;base64,{b64}" download="classified_emails_{timestamp}.csv" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1rem; border-radius: 10px; text-decoration: none; font-weight: bold;">📥 Download Results CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.markdown("#### 📈 Summary Statistics")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total", len(df))
                    with col2:
                        spam = (df['prediction'].str.contains('Spam', na=False)).sum()
                        st.metric("🚫 Spam", spam)
                    with col3:
                        promo = (df['prediction'].str.contains('Promotion', na=False)).sum()
                        st.metric("📢 Promotion", promo)
                    with col4:
                        social = (df['prediction'].str.contains('Social', na=False)).sum()
                        st.metric("👥 Social", social)
                    with col5:
                        imp = (df['prediction'].str.contains('Important', na=False)).sum()
                        st.metric("⭐ Important", imp)
                    
                    fig = px.pie(values=[spam, promo, social, imp],
                                 names=['Spam', 'Promotion', 'Social', 'Important'],
                                 title='Prediction Distribution',
                                 color_discrete_sequence=['#f5576c', '#fda085', '#4facfe', '#43e97b'])
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': 'white'},
                        title_font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: ANALYTICS
# ============================================
with tab3:
    st.markdown("### 📊 Model Performance Analytics")
    
    st.markdown("#### 🔑 Top Keywords per Category")
    
    if hasattr(model, 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        
        keyword_cols = st.columns(2)
        for i, category in enumerate(label_encoder.classes_):
            with keyword_cols[i % 2]:
                coef = model.coef_[i]
                top_idx = np.argsort(coef)[-10:][::-1]
                top_words = [feature_names[idx] for idx in top_idx]
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0;">📌 {category}</h4>
                    <p style="margin: 0; font-size: 0.9rem;">{', '.join(top_words)}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Feature importance not available for this model type")
    
    st.markdown("#### 📊 Confusion Matrix")
    if os.path.exists('figures/confusion_matrix.png'):
        st.image('figures/confusion_matrix.png', use_container_width=True)
    else:
        st.info("Run training to generate confusion matrix")
    
    st.markdown("#### 📈 ROC Curves (One-vs-Rest)")
    if os.path.exists('figures/roc_curves.png'):
        st.image('figures/roc_curves.png', use_container_width=True)
    else:
        st.info("Run training to generate ROC curves")
    
    st.markdown("#### 🎯 Per-Class Performance")
    if os.path.exists('figures/per_class_performance.png'):
        st.image('figures/per_class_performance.png', use_container_width=True)
    else:
        st.info("Run training to generate per-class performance chart")
    
    st.markdown("---")
    st.markdown("#### 💡 Tips for Better Results")
    
    col_tip1, col_tip2 = st.columns(2)
    with col_tip1:
        st.markdown("""
        <div style="background: rgba(67,233,123,0.1); border-left: 4px solid #43e97b; padding: 1rem; border-radius: 10px;">
            <strong>✅ DO's</strong><br>
            • Use complete emails (50+ words)<br>
            • Include clear context and details<br>
            • Use proper grammar and spelling
        </div>
        """, unsafe_allow_html=True)
    with col_tip2:
        st.markdown("""
        <div style="background: rgba(245,87,108,0.1); border-left: 4px solid #f5576c; padding: 1rem; border-radius: 10px;">
            <strong>❌ DON'Ts</strong><br>
            • Very short emails (&lt;10 chars) → UNCERTAIN<br>
            • All caps or excessive punctuation<br>
            • Check confidence threshold settings
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p>Email Classification System | Powered by Logistic Regression | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)