"""
Email Classification System - Streamlit Web Application
Complete version with Single Prediction, Batch Processing, and Analytics tabs
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
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Email Classifier",
    page_icon="📧",
    layout="wide"
)

# Custom CSS - Improved compact design
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 0.8rem;
        margin: 0.3rem 0 0 0;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .spam { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .promotion { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); color: white; }
    .social { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .important { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    .uncertain { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }
    
    /* Sidebar compact metrics */
    .compact-metric {
        background: #2d2d44;
        padding: 0.4rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.2rem 0;
    }
    .compact-metric label {
        font-size: 0.7rem;
        color: #aaa;
        display: block;
    }
    .compact-metric value {
        font-size: 1rem;
        font-weight: bold;
        color: white;
        display: block;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
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
# SIDEBAR - COMPACT VERSION
# ============================================
with st.sidebar:
    st.markdown("### 📧 Email Classifier")
    
    model, vectorizer, label_encoder = load_models()
    
    if model is None:
        st.error("Models not loaded!")
        st.stop()
    
    st.markdown("---")
    
    # Compact model info
    st.markdown("#### 🤖 Model Info")
    
    st.markdown("""
    <div class="compact-metric">
        <label>Model</label>
        <value>Logistic Regression</value>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="compact-metric">
        <label>F1 Score</label>
        <value>0.95</value>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="compact-metric">
        <label>Categories</label>
        <value>4</value>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="compact-metric">
        <label>Features</label>
        <value>15K</value>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    st.markdown("#### ⚙️ Settings")
    threshold = st.slider(
        "Confidence Threshold",
        0.5, 0.95, 0.65, 0.05
    )
    
    st.markdown("---")
    st.caption("ML-powered classification | Logistic Regression")

# ============================================
# MAIN HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>📧 Email Classification System</h1>
    <p>Classify emails as Spam, Promotion, Social, or Important</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# THREE TABS - Analytics TAB RESTORED
# ============================================
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Processing", "📊 Analytics"])

# ============================================
# TAB 1: SINGLE EMAIL
# ============================================
with tab1:
    st.markdown("### 📝 Enter Email Content")
    
    email_text = st.text_area(
        "",
        height=200,
        placeholder="Paste your email content here...\n\nExample: Congratulations! You've won $1,000,000! Click here to claim your prize now!!!",
        key="email_input"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    with col2:
        analyze_btn = st.button("🔍 Classify Email", use_container_width=True, type="primary")
    
    if clear_btn:
        st.session_state.email_input = ""
        st.rerun()
    
    if analyze_btn and email_text:
        with st.spinner("Analyzing..."):
            result = predict_email(email_text, model, vectorizer, label_encoder, threshold)
            
            # Prediction box
            st.markdown(f"""
            <div class="prediction-box {result['box_class']}">
                <h2>{result['display']}</h2>
                <p>{'✨ Important work or personal email' if result['category'] == 'Important' else '📢 Marketing or promotional content' if result['category'] == 'Promotion' else '👥 Social notification' if result['category'] == 'Social' else '🚫 Unsolicited spam email' if result['category'] == 'Spam' else '⚠️ Low confidence - please review'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if result['message']:
                st.warning(result['message'])
            
            # Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with col_m2:
                status = "🟢 High" if result['confidence'] >= threshold else "🟡 Low"
                st.metric("Confidence Level", status)
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                title={'text': "Confidence Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, threshold*100], 'color': "#ffebee"},
                        {'range': [threshold*100, 100], 'color': "#e8f5e9"}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability bars
            st.markdown("#### 📊 Probability by Category")
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
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: BATCH PROCESSING
# ============================================
with tab2:
    st.markdown("### 📁 Batch Email Classification")
    st.write("Upload a CSV file with a **'text'** column")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"📊 Loaded {len(df)} rows")
        
        with st.expander("Preview uploaded data"):
            st.dataframe(df.head())
        
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column")
            available = df.columns.tolist()
            selected = st.selectbox("Select text column:", available)
            if st.button("Use this column"):
                df.rename(columns={selected: 'text'}, inplace=True)
                st.rerun()
        else:
            if st.button("🚀 Process Batch", use_container_width=True):
                with st.spinner(f"Processing {len(df)} emails..."):
                    results = []
                    for idx, row in df.iterrows():
                        result = predict_email(str(row['text']), model, vectorizer, label_encoder, threshold)
                        results.append(result)
                    
                    df['prediction'] = [r['display'] for r in results]
                    df['confidence'] = [r['confidence'] for r in results]
                    
                    st.success("✅ Processing complete!")
                    st.dataframe(df[['text', 'prediction', 'confidence']].head(20))
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    href = f'<a href="data:file/csv;base64,{b64}" download="classified_emails_{timestamp}.csv">📥 Download Results CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Summary statistics
                    st.markdown("### 📊 Summary")
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
                    
                    # Pie chart
                    fig = px.pie(values=[spam, promo, social, imp],
                                 names=['Spam', 'Promotion', 'Social', 'Important'],
                                 title='Prediction Distribution',
                                 color_discrete_sequence=['#f5576c', '#fda085', '#4facfe', '#43e97b'])
                    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: ANALYTICS (RESTORED)
# ============================================
with tab3:
    st.markdown("### 📊 Model Performance Analytics")
    
    # Try to load model comparison results
    try:
        results_df = pd.read_csv('figures/model_results.csv')
        st.markdown("#### Model Comparison")
        st.dataframe(results_df)
        
        # Model comparison chart
        fig = go.Figure()
        for model_name in results_df.index:
            fig.add_trace(go.Bar(
                name=model_name,
                x=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                y=results_df.loc[model_name, ['Accuracy', 'Precision', 'Recall', 'F1 Score']].values
            ))
        fig.update_layout(barmode='group', title="Performance Metrics by Model")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Run training first to see model comparison metrics")
    
    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    if os.path.exists('figures/confusion_matrix.png'):
        st.image('figures/confusion_matrix.png', use_container_width=True)
    else:
        st.info("Run training to generate confusion matrix")
    
    # ROC Curves
    st.markdown("#### ROC Curves (One-vs-Rest)")
    if os.path.exists('figures/roc_curves.png'):
        st.image('figures/roc_curves.png', use_container_width=True)
    else:
        st.info("Run training to generate ROC curves")
    
    # Per-Class Performance
    st.markdown("#### Per-Class Performance")
    if os.path.exists('figures/per_class_performance.png'):
        st.image('figures/per_class_performance.png', use_container_width=True)
    else:
        st.info("Run training to generate per-class performance chart")
    
    # Top Keywords per Category
    st.markdown("#### 🔑 Top Keywords per Category")
    
    if hasattr(model, 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        
        for i, category in enumerate(label_encoder.classes_):
            coef = model.coef_[i]
            top_idx = np.argsort(coef)[-10:][::-1]
            top_words = [feature_names[idx] for idx in top_idx]
            
            with st.expander(f"📌 {category}"):
                st.write(f"**Top indicators:** {', '.join(top_words)}")
    else:
        st.info("Feature importance not available for this model type")
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Classification")
    st.info("""
    - Use complete emails (50+ words) for better accuracy
    - Very short emails (<10 chars) will return UNCERTAIN
    - Adjust confidence threshold based on your needs
    - Lower threshold = more predictions (less accurate)
    - Higher threshold = fewer predictions (more reliable)
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8rem;'>Email Classification System | Powered by Logistic Regression</p>",
    unsafe_allow_html=True
)