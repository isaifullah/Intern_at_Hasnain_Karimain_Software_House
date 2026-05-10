"""
Fake News Detection System - Streamlit Web Application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import plotly.graph_objects as go
import plotly.express as px
import base64
from datetime import datetime
import json
import os
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    .fake-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .real-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .uncertain-box {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
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
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        model_info = {}
        try:
            with open('models/model_info.json', 'r') as f:
                model_info = json.load(f)
        except:
            pass
        return model, vectorizer, label_encoder, model_info
    except Exception as e:
        return None, None, None, {}

# ============================================
# TEXT PREPROCESSING
# ============================================
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn

download_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

URL_PATTERN = re.compile(r'http\S+|www\S+')
HTML_PATTERN = re.compile(r'<.*?>')
MENTION_PATTERN = re.compile(r'@\w+')
NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?')
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')
MULTI_SPACE_PATTERN = re.compile(r'\s+')

def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    return wn.NOUN

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = URL_PATTERN.sub(' ', text)
    text = HTML_PATTERN.sub(' ', text)
    text = MENTION_PATTERN.sub(' ', text)
    text = NUMBER_PATTERN.sub(' number ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = REPEATED_CHARS_PATTERN.sub(r'\1\1', text)
    
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    if not tokens:
        return ""
    
    try:
        pos_tags = pos_tag(tokens)
        cleaned_words = []
        for word, tag in pos_tags:
            if word not in stop_words and len(word) > 2:
                wordnet_tag = get_wordnet_pos(tag)
                lemma = lemmatizer.lemmatize(word, wordnet_tag)
                cleaned_words.append(lemma)
    except:
        cleaned_words = [lemmatizer.lemmatize(word) for word in tokens 
                       if word not in stop_words and len(word) > 2]
    
    final_words = []
    for word in cleaned_words:
        if not final_words or final_words[-1] != word:
            final_words.append(word)
    
    text = ' '.join(final_words)
    text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
    
    return text

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_news(text, model, vectorizer, threshold=0.65):
    if not text or len(text.strip()) < 10:
        return {
            'prediction': 'INVALID',
            'label': '⚠️ Too short',
            'confidence': 0.0,
            'fake_prob': 0.0,
            'real_prob': 0.0,
        }
    
    cleaned = preprocess_text(text)
    
    if len(cleaned.split()) < 3:
        return {
            'prediction': 'UNCERTAIN',
            'label': '⚠️ Low quality',
            'confidence': 0.0,
            'fake_prob': 0.0,
            'real_prob': 0.0,
        }
    
    X_input = vectorizer.transform([cleaned])
    probs = model.predict_proba(X_input)[0]
    fake_prob, real_prob = probs[0], probs[1]
    confidence = max(probs)
    prediction = 1 if real_prob > fake_prob else 0
    
    if confidence < threshold:
        result = 'UNCERTAIN'
        label = '⚠️ UNCERTAIN'
    else:
        result = 'REAL' if prediction == 1 else 'FAKE'
        label = '✅ REAL' if prediction == 1 else '❌ FAKE'
    
    return {
        'prediction': result,
        'label': label,
        'confidence': confidence,
        'fake_prob': fake_prob,
        'real_prob': real_prob,
    }

# ============================================
# BATCH PREDICTION FUNCTION
# ============================================
def process_batch(df, text_column, model, vectorizer, threshold):
    """Process batch predictions"""
    results = []
    for idx, row in df.iterrows():
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        result = predict_news(text, model, vectorizer, threshold)
        results.append(result)
    
    df['prediction'] = [r['prediction'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    df['fake_prob'] = [r['fake_prob'] for r in results]
    df['real_prob'] = [r['real_prob'] for r in results]
    
    return df

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 📊 Model Info")
    
    model, vectorizer, label_encoder, model_info = load_models()
    
    if model_info:
        st.markdown("**Logistic Regression**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1 Score", f"{model_info.get('best_f1_score', 0):.3f}")
        with col2:
            st.metric("Accuracy", f"{model_info.get('accuracy', 0):.3f}")
        st.metric("Features", f"{model_info.get('features_count', 0):,}")
    else:
        st.info("Run training first")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Confidence Threshold",
        0.5, 0.95, 0.65, 0.05
    )
    
    st.markdown("---")
    st.caption("AI-Powered Fake News Detection")

# ============================================
# MAIN HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>📰 Fake News Detection System</h1>
    <p>Machine Learning powered news authenticity classifier</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Models not loaded. Please run the training notebook first.")
else:
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Processing", "📊 Analytics & Help"])
    
    # ============================================
    # TAB 1: SINGLE PREDICTION
    # ============================================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            news_text = st.text_area(
                "Enter news text:",
                height=200,
                placeholder="Paste news article here..."
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze = st.button("🔍 Analyze", use_container_width=True)
            with col_btn2:
                clear = st.button("🗑️ Clear", use_container_width=True)
        
        with col2:
            st.markdown("**Quick Samples**")
            if st.button("📰 Fake News Sample", use_container_width=True):
                news_text = "BREAKING: Government hiding cure for cancer! Secret vaccine microchips! SHARE NOW!"
                st.rerun()
            if st.button("📰 Real News Sample", use_container_width=True):
                news_text = "Federal Reserve announces 0.25% interest rate cut to support economic growth."
                st.rerun()
        
        if clear:
            news_text = ""
            st.rerun()
        
        if analyze and news_text:
            with st.spinner("Analyzing..."):
                result = predict_news(news_text, model, vectorizer, threshold)
                
                if result['prediction'] == 'FAKE':
                    st.markdown(f'<div class="prediction-box fake-box"><h2>{result["label"]}</h2><p>⚠️ This shows patterns of misinformation</p></div>', unsafe_allow_html=True)
                elif result['prediction'] == 'REAL':
                    st.markdown(f'<div class="prediction-box real-box"><h2>{result["label"]}</h2><p>✅ This appears to be legitimate</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box uncertain-box"><h2>{result["label"]}</h2><p>⚠️ Low confidence - please verify</p></div>', unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col_m2:
                    st.metric("Fake Probability", f"{result['fake_prob']:.1%}")
                with col_m3:
                    st.metric("Real Probability", f"{result['real_prob']:.1%}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['real_prob'] * 100,
                    title={'text': "Real News Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ff6b6b"},
                            {'range': [30, 70], 'color': "#ffd93d"},
                            {'range': [70, 100], 'color': "#6bcf7f"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 2: BATCH PROCESSING (FIXED - SIMPLER)
    # ============================================
    with tab2:
        st.markdown("### 📁 Batch Prediction")
        st.write("Upload a CSV file to classify multiple articles at once.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="batch_uploader")
        
        if uploaded_file is not None:
            # Load the file
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 **Loaded {len(df)} rows**")
            
            with st.expander("Preview uploaded data"):
                st.dataframe(df.head())
            
            # Column selection
            st.markdown("### Select Text Column")
            available_columns = df.columns.tolist()
            
            # Default to 'text' if it exists, otherwise first column
            default_index = available_columns.index('text') if 'text' in available_columns else 0
            
            selected_column = st.selectbox(
                "Which column contains the news text?",
                options=available_columns,
                index=default_index,
                help="Select the column that has the news articles you want to analyze"
            )
            
            st.info(f"📝 Using column: **{selected_column}**")
            
            # Show sample of selected column
            with st.expander(f"Sample of '{selected_column}' column"):
                sample_texts = df[selected_column].head(3).tolist()
                for i, text in enumerate(sample_texts, 1):
                    st.write(f"{i}. {str(text)[:200]}...")
            
            # Process button
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} articles..."):
                    # Process the batch
                    result_df = process_batch(df.copy(), selected_column, model, vectorizer, threshold)
                    
                    st.success("✅ Processing complete!")
                    
                    # Show results
                    st.markdown("### 📊 Results")
                    
                    # Display results dataframe
                    display_cols = [selected_column, 'prediction', 'confidence', 'fake_prob', 'real_prob']
                    st.dataframe(result_df[display_cols])
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Articles", len(result_df))
                    with col2:
                        real_count = (result_df['prediction'] == 'REAL').sum()
                        st.metric("✅ REAL News", real_count)
                    with col3:
                        fake_count = (result_df['prediction'] == 'FAKE').sum()
                        st.metric("❌ FAKE News", fake_count)
                    with col4:
                        uncertain_count = (result_df['prediction'] == 'UNCERTAIN').sum()
                        st.metric("⚠️ UNCERTAIN", uncertain_count)
                    
                    # Pie chart
                    fig_pie = px.pie(
                        values=[real_count, fake_count, uncertain_count],
                        names=['REAL News', 'FAKE News', 'UNCERTAIN'],
                        title='Prediction Distribution',
                        color_discrete_sequence=['#4facfe', '#f5576c', '#f6d365']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Confidence distribution
                    fig_hist = px.histogram(
                        result_df, 
                        x='confidence',
                        nbins=20,
                        title='Confidence Score Distribution',
                        color_discrete_sequence=['#667eea']
                    )
                    fig_hist.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Bar chart for predictions by confidence level
                    result_df['confidence_level'] = pd.cut(
                        result_df['confidence'],
                        bins=[0, 0.6, 0.75, 0.9, 1],
                        labels=['Low (<0.6)', 'Moderate (0.6-0.75)', 'High (0.75-0.9)', 'Very High (>0.9)']
                    )
                    
                confidence_counts = result_df['confidence_level'].value_counts().reset_index()
                confidence_counts.columns = ['Confidence Level', 'Count']
                
                fig_bar = px.bar(
                    confidence_counts,
                    x='Confidence Level',
                    y='Count',
                    title='Predictions by Confidence Level',
                    color='Confidence Level',
                    color_discrete_sequence=['#ff6b6b', '#ffd93d', '#6bcf7f', '#2ecc71']
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Download button
                csv = result_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                href = f'<a href="data:file/csv;base64,{b64}" download="batch_results_{timestamp}.csv" style="text-decoration: none;"><div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.75rem; border-radius: 8px; text-align: center; color: white; font-weight: bold;">📥 Download Results CSV</div></a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # ============================================
    # TAB 3: ANALYTICS & HELP
    # ============================================
    with tab3:
        st.markdown("### 📊 Model Analytics")
        
        if model_info:
            # Display model metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Best Model", model_info.get('best_model_name', 'Logistic Regression'))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{model_info.get('best_f1_score', 0):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{model_info.get('accuracy', 0):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("### 🔑 Top Predictive Words")
        
        if hasattr(model, 'coef_'):
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            
            top_fake_idx = np.argsort(coefficients)[:15]
            top_real_idx = np.argsort(coefficients)[-15:][::-1]
            
            col_fake, col_real = st.columns(2)
            
            with col_fake:
                st.markdown("#### ❌ Fake News Indicators")
                fake_words = [(feature_names[i], coefficients[i]) for i in top_fake_idx]
                for word, coef in fake_words:
                    st.write(f"• **{word}** ({coef:.4f})")
            
            with col_real:
                st.markdown("#### ✅ Real News Indicators")
                real_words = [(feature_names[i], coefficients[i]) for i in top_real_idx]
                for word, coef in real_words:
                    st.write(f"• **{word}** ({coef:.4f})")
        
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        
        st.markdown("""
        **Single Prediction**
        1. Enter or paste news text
        2. Click "Analyze"
        3. View prediction with confidence score
        
        **Batch Processing**
        1. Prepare CSV file (any column name works)
        2. Upload the file
        3. Select which column contains the news text
        4. Click "Process Batch"
        5. Download results with predictions
        
        **Understanding Results**
        - **REAL** ✅: Content appears legitimate
        - **FAKE** ❌: Shows misinformation patterns  
        - **UNCERTAIN** ⚠️: Confidence below threshold
        
        **Tips for Best Results**
        - Use articles with 100+ words
        - Lower threshold = more predictions (less accurate)
        - Higher threshold = fewer predictions (more reliable)
        """)
        
        st.info("💡 **Pro Tip:** Always verify important information with multiple reliable sources.")

print("✅ App ready! Run with: streamlit run streamlit_app.py")