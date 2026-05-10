import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="NLP Model Comparison", page_icon="🤖", layout="wide")

# Download NLTK
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

download_nltk()

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    
    .header-container {
        text-align: center; padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; margin-bottom: 1.5rem;
    }
    .header-title { font-size: 2.5rem; font-weight: 700; color: white; margin: 0; }
    .header-subtitle { font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 0.3rem; }
    
    .metric-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0.8rem; text-align: center; color: white; margin: 0.3rem 0;
    }
    .metric-card h4 { font-size: 0.8rem; margin: 0 0 0.3rem 0; opacity: 0.8; }
    .metric-card h2 { font-size: 1.2rem; margin: 0; }
    
    .best-model {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px; padding: 1rem; text-align: center; color: white; margin: 0.8rem 0;
    }
    .best-model h3 { font-size: 0.9rem; margin: 0 0 0.3rem 0; }
    .best-model h2 { font-size: 1.3rem; margin: 0; }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2); color: white;
        border: none; padding: 0.6rem 1rem; font-size: 0.85rem; font-weight: 600;
        border-radius: 8px; transition: all 0.3s; height: 42px; width: 100%;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); }
    
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px; color: white; padding: 0.6rem; font-size: 0.9rem;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea; box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .sentiment-positive { color: #38ef7d; font-weight: 700; }
    .sentiment-negative { color: #ff6b6b; font-weight: 700; }
    .sentiment-neutral { color: #f39c12; font-weight: 700; }
    
    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 0.3rem; gap: 0.3rem; }
    .stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.6); font-weight: 500; font-size: 0.9rem; padding: 0.5rem 1rem; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #667eea; background: rgba(102, 126, 234, 0.1); border-radius: 8px; }
    
    .section-title { font-size: 1.3rem; font-weight: 600; color: #667eea; margin: 1rem 0 0.5rem 0; }
    
    .clear-btn button {
        background: rgba(255,255,255,0.08) !important; color: #aaa !important; 
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 0.6rem 1rem !important; font-size: 0.85rem !important; height: 42px !important;
    }
    .clear-btn button:hover { background: rgba(255,107,107,0.2) !important; color: #ff6b6b !important; border-color: #ff6b6b !important; }
    
    .input-row { display: flex; align-items: flex-end; gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Text preprocessing
@st.cache_resource
def create_preprocessor():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(tag):
        if tag.startswith('J'): return 'a'
        elif tag.startswith('V'): return 'v'
        elif tag.startswith('N'): return 'n'
        elif tag.startswith('R'): return 'r'
        else: return 'n'
    
    def preprocess_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\$\s?\d+(\.\d+)?', ' money ', text)
        text = re.sub(r'\d+(\.\d+)?', ' number ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        cleaned_words = []
        for word, tag in pos_tags:
            if word not in stop_words and len(word) > 2:
                pos = get_wordnet_pos(tag)
                lemma = lemmatizer.lemmatize(word, pos)
                cleaned_words.append(lemma)
        text = ' '.join(cleaned_words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    return preprocess_text

preprocess_text = create_preprocessor()

# Model keys and labels (defined globally)
model_keys = ['MultinomialNB', 'LinearSVC', 'LogisticRegression']
model_labels = ['Naive Bayes', 'LinearSVC', 'Logistic Regression']

# Create default model
@st.cache_resource
def create_default_model():
    training_data = [
        ("this product is amazing and works perfectly great quality", "positive"),
        ("i love this product fantastic quality excellent purchase", "positive"),
        ("excellent service wonderful experience very happy", "positive"),
        ("best purchase ever highly recommend great value", "positive"),
        ("outstanding product very happy with it amazing", "positive"),
        ("great value for money absolutely love it superb", "positive"),
        ("wonderful exceeded my expectations fantastic quality", "positive"),
        ("very satisfied with the quality perfect amazing", "positive"),
        ("incredible will definitely buy again excellent", "positive"),
        ("perfect fit works great so happy wonderful", "positive"),
        ("this product is terrible does not work at all", "negative"),
        ("worst purchase ever waste of money horrible", "negative"),
        ("very disappointed quality broke immediately bad", "negative"),
        ("poor service bad experience overall terrible", "negative"),
        ("do not buy this horrible useless garbage product", "negative"),
        ("hate this product completely worthless awful", "negative"),
        ("awful quality nothing like description poor", "negative"),
        ("total garbage broke after first use terrible", "negative"),
        ("extremely disappointed give zero stars worst", "negative"),
        ("pathetic product money wasted completely bad", "negative"),
    ]
    
    df = pd.DataFrame(training_data, columns=['review', 'sentiment'])
    df['review'] = df['review'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['review']).toarray()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])
    
    models = {
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(random_state=42, dual=False),
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42)
    }
    
    for model in models.values():
        model.fit(X, y)
    
    return models, vectorizer, label_encoder

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = 'LogisticRegression'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Load or create models
def load_or_create_models():
    try:
        if os.path.exists('models/all_models.pkl'):
            models = pickle.load(open('models/all_models.pkl', 'rb'))
            best_model_name = pickle.load(open('models/best_model_name.pkl', 'rb'))
            vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
            label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
            return models, vectorizer, label_encoder, best_model_name, True
    except:
        pass
    models, vectorizer, label_encoder = create_default_model()
    return models, vectorizer, label_encoder, 'LogisticRegression', False

models, vectorizer, label_encoder, best_model_name, from_saved = load_or_create_models()

st.session_state.models_trained = True
st.session_state.models = models
st.session_state.vectorizer = vectorizer
st.session_state.label_encoder = label_encoder
st.session_state.best_model_name = best_model_name

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 NLP Model Comparison</h1>
    <p class="header-subtitle">Train • Compare • Predict — Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)

# Quick Sentiment Check
st.markdown('<p class="section-title">⚡ Quick Sentiment Check</p>', unsafe_allow_html=True)

# Input row with aligned buttons
col1, col2, col3 = st.columns([5, 1.2, 1.2])
with col1:
    quick_text = st.text_input(
        "Enter text",
        placeholder="Type a review to analyze sentiment...",
        label_visibility="collapsed",
        key="quick_input_widget"
    )
with col2:
    predict_btn = st.button("🔍 Analyze", use_container_width=True, key="quick_btn")
with col3:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("✕ Clear", use_container_width=True, key="clear_btn"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn and quick_text:
    processed = preprocess_text(quick_text)
    vec = vectorizer.transform([processed]).toarray()
    
    predictions = {}
    
    cols = st.columns(3)
    for idx, (key, label) in enumerate(zip(model_keys, model_labels)):
        pred = models[key].predict(vec)[0]
        sentiment = label_encoder.inverse_transform([pred])[0]
        predictions[label] = sentiment
        
        emoji = "😊" if sentiment.lower() == "positive" else "😞" if sentiment.lower() == "negative" else "😐"
        
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{label}</h4>
                <h2 class="sentiment-{sentiment.lower()}">{sentiment.upper()} {emoji}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    best_pred = models[best_model_name].predict(vec)[0]
    best_sentiment = label_encoder.inverse_transform([best_pred])[0]
    
    st.markdown(f"""
    <div class="best-model">
        <h3>🏆 Best Model: {best_model_name}</h3>
        <h2>Final Prediction → {best_sentiment.upper()}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Batch Prediction", "📊 Results & Analytics", "📂 Dataset & Training"])

# Tab 1: Batch Prediction
with tab1:
    st.markdown('<p class="section-title">🎯 Batch Sentiment Prediction</p>', unsafe_allow_html=True)
    
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=180,
        placeholder="This product is amazing and works perfectly!\nTerrible experience, would not recommend.\nIt's okay, nothing special."
    )
    
    if st.button("🔮 Predict All", type="primary", use_container_width=True):
        if batch_text:
            texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
            predictions_list = []
            
            for text in texts:
                processed = preprocess_text(text)
                vec = vectorizer.transform([processed]).toarray()
                
                preds = {'Text': text[:60] + '...' if len(text) > 60 else text}
                
                for key, label in zip(model_keys, model_labels):
                    pred = models[key].predict(vec)[0]
                    preds[label] = label_encoder.inverse_transform([pred])[0]
                
                best_pred = models[best_model_name].predict(vec)[0]
                preds['🏆 Best'] = label_encoder.inverse_transform([best_pred])[0]
                
                predictions_list.append(preds)
            
            pred_df = pd.DataFrame(predictions_list)
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.error("Please enter some text!")

# Tab 2: Results & Analytics
with tab2:
    st.markdown('<p class="section-title">📊 Results & Analytics</p>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.info("👈 Train models on your dataset to see detailed analytics!")
    else:
        results = st.session_state.results
        
        results_df = pd.DataFrame({
            'Model': model_labels,
            'Accuracy': [results[k]['accuracy'] for k in model_keys],
            'Precision': [results[k]['precision'] for k in model_keys],
            'Recall': [results[k]['recall'] for k in model_keys],
            'F1 Score': [results[k]['f1_score'] for k in model_keys]
        }).set_index('Model')
        
        st.dataframe(results_df.style.highlight_max(axis=0, color='#11998e'), use_container_width=True)
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Accuracy", "Confusion Matrix", "All Metrics"])
        
        with viz_tab1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(results_df.index, results_df['Accuracy'], color=['#667eea', '#764ba2', '#38ef7d'])
            for i, v in enumerate(results_df['Accuracy']):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12, color='white')
            ax.set_ylim(0, 1.15)
            ax.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14, color='white')
            ax.set_ylabel('Accuracy', color='white')
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#1a1a2e')
            st.pyplot(fig)
        
        with viz_tab2:
            if st.session_state.X_test is not None:
                fig, axes = plt.subplots(3, 1, figsize=(8, 13))
                for idx, (key, label) in enumerate(zip(model_keys, model_labels)):
                    cm = confusion_matrix(st.session_state.y_test, results[key]['y_pred'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[idx],
                               xticklabels=st.session_state.label_encoder.classes_,
                               yticklabels=st.session_state.label_encoder.classes_,
                               annot_kws={'size': 12, 'fontweight': 'bold'})
                    axes[idx].set_title(f'Confusion Matrix - {label}', fontweight='bold', fontsize=12, color='white')
                    axes[idx].set_xlabel('Predicted', color='white')
                    axes[idx].set_ylabel('Actual', color='white')
                    axes[idx].set_facecolor('#1a1a2e')
                    axes[idx].tick_params(colors='white')
                fig.patch.set_facecolor('#1a1a2e')
                plt.tight_layout()
                st.pyplot(fig)
        
        with viz_tab3:
            fig, ax = plt.subplots(figsize=(10, 5))
            results_df.plot(kind='bar', ax=ax, color=['#667eea', '#764ba2', '#38ef7d', '#f39c12'])
            ax.set_ylim(0, 1.15)
            ax.set_title('All Metrics Comparison', fontweight='bold', fontsize=14, color='white')
            ax.set_ylabel('Score', color='white')
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.legend(loc='lower right', fontsize=10)
            plt.xticks(rotation=0)
            fig.patch.set_facecolor('#1a1a2e')
            st.pyplot(fig)

# Tab 3: Dataset & Training
with tab3:
    st.markdown('<p class="section-title">📂 Dataset & Training</p>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("**Data Source**")
        data_option = st.radio("Choose input method:", ["📁 Upload CSV", "✏️ Enter Text Manually"], horizontal=True, label_visibility="collapsed")
        
        if data_option == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload sentiment dataset (CSV)", type=['csv'], label_visibility="collapsed")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} samples")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                with c2:
                    if 'sentiment' in df.columns:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        df['sentiment'].value_counts().plot(kind='bar', ax=ax, color=['#667eea', '#764ba2', '#11998e'])
                        ax.set_facecolor('#1a1a2e')
                        ax.tick_params(colors='white')
                        fig.patch.set_facecolor('#1a1a2e')
                        st.pyplot(fig)
        else:
            text_input = st.text_area("Enter texts (one per line):", height=100, placeholder="Amazing product!\nTerrible experience\nIt's okay")
            sentiment_input = st.text_area("Enter sentiments (one per line):", height=100, placeholder="positive\nnegative\nneutral")
            
            if st.button("📊 Create Dataset", use_container_width=True):
                texts = [t.strip() for t in text_input.split('\n') if t.strip()]
                sentiments = [s.strip() for s in sentiment_input.split('\n') if s.strip()]
                if len(texts) == len(sentiments):
                    st.session_state.df = pd.DataFrame({'review': texts, 'sentiment': sentiments})
                    st.success(f"✅ Created {len(texts)} samples!")
                else:
                    st.error("Count mismatch!")
    
    with col_right:
        st.markdown("**Training Parameters**")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        max_features = st.selectbox("Max Features", [3000, 5000, 7000])
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if 'df' in st.session_state and st.session_state.df is not None:
                with st.spinner("Training models..."):
                    df = st.session_state.df.copy()
                    df = df.dropna(subset=['review']).drop_duplicates()
                    
                    def filter_short(text):
                        if not isinstance(text, str): return np.nan
                        if len(text.split()) < 20: return np.nan
                        return text
                    
                    df['review'] = df['review'].apply(filter_short)
                    df = df.dropna(subset=['review']).reset_index(drop=True)
                    
                    progress = st.empty()
                    progress.info("📝 Preprocessing text...")
                    df['review'] = df['review'].apply(preprocess_text)
                    df = df[df['review'] != ""]
                    
                    progress.info("🔢 Extracting features...")
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(df['sentiment'])
                    
                    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
                    X = vectorizer.fit_transform(df['review']).toarray()
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                    
                    progress.info("🤖 Training models...")
                    models = {
                        'MultinomialNB': MultinomialNB(),
                        'LinearSVC': LinearSVC(random_state=42, dual=False),
                        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42)
                    }
                    
                    results = {}
                    pbar = st.progress(0)
                    
                    for idx, (name, model) in enumerate(models.items()):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        results[name] = {
                            'model': model, 'y_pred': y_pred,
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        }
                        pbar.progress((idx + 1) / 3)
                    
                    best_name = max(results, key=lambda x: results[x]['accuracy'])
                    
                    progress.info(f"🔧 Tuning {best_name}...")
                    if best_name == 'LogisticRegression':
                        grid = GridSearchCV(LogisticRegression(max_iter=500, random_state=42), {"C": [0.1, 1, 10]}, cv=3)
                        grid.fit(X_train, y_train)
                        best_model = LogisticRegression(max_iter=500, random_state=42, **grid.best_params_)
                        best_model.fit(X_train, y_train)
                    elif best_name == 'LinearSVC':
                        grid = GridSearchCV(LinearSVC(random_state=42, dual=False), {"C": [0.1, 1, 10]}, cv=3)
                        grid.fit(X_train, y_train)
                        best_model = LinearSVC(random_state=42, dual=False, **grid.best_params_)
                        best_model.fit(X_train, y_train)
                    else:
                        best_model = results[best_name]['model']
                    
                    os.makedirs('models', exist_ok=True)
                    pickle.dump(models, open('models/all_models.pkl', 'wb'))
                    pickle.dump(best_name, open('models/best_model_name.pkl', 'wb'))
                    pickle.dump(best_model, open('models/best_model.pkl', 'wb'))
                    pickle.dump(vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))
                    pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))
                    
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.vectorizer = vectorizer
                    st.session_state.label_encoder = label_encoder
                    st.session_state.best_model_name = best_name
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    pbar.empty()
                    progress.empty()
                    
                    st.markdown(f"""
                    <div class="best-model">
                        <h3>🎉 Training Complete!</h3>
                        <h3>🏆 Best Model: {best_name}</h3>
                        <h4>Accuracy: {results[best_name]['accuracy']:.4f}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3)
                    colors = ['#667eea', '#764ba2', '#38ef7d']
                    
                    for col, key, label, color in zip([c1, c2, c3], model_keys, model_labels, colors):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card" style="background: {color};">
                                <h4>{label}</h4>
                                <h2>{results[key]['accuracy']:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.rerun()
            else:
                st.error("❌ Please upload or create a dataset first!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; padding: 1rem; font-size: 0.85rem;">
    NLP Model Comparison Dashboard • Built with Streamlit
</div>
""", unsafe_allow_html=True)