# ================= IMPORT LIBRARIES =================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

from sklearn.preprocessing import LabelEncoder, label_binarize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

download_nltk_resources()

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Advanced NLP preprocessing pipeline
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # 4. Normalize currency
    text = re.sub(r'\$\s?\d+(\.\d+)?', ' money ', text)
    
    # 5. Normalize numbers
    text = re.sub(r'\d+(\.\d+)?', ' number ', text)
    
    # 6. Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 7. Tokenization
    tokens = word_tokenize(text)
    
    # 8. POS tagging
    pos_tags = pos_tag(tokens)
    
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'
    
    # 9. Lemmatization + stopword removal
    cleaned_words = []
    for word, tag in pos_tags:
        if word not in stop_words and len(word) > 2:
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos)
            cleaned_words.append(lemma)
    
    # 10. Join and clean
    text = ' '.join(cleaned_words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Model Evaluation Dashboard", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= DARK THEME CSS =================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: #e94560;
        margin: 0;
        font-size: 2rem;
    }
    
    .main-header p {
        color: #ccc;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e94560;
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(233,69,96,0.2);
    }
    
    .metric-card h3 {
        color: #e94560;
        margin: 0;
        font-size: 0.85rem;
    }
    
    .metric-card h1 {
        color: white;
        margin: 0.5rem 0;
        font-size: 2rem;
    }
    
    /* Section header styling */
    .section-header {
        background: linear-gradient(90deg, #0f3460 0%, #1a1a2e 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e94560;
        margin: 1.5rem 0 1rem 0;
    }
    
    .section-header h3 {
        color: #e94560;
        margin: 0;
        font-size: 1.25rem;
    }
    
    /* Best model card styling */
    .best-model-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        color: white;
    }
    
    .best-model-card strong {
        color: #28a745;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background: #16213e;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        border-top: 1px solid #333;
        margin-top: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: #e94560;
        border: 1px solid #e94560;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
        color: white;
        border: 1px solid #0f3460;
        box-shadow: 0 4px 12px rgba(233,69,96,0.4);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #0f3460;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #ccc;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e94560;
        color: white;
    }
    
    /* Warning box styling */
    .warning-box {
        background: linear-gradient(135deg, #330000 0%, #1a0000 100%);
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <h1>📊 Model Evaluation Dashboard</h1>
    <p>Compare machine learning models with interactive visualizations and comprehensive metrics</p>
</div>
""", unsafe_allow_html=True)

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("📁 Upload Dataset (CSV)", type=["csv"], help="Upload a CSV file with text and label columns")

# =========================================================
# MAIN APP
# =========================================================
if uploaded_file is not None:
    
    # Load data with progress indicator
    with st.spinner("Loading dataset..."):
        time.sleep(0.5)
        df = pd.read_csv(uploaded_file)
    
    # ================= CREATE TABS =================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "⚙️ Model Training", 
        "📈 Results & Metrics", 
        "🔍 Advanced Analysis",
        "📊 Interactive Visualizations",
        "💾 Export Results"
    ])
    
    with tab1:
        st.markdown('<div class="section-header"><h3>📂 Dataset Overview</h3></div>', unsafe_allow_html=True)
        
        # ================= METRIC CARDS (TOP - MOST IMPORTANT) =================
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Rows</h3>
                <h1>{df.shape[0]:,}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Columns</h3>
                <h1>{df.shape[1]}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_total = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Missing Values</h3>
                <h1>{missing_total:,}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            duplicate_count = df.duplicated().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Duplicate Rows</h3>
                <h1>{duplicate_count:,}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Numeric Columns</h3>
                <h1>{numeric_cols}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            categorical_cols = df.select_dtypes(include=['object']).shape[1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Categorical Cols</h3>
                <h1>{categorical_cols}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # ================= MISSING & DUPLICATE WARNINGS (BELOW METRICS) =================
        missing_cols = df.columns[df.isnull().any()].tolist()
        duplicate_count = df.duplicated().sum()
        
        if missing_cols or duplicate_count > 0:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Data Quality Issues Detected:</strong><br>
            </div>
            """, unsafe_allow_html=True)
            
            # Missing Values Section
            if missing_cols:
                st.warning(f"📊 Missing values found in columns: {', '.join(missing_cols)}")
                missing_df = pd.DataFrame({
                    'Column': missing_cols,
                    'Missing Count': df[missing_cols].isnull().sum().values,
                    'Missing Percentage': (df[missing_cols].isnull().sum().values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
                
                rows_with_missing = df[df[missing_cols].isnull().any(axis=1)]
                st.markdown(f"**Rows with Missing Values (Total: {len(rows_with_missing)} rows):**")
                st.dataframe(rows_with_missing, use_container_width=True)
            
            # Duplicate Values Section
            if duplicate_count > 0:
                st.warning(f"🔄 Found {duplicate_count} duplicate rows in the dataset")
                duplicate_rows = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
                st.markdown(f"**Duplicate Rows Found (Total: {len(duplicate_rows)} rows):**")
                st.dataframe(duplicate_rows, use_container_width=True)
        else:
            st.success("✅ No missing values or duplicate rows found in the dataset!")
        
        # Data preview
        st.markdown("#### 🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Comprehensive Dataset Summary
        with st.expander("📊 Comprehensive Dataset Summary", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Statistical Summary (Numeric Columns)**")
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    st.dataframe(df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found")
                
                st.markdown("**📊 Data Types & Missing Info**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Null Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
                    'Unique Values': df.nunique().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Correlation Matrix Heatmap**")
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr()
                    fig_corr = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu',
                                          title='Feature Correlation Heatmap')
                    fig_corr.update_layout(height=500, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation heatmap")
                
                st.markdown("**📊 Missing Values Visualization**")
                if df.isnull().sum().sum() > 0:
                    missing_df_viz = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum().values
                    })
                    missing_df_viz = missing_df_viz[missing_df_viz['Missing Count'] > 0]
                    fig_missing = px.bar(missing_df_viz, x='Column', y='Missing Count', title='Missing Values per Column',
                                        color='Missing Count', color_continuous_scale='Reds')
                    fig_missing.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("✅ No missing values found in the dataset!")
        
        # Interactive Distribution Plots
        st.markdown("#### 📊 Interactive Data Visualizations")
        
        # Select column for visualization
        viz_col = st.selectbox("Select column for visualization", df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if df[viz_col].dtype in ['int64', 'float64']:
                fig_hist = px.histogram(df, x=viz_col, title=f'Distribution of {viz_col}',
                                        marginal='box', color_discrete_sequence=['#e94560'])
                fig_hist.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)
                
                fig_box = px.box(df, y=viz_col, title=f'Box Plot of {viz_col}', color_discrete_sequence=['#e94560'])
                fig_box.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                value_counts = df[viz_col].value_counts().head(20)
                fig_bar = px.bar(x=value_counts.index, y=value_counts.values, title=f'Top 20 Categories in {viz_col}',
                                color=value_counts.values, color_continuous_scale='Reds')
                fig_bar.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     xaxis_title=viz_col, yaxis_title='Count')
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            if df[viz_col].dtype in ['int64', 'float64']:
                fig_violin = px.violin(df, y=viz_col, title=f'Violin Plot of {viz_col}', color_discrete_sequence=['#e94560'])
                fig_violin.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                top_categories = df[viz_col].value_counts().head(10)
                fig_pie = px.pie(values=top_categories.values, names=top_categories.index, title=f'Top 10 Categories in {viz_col}',
                                color_discrete_sequence=px.colors.sequential.Reds_r)
                fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header"><h3>⚙️ Column Selection & Training</h3></div>', unsafe_allow_html=True)
        
        # Smart column detection
        cols = df.columns
        text_keywords = ["text", "message", "content", "email", "review", "comment", "sentence", "body", "description"]
        label_keywords = ["label", "class", "target", "category", "sentiment", "spam", "type"]
        
        possible_text_cols = [c for c in cols if any(k in c.lower() for k in text_keywords)]
        possible_label_cols = [c for c in cols if any(k in c.lower() for k in label_keywords)]
        
        if not possible_text_cols:
            possible_text_cols = list(cols)
        if not possible_label_cols:
            possible_label_cols = list(cols)
        
        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox("📝 Select TEXT column", possible_text_cols, 
                                   help="Column containing the text data to analyze")
        with col2:
            label_col = st.selectbox("🏷️ Select LABEL column", possible_label_cols,
                                   help="Column containing the target labels")
        
        if text_col == label_col:
            st.error("⚠️ Text and Label columns must be different")
            st.stop()
        
        # Advanced settings in expander
        with st.expander("⚙️ Advanced Training Settings"):
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42)
            with col3:
                max_features = st.slider("Max TF-IDF Features", 1000, 10000, 5000, 500)
            
            st.markdown("**Model Selection:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                train_lr = st.checkbox("Logistic Regression", value=True)
            with col2:
                train_svm = st.checkbox("SVM", value=True)
            with col3:
                train_nb = st.checkbox("Naive Bayes", value=True)
        
        # Class distribution preview
        st.markdown("#### 📊 Class Distribution")
        class_counts = df[label_col].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index, title='Target Variable Distribution',
                    color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Train button
        train_button = st.button("🚀 Train Models", use_container_width=True, type="primary")
        
        if train_button:
            # Preprocess data
            with st.spinner("Preprocessing text data..."):
                df[text_col] = df[text_col].astype(str)
                df[text_col] = df[text_col].apply(preprocess_text)
                
                X = df[text_col]
                y = df[label_col]
                
                # Label encoding
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Train-test split
            with st.spinner("Splitting dataset..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            
            # TF-IDF Vectorization
            with st.spinner("Applying TF-IDF vectorization..."):
                vectorizer = TfidfVectorizer(max_features=max_features)
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
            
            # Initialize models
            models = {}
            if train_lr:
                models["Logistic Regression"] = LogisticRegression(max_iter=1000)
            if train_svm:
                models["SVM"] = SVC(probability=True)
            if train_nb:
                models["Naive Bayes"] = MultinomialNB()
            
            if not models:
                st.error("Please select at least one model to train.")
                st.stop()
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            preds_dict = {}
            
            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train_vec, y_train)
                preds = model.predict(X_test_vec)
                preds_dict[name] = preds
                
                results.append([
                    name,
                    accuracy_score(y_test, preds),
                    precision_score(y_test, preds, average='weighted'),
                    recall_score(y_test, preds, average='weighted'),
                    f1_score(y_test, preds, average='weighted')
                ])
                
                progress_bar.progress((i + 1) / len(models))
            
            results_df = pd.DataFrame(results, columns=[
                "Model", "Accuracy", "Precision", "Recall", "F1-score"
            ])
            
            status_text.text("✅ Training complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"Successfully trained {len(models)} model(s)!")
            st.balloons()
            
            # Store in session state
            st.session_state.results_df = results_df
            st.session_state.models = models
            st.session_state.preds_dict = preds_dict
            st.session_state.y_test = y_test
            st.session_state.X_test_vec = X_test_vec
            st.session_state.label_encoder = le
            st.session_state.vectorizer = vectorizer
            st.session_state.trained = True
    
    with tab3:
        if 'trained' in st.session_state and st.session_state.trained:
            results_df = st.session_state.results_df
            
            st.markdown('<div class="section-header"><h3>📈 Model Performance Results</h3></div>', unsafe_allow_html=True)
            
            # Interactive metrics chart
            st.markdown("#### 📊 Interactive Metrics Comparison")
            
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
            colors = ['#e94560', '#ff6b6b', '#c0392b', '#e74c3c']
            
            for metric, color in zip(metrics, colors):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df['Model'],
                    y=results_df[metric],
                    marker_color=color,
                    text=results_df[metric].round(4),
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group',
                height=500,
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table with formatting
            st.markdown("#### 📋 Detailed Metrics Table")
            styled_df = results_df.style.background_gradient(cmap='Reds', subset=['Accuracy', 'Precision', 'Recall', 'F1-score'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Best model highlight
            best_model = results_df.loc[results_df['F1-score'].idxmax(), 'Model']
            best_score = results_df['F1-score'].max()
            
            st.markdown(f"""
            <div class="best-model-card">
                <strong>🏆 Best Performing Model:</strong> {best_model} achieved F1-score of {best_score:.4f}
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection for detailed view
            st.markdown("#### 🔍 Model-Specific Analysis")
            selected_model = st.selectbox("Select Model for Detailed Analysis", results_df["Model"])
            
            if selected_model:
                selected_data = results_df[results_df["Model"] == selected_model].iloc[0]
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{selected_data['Accuracy']:.4f}", delta=f"{selected_data['Accuracy']-results_df['Accuracy'].mean():+.4f} vs avg")
                with col2:
                    st.metric("Precision", f"{selected_data['Precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{selected_data['Recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{selected_data['F1-score']:.4f}")
                
                # Confusion Matrix
                st.markdown("#### 📊 Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.preds_dict[selected_model])
                
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Reds',
                                   labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                   title=f'Confusion Matrix - {selected_model}')
                fig_cm.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Classification Report
                with st.expander("📄 View Detailed Classification Report"):
                    report = classification_report(st.session_state.y_test, st.session_state.preds_dict[selected_model], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
        else:
            st.info("ℹ️ Please train models in the 'Model Training' tab first.")
    
    with tab4:
        if 'trained' in st.session_state and st.session_state.trained:
            st.markdown('<div class="section-header"><h3>🔍 Advanced Analysis</h3></div>', unsafe_allow_html=True)
            
            # Side-by-side model comparison
            st.markdown("#### ⚖️ Side-by-Side Model Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                model1 = st.selectbox("Select First Model", st.session_state.results_df["Model"], key="model1")
            with col2:
                model2 = st.selectbox("Select Second Model", st.session_state.results_df["Model"], key="model2")
            
            if model1 and model2:
                model1_data = st.session_state.results_df[st.session_state.results_df["Model"] == model1].iloc[0]
                model2_data = st.session_state.results_df[st.session_state.results_df["Model"] == model2].iloc[0]
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                    model1: [model1_data['Accuracy'], model1_data['Precision'], model1_data['Recall'], model1_data['F1-score']],
                    model2: [model2_data['Accuracy'], model2_data['Precision'], model2_data['Recall'], model2_data['F1-score']],
                    'Difference': [
                        model1_data['Accuracy'] - model2_data['Accuracy'],
                        model1_data['Precision'] - model2_data['Precision'],
                        model1_data['Recall'] - model2_data['Recall'],
                        model1_data['F1-score'] - model2_data['F1-score']
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Comparison bar chart
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(name=model1, x=comparison_df['Metric'], y=comparison_df[model1], marker_color='#e94560'))
                fig_compare.add_trace(go.Bar(name=model2, x=comparison_df['Metric'], y=comparison_df[model2], marker_color='#ff6b6b'))
                fig_compare.update_layout(title=f"Model Comparison: {model1} vs {model2}", barmode='group', height=450,
                                         template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_compare, use_container_width=True)
            
            # ROC Curve
            st.markdown("#### 📈 ROC Curve Analysis")
            selected_roc = st.selectbox("Select Model for ROC Curve", st.session_state.results_df["Model"], key="roc_model")
            
            if selected_roc and selected_roc in st.session_state.models:
                model = st.session_state.models[selected_roc]
                y_test = st.session_state.y_test
                X_test_vec = st.session_state.X_test_vec
                
                if hasattr(model, 'predict_proba'):
                    num_classes = len(np.unique(y_test))
                    
                    if num_classes == 2:
                        probs = model.predict_proba(X_test_vec)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})',
                                                      line=dict(color='#e94560', width=2)))
                        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier', 
                                                     line=dict(dash='dash', color='gray')))
                        fig_roc.update_layout(title=f'ROC Curve - {selected_roc}', 
                                             xaxis_title='False Positive Rate', 
                                             yaxis_title='True Positive Rate', 
                                             height=450,
                                             template='plotly_dark', 
                                             paper_bgcolor='rgba(0,0,0,0)', 
                                             plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info(f"Multi-class ROC curves are available for {selected_roc}")
                        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                        probs = model.predict_proba(X_test_vec)
                        
                        fig_roc_multi = go.Figure()
                        for i in range(num_classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                            roc_auc = auc(fpr, tpr)
                            fig_roc_multi.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                                               name=f'Class {i} (AUC = {roc_auc:.3f})'))
                        
                        fig_roc_multi.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                                           line=dict(dash='dash', color='gray')))
                        fig_roc_multi.update_layout(title=f'Multi-class ROC Curves - {selected_roc}',
                                                    xaxis_title='False Positive Rate',
                                                    yaxis_title='True Positive Rate',
                                                    height=450,
                                                    template='plotly_dark',
                                                    paper_bgcolor='rgba(0,0,0,0)',
                                                    plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_roc_multi, use_container_width=True)
                else:
                    st.warning(f"{selected_roc} does not support probability predictions. Try Logistic Regression or SVM with probability=True")
        else:
            st.info("ℹ️ Please train models in the 'Model Training' tab first.")
    
    with tab5:
        if 'trained' in st.session_state and st.session_state.trained:
            st.markdown('<div class="section-header"><h3>📊 Interactive Visualizations</h3></div>', unsafe_allow_html=True)
            
            # ================= SUNBURST CHART =================
            st.markdown("#### 🎯 Model Performance - Sunburst Chart")
            st.markdown("*Sunburst chart shows hierarchical performance - better visual comparison than radar charts*")
            
            # Prepare data for sunburst chart
            sunburst_data = []
            for idx, row in st.session_state.results_df.iterrows():
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
                    sunburst_data.append({
                        'Model': row['Model'],
                        'Metric': metric,
                        'Score': row[metric]
                    })
            
            sunburst_df = pd.DataFrame(sunburst_data)
            
            fig_sunburst = px.sunburst(sunburst_df, 
                                        path=['Model', 'Metric'], 
                                        values='Score',
                                        color='Score',
                                        color_continuous_scale='Reds',
                                        title='Model Performance Hierarchy - Larger areas = Better Performance',
                                        height=600)
            fig_sunburst.update_layout(template='plotly_dark', 
                                       paper_bgcolor='rgba(0,0,0,0)', 
                                       plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            with st.expander("ℹ️ How to read the Sunburst Chart"):
                st.markdown("""
                - **Larger colored areas** indicate better performance
                - **Deeper red color** means higher scores
                - Each model has 4 segments (Accuracy, Precision, Recall, F1-score)
                - Hover over any segment to see exact values
                - Click on a model to zoom in and focus
                """)
            
            # ================= BUBBLE CHART =================
            st.markdown("#### 🎯 Performance Bubble Chart")
            st.markdown("*Bubble chart shows all metrics simultaneously - bubble size represents F1-score*")
            
            # Prepare data for bubble chart
            bubble_data = []
            for idx, row in st.session_state.results_df.iterrows():
                bubble_data.append({
                    'Model': row['Model'],
                    'Accuracy': row['Accuracy'],
                    'Precision': row['Precision'],
                    'Recall': row['Recall'],
                    'F1_Score': row['F1-score'],
                    'Marker_Size': row['F1-score'] * 100
                })
            
            bubble_df = pd.DataFrame(bubble_data)
            
            fig_bubble = go.Figure()
            
            for idx, row in bubble_df.iterrows():
                fig_bubble.add_trace(go.Scatter(
                    x=[row['Accuracy']],
                    y=[row['Precision']],
                    mode='markers+text',
                    marker=dict(
                        size=row['Marker_Size'],
                        color='#e94560',
                        line=dict(color='white', width=2),
                        sizemode='area',
                        sizeref=2.*max(bubble_df['Marker_Size'])/(100**2),
                        sizemin=4
                    ),
                    text=row['Model'],
                    textposition="top center",
                    name=row['Model'],
                    hovertemplate=f"<b>{row['Model']}</b><br>" +
                                  f"Accuracy: {row['Accuracy']:.4f}<br>" +
                                  f"Precision: {row['Precision']:.4f}<br>" +
                                  f"Recall: {row['Recall']:.4f}<br>" +
                                  f"F1-Score: {row['F1_Score']:.4f}<br>" +
                                  "<extra></extra>"
                ))
            
            fig_bubble.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Balance',
                line=dict(dash='dash', color='gray'),
                showlegend=True
            ))
            
            fig_bubble.update_layout(
                title='Bubble Chart: Accuracy vs Precision (Bubble Size = F1-Score)',
                xaxis_title='Accuracy',
                yaxis_title='Precision',
                xaxis=dict(range=[0, 1], gridcolor='#333'),
                yaxis=dict(range=[0, 1], gridcolor='#333'),
                height=550,
                hovermode='closest',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_bubble, use_container_width=True)
            
            with st.expander("ℹ️ How to read the Bubble Chart"):
                st.markdown("""
                - **X-axis**: Accuracy score
                - **Y-axis**: Precision score  
                - **Bubble size**: F1-score (larger bubble = better overall performance)
                - **Ideal position**: Top-right corner (high accuracy AND high precision)
                - **Dashed line**: Perfect balance line
                - **Hover** over any bubble for full details
                """)
            
            # ================= AREA CHART =================
            st.markdown("#### 📈 Cumulative Performance Area Chart")
            st.markdown("*Shows how each model performs across all metrics cumulatively*")
            
            area_df = st.session_state.results_df.melt(id_vars=['Model'], 
                                                        var_name='Metric', 
                                                        value_name='Score')
            
            fig_area = px.area(area_df, x='Metric', y='Score', color='Model',
                               title='Cumulative Performance Across Metrics',
                               color_discrete_sequence=px.colors.sequential.Reds_r)
            
            fig_area.update_layout(
                xaxis_title='Metrics',
                yaxis_title='Score',
                height=450,
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_area, use_container_width=True)
            
            # ================= WATERFALL CHART =================
            st.markdown("#### 📊 Performance Waterfall Chart")
            st.markdown("*Shows the progression of metrics from baseline*")
            
            selected_waterfall_model = st.selectbox("Select Model for Waterfall Analysis", 
                                                     st.session_state.results_df["Model"],
                                                     key="waterfall_model")
            
            if selected_waterfall_model:
                model_data = st.session_state.results_df[st.session_state.results_df["Model"] == selected_waterfall_model].iloc[0]
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
                values = [model_data[m] for m in metrics]
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name="Performance",
                    orientation="v",
                    measure=["absolute"] + ["relative"] * (len(metrics)-1),
                    x=metrics,
                    y=values,
                    text=[f"{v:.4f}" for v in values],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#28a745"}},
                    decreasing={"marker": {"color": "#e94560"}},
                    totals={"marker": {"color": "#0f3460"}}
                ))
                
                fig_waterfall.update_layout(
                    title=f'Performance Waterfall - {selected_waterfall_model}',
                    yaxis_title="Score",
                    height=450,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # ================= HEATMAP =================
            st.markdown("#### 🔥 Model Performance Heatmap")
            
            metrics_matrix = st.session_state.results_df[['Accuracy', 'Precision', 'Recall', 'F1-score']].T
            fig_heatmap = px.imshow(metrics_matrix.values, 
                                    x=st.session_state.results_df['Model'], 
                                    y=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                                    text_auto='.4f',
                                    color_continuous_scale='Reds',
                                    aspect='auto',
                                    title='Performance Heatmap (Deeper Red = Better)')
            fig_heatmap.update_layout(height=450, 
                                     template='plotly_dark', 
                                     paper_bgcolor='rgba(0,0,0,0)', 
                                     plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # ================= GAUGE CHARTS =================
            st.markdown("#### 🎯 Best Model Performance Gauges")
            
            best_model_idx = st.session_state.results_df['F1-score'].idxmax()
            best_model_data = st.session_state.results_df.loc[best_model_idx]
            
            best_accuracy = float(best_model_data['Accuracy']) * 100
            best_precision = float(best_model_data['Precision']) * 100
            best_recall = float(best_model_data['Recall']) * 100
            best_f1 = float(best_model_data['F1-score']) * 100
            best_model_name = best_model_data['Model']
            
            st.markdown(f"**🏆 Best Model: {best_model_name}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=best_accuracy,
                    title={'text': "Accuracy (%)", 'font': {'color': '#e94560'}},
                    gauge={'axis': {'range': [0, 100]}, 
                          'bar': {'color': "#e94560"},
                          'steps': [
                              {'range': [0, 50], 'color': "#330000"},
                              {'range': [50, 75], 'color': "#661111"},
                              {'range': [75, 100], 'color': "#992222"}
                          ]}
                ))
                fig_gauge.update_layout(height=250, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=best_precision,
                    title={'text': "Precision (%)", 'font': {'color': '#ff6b6b'}},
                    gauge={'axis': {'range': [0, 100]}, 
                          'bar': {'color': "#ff6b6b"},
                          'steps': [
                              {'range': [0, 50], 'color': "#330000"},
                              {'range': [50, 75], 'color': "#661111"},
                              {'range': [75, 100], 'color': "#992222"}
                          ]}
                ))
                fig_gauge.update_layout(height=250, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col3:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=best_recall,
                    title={'text': "Recall (%)", 'font': {'color': '#c0392b'}},
                    gauge={'axis': {'range': [0, 100]}, 
                          'bar': {'color': "#c0392b"},
                          'steps': [
                              {'range': [0, 50], 'color': "#330000"},
                              {'range': [50, 75], 'color': "#661111"},
                              {'range': [75, 100], 'color': "#992222"}
                          ]}
                ))
                fig_gauge.update_layout(height=250, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col4:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=best_f1,
                    title={'text': "F1-Score (%)", 'font': {'color': '#e74c3c'}},
                    gauge={'axis': {'range': [0, 100]}, 
                          'bar': {'color': "#e74c3c"},
                          'steps': [
                              {'range': [0, 50], 'color': "#330000"},
                              {'range': [50, 75], 'color': "#661111"},
                              {'range': [75, 100], 'color': "#992222"}
                          ]}
                ))
                fig_gauge.update_layout(height=250, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # ================= PARALLEL CATEGORIES =================
            st.markdown("#### 🔄 Parallel Categories Diagram")
            
            cat_data = st.session_state.results_df.copy()
            cat_data['Performance_Level'] = pd.cut(cat_data['F1-score'], 
                                                    bins=[0, 0.6, 0.8, 1.0], 
                                                    labels=['Low', 'Medium', 'High'])
            
            fig_parallel_cat = px.parallel_categories(cat_data, 
                                                       dimensions=['Model', 'Performance_Level'],
                                                       color='F1-score',
                                                       color_continuous_scale='Reds',
                                                       title='Model Performance Categories')
            fig_parallel_cat.update_layout(height=400, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_parallel_cat, use_container_width=True)
            
        else:
            st.info("ℹ️ Please train models in the 'Model Training' tab first.")
    
    with tab6:
        if 'trained' in st.session_state and st.session_state.trained:
            st.markdown('<div class="section-header"><h3>💾 Export Results</h3></div>', unsafe_allow_html=True)
            
            st.markdown("#### 📥 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = st.session_state.results_df.to_csv(index=False)
                st.download_button("📊 Download Results as CSV", data=csv, file_name="model_results.csv", mime="text/csv", use_container_width=True)
            
            with col2:
                report_data = {
                    "Best Model": st.session_state.results_df.loc[st.session_state.results_df['F1-score'].idxmax(), 'Model'],
                    "Best F1-Score": f"{st.session_state.results_df['F1-score'].max():.4f}",
                    "Best Accuracy": f"{st.session_state.results_df['Accuracy'].max():.4f}",
                    "Average Accuracy": f"{st.session_state.results_df['Accuracy'].mean():.4f}",
                    "Models Trained": len(st.session_state.results_df)
                }
                report_df = pd.DataFrame([report_data])
                csv_report = report_df.to_csv(index=False)
                st.download_button("📋 Download Summary Report", data=csv_report, file_name="summary_report.csv", mime="text/csv", use_container_width=True)
            
            with col3:
                predictions_df = pd.DataFrame(st.session_state.preds_dict)
                csv_pred = predictions_df.to_csv(index=False)
                st.download_button("🎯 Download Predictions", data=csv_pred, file_name="predictions.csv", mime="text/csv", use_container_width=True)
            
            st.markdown("#### 📋 Summary Statistics")
            
            summary_data = {
                "Metric": ["Total Models Trained", "Best Model", "Best F1-Score", "Best Accuracy", "Average Accuracy"],
                "Value": [
                    len(st.session_state.results_df),
                    st.session_state.results_df.loc[st.session_state.results_df['F1-score'].idxmax(), 'Model'],
                    f"{st.session_state.results_df['F1-score'].max():.4f}",
                    f"{st.session_state.results_df['Accuracy'].max():.4f}",
                    f"{st.session_state.results_df['Accuracy'].mean():.4f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Please train models in the 'Model Training' tab first.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%); border-radius: 1rem; margin: 2rem 0; border: 1px solid #e94560;">
        <h2 style="color: #e94560;">🚀 Welcome to Model Evaluation Dashboard</h2>
        <p style="font-size: 1.1rem; color: #ccc;">Upload a CSV file to start comparing machine learning models</p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span style="background: #e9456020; padding: 0.5rem 1rem; border-radius: 2rem; color: #e94560;">📊 Logistic Regression</span>
            <span style="background: #e9456020; padding: 0.5rem 1rem; border-radius: 2rem; color: #e94560;">🎯 SVM</span>
            <span style="background: #e9456020; padding: 0.5rem 1rem; border-radius: 2rem; color: #e94560;">📈 Naive Bayes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📁 Need a sample dataset?"):
        st.markdown("""
        Your CSV file should have:
        - A **text column** containing messages/reviews/content
        - A **label column** containing categories/classes
        
        **Sample format:**
        | text | label |
        |------|-------|
        | This is great! | positive |
        | Very disappointed | negative |
        """)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    <p>Model Evaluation Dashboard | Built with Streamlit, Scikit-learn, and Plotly</p>
</div>
""", unsafe_allow_html=True)