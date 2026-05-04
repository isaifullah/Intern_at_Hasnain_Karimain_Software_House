"""
================================================================================
LANGUAGE DETECTION STREAMLIT APP
================================================================================
Using the provided dataset with 20+ languages
"""

# ============================================================================
# CELL 1: IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Try langdetect for comparison
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ============================================================================
# CELL 2: PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Language Detection System",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .prediction-box p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CELL 3: TEXT PREPROCESSING
# ============================================================================

def clean_text(text):
    """Clean text while preserving language-specific characters"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================================
# CELL 4: LOAD AND TRAIN MODEL (CACHED)
# ============================================================================

@st.cache_resource
def load_and_train_model():
    """Load the dataset and train the model"""
    
    # Try to load the dataset - update this path
    import os
    
    # Check common locations for the dataset
    possible_paths = [
        "language_detection_dataset.csv",
        "data/language_detection_dataset.csv",
        "../language_detection_dataset.csv",
        "Language_Detection_Dataset.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.success(f"✅ Loaded dataset from: {path}")
            break
    
    if df is None:
        st.error("❌ Dataset not found! Please upload your CSV file.")
        return None, None, None, None, None
    
    # Get text and language columns
    text_col = 'Text' if 'Text' in df.columns else 'text'
    lang_col = 'language' if 'language' in df.columns else 'Language'
    
    # Clean text
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Filter languages with minimum samples
    MIN_SAMPLES = 3
    lang_counts = df[lang_col].value_counts()
    languages_to_keep = lang_counts[lang_counts >= MIN_SAMPLES].index.tolist()
    df = df[df[lang_col].isin(languages_to_keep)]
    
    # Prepare data
    X = df['cleaned_text'].tolist()
    y = df[lang_col].tolist()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        max_features=20000,
        min_df=2,
        max_df=0.95
    )
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Create ensemble model
    model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model_nb = ComplementNB(alpha=0.1)
    model_svc = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=3)
    
    ensemble = VotingClassifier(
        estimators=[('lr', model_lr), ('nb', model_nb), ('svc', model_svc)],
        voting='soft',
        weights=[2, 1, 2]
    )
    
    # Train
    ensemble.fit(X_train_features, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    
    return ensemble, vectorizer, label_encoder, accuracy, df

# ============================================================================
# CELL 5: PREDICTION FUNCTION
# ============================================================================

def predict_language(text, model, vectorizer, label_encoder):
    """Predict language for given text"""
    if not text or len(text.strip()) < 3:
        return None, 0, []
    
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    
    # Get probabilities
    probabilities = model.predict_proba(features)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    
    top_predictions = []
    for idx in top_indices:
        lang = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx]
        top_predictions.append((lang, confidence))
    
    best_lang, best_conf = top_predictions[0]
    
    return best_lang, best_conf, top_predictions

# ============================================================================
# CELL 6: MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Multilingual Language Detection System</h1>
        <p>Detect languages from text using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, vectorizer, label_encoder, accuracy, df = load_and_train_model()
    
    if model is None:
        st.warning("📁 Please upload your dataset file to continue")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
        if uploaded_file is not None:
            # Save uploaded file
            with open("language_detection_dataset.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ File uploaded! Please refresh the page.")
            st.rerun()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Info")
        st.metric("🎯 Accuracy", f"{accuracy*100:.1f}%")
        st.metric("🌍 Languages", len(label_encoder.classes_))
        st.metric("📝 Samples", len(df))
        
        st.markdown("---")
        st.markdown("### Supported Languages")
        for lang in sorted(label_encoder.classes_):
            count = len(df[df['language'] == lang])
            st.markdown(f"✅ {lang} ({count} samples)")
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Type or paste text in the input box
        2. Click 'Detect Language'
        3. See instant results with confidence score
        4. View top 2 alternative predictions
        """)
    
    # Main content - Two tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Real-time Detection", "📊 Batch Prediction", "📈 Performance"])
    
    # TAB 1: Real-time Detection
    with tab1:
        st.markdown("### Enter text to detect language")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Text Input",
                height=150,
                placeholder="Type or paste text in any language...\n\nExamples:\nHello, how are you?\nآپ کیسے ہیں؟\n¿Cómo estás?\nBonjour le monde"
            )
            
            if st.button("🔍 Detect Language", type="primary", use_container_width=True):
                if user_input and len(user_input.strip()) >= 3:
                    with st.spinner("Analyzing..."):
                        lang, confidence, top_preds = predict_language(
                            user_input, model, vectorizer, label_encoder
                        )
                        
                        # Display result
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🎯 {lang}</h2>
                            <p>Confidence: {confidence*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top predictions
                        if len(top_preds) > 1:
                            st.markdown("#### 🏆 Top Predictions")
                            cols = st.columns(len(top_preds))
                            for i, (col, (pred_lang, pred_conf)) in enumerate(zip(cols, top_preds)):
                                with col:
                                    st.metric(
                                        f"#{i+1} {pred_lang}",
                                        f"{pred_conf*100:.1f}%"
                                    )
                        
                        # Text stats
                        with st.expander("📊 Text Statistics"):
                            st.text(f"Characters: {len(user_input)}")
                            st.text(f"Words: {len(user_input.split())}")
                            st.text(f"Unique characters: {len(set(user_input))}")
                            
                elif user_input and len(user_input.strip()) < 3:
                    st.warning("⚠️ Please enter at least 3 characters for reliable detection")
                else:
                    st.info("💡 Enter some text above to detect its language")
        
        with col2:
            st.markdown("#### 📝 Try Examples")
            examples = {
                "English": "The weather is beautiful today",
                "Urdu": "آج موسم بہت اچھا ہے",
                "Spanish": "¿Cómo estás amigo?",
                "French": "Bonjour, comment allez-vous?",
                "Arabic": "كيف حالك اليوم؟",
                "German": "Guten Morgen, wie geht es dir?"
            }
            
            for lang, example in examples.items():
                if st.button(f"📌 {lang}", key=lang):
                    user_input = example
                    st.rerun()
    
    # TAB 2: Batch Prediction
    with tab2:
        st.markdown("### Batch Prediction from CSV")
        st.markdown("Upload a CSV file with a 'text' column to get predictions for all entries")
        
        uploaded_batch = st.file_uploader("Upload CSV for batch prediction", type=['csv'], key="batch")
        
        if uploaded_batch is not None:
            batch_df = pd.read_csv(uploaded_batch)
            st.write("**Input Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Find text column
            text_column = None
            for col in batch_df.columns:
                if col.lower() in ['text', 'Text', 'sentence', 'Sentence', 'content', 'Content']:
                    text_column = col
                    break
            
            if text_column is None:
                text_column = batch_df.columns[0]
                st.info(f"Using column: '{text_column}'")
            
            if st.button("🚀 Process Batch", use_container_width=True):
                with st.spinner("Processing..."):
                    predictions = []
                    confidences = []
                    top2_list = []
                    
                    progress_bar = st.progress(0)
                    for i, text in enumerate(batch_df[text_column].head(100)):
                        if pd.isna(text) or len(str(text).strip()) < 3:
                            predictions.append("Unknown")
                            confidences.append("0%")
                            top2_list.append("N/A")
                        else:
                            lang, conf, top_preds = predict_language(
                                str(text), model, vectorizer, label_encoder
                            )
                            predictions.append(lang)
                            confidences.append(f"{conf*100:.1f}%")
                            if len(top_preds) > 1:
                                top2_list.append(f"{top_preds[1][0]} ({top_preds[1][1]*100:.0f}%)")
                            else:
                                top2_list.append("N/A")
                        
                        progress_bar.progress((i + 1) / min(len(batch_df), 100))
                    
                    # Add results to dataframe
                    result_df = batch_df.head(100).copy()
                    result_df['Predicted_Language'] = predictions
                    result_df['Confidence'] = confidences
                    result_df['Second_Best'] = top2_list
                    
                    st.success(f"✅ Processed {len(result_df)} texts")
                    
                    st.write("**Results:**")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download button
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name="language_detection_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    # TAB 3: Performance
    with tab3:
        st.markdown("### Model Performance")
        
        # Need to retrain to get test data
        X = df['cleaned_text'].tolist()
        y = df['language'].tolist()
        y_encoded = label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_test_features = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_features)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        
        # Classification Report
        with st.expander("📋 Classification Report", expanded=True):
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        
        # Confusion Matrix
        with st.expander("🎨 Confusion Matrix"):
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_,
                       ax=ax)
            ax.set_title('Confusion Matrix', fontsize=14)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig)
            plt.close()
        
        # Sample predictions
        with st.expander("🧪 Sample Predictions"):
            samples = []
            for i in range(min(20, len(X_test))):
                text = X_test[i][:80]
                true_lang = label_encoder.inverse_transform([y_test[i]])[0]
                pred_lang = label_encoder.inverse_transform([y_pred[i]])[0]
                correct = "✅" if y_test[i] == y_pred[i] else "❌"
                samples.append({
                    'Text': text,
                    'True': true_lang,
                    'Predicted': pred_lang,
                    'Status': correct
                })
            
            samples_df = pd.DataFrame(samples)
            st.dataframe(samples_df, use_container_width=True)
    
    # Compare with langdetect section
    if LANGDETECT_AVAILABLE:
        with st.sidebar.expander("🔄 Compare with langdetect"):
            st.markdown("Test a sentence against both models")
            test_text = st.text_input("Enter text to compare:", key="compare")
            if test_text:
                our_lang, our_conf, _ = predict_language(test_text, model, vectorizer, label_encoder)
                try:
                    ld_lang = detect(test_text)
                    # Map langdetect codes to full names
                    lang_map = {'en': 'English', 'ur': 'Urdu', 'es': 'Spanish', 'fr': 'French', 
                               'de': 'German', 'ar': 'Arabic', 'hi': 'Hindi'}
                    ld_lang = lang_map.get(ld_lang, ld_lang)
                except:
                    ld_lang = "Error"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Our Model:** {our_lang}")
                    st.markdown(f"*Confidence: {our_conf*100:.1f}%*")
                with col2:
                    st.markdown(f"**LangDetect:** {ld_lang}")
                
                if our_lang.lower() == ld_lang.lower():
                    st.success("✅ Models agree!")
                else:
                    st.info("🤔 Models disagree - try a different text")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>🌐 Powered by Machine Learning | Character-level n-gram features</p>",
        unsafe_allow_html=True
    )

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()