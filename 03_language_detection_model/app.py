"""
================================================================================
PROFESSIONAL LANGUAGE DETECTION SYSTEM - STREAMLIT APP
================================================================================
"""

# ============================================================================
# CELL 1: IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CELL 3: CUSTOM CSS - DARK THEME, NO WHITE BACKGROUNDS
# ============================================================================

st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="collapsedControl"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    /* Main dark background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Remove white backgrounds from ALL components */
    .stMarkdown, .stTextArea, .stButton, .stDataFrame, 
    .stAlert, .stSelectbox, .stFileUploader, .stExpander,
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
    .element-container, .stTabs, .stSpinner {
        background-color: transparent !important;
    }
    
    /* Header styling - dark theme */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #2a3f5e;
    }
    
    .main-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: #a8b8d4 !important;
        margin-top: 0.5rem;
    }
    
    /* Prediction box - transparent with border */
    .prediction-box {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #2a3f5e;
    }
    
    .prediction-box h2 {
        color: #ffffff !important;
        font-size: 3rem;
        margin: 0;
    }
    
    .prediction-box p {
        color: #a8b8d4 !important;
        font-size: 1.2rem;
    }
    
    /* Confidence bar */
    .confidence-bar-container {
        background-color: #2a2a3e;
        border-radius: 10px;
        height: 35px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 15px;
        color: #ffffff !important;
        font-weight: bold;
    }
    
    /* Cards - dark theme, no white */
    .card {
        background: #1a1a2e;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #2a3f5e;
    }
    
    .card h3 {
        color: #00b4d8 !important;
        margin-top: 0;
    }
    
    .card p, .card li {
        color: #c8d6e5 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: #ffffff !important;
        border: 1px solid #2a3f5e;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a4a7a 0%, #2a2a4e 100%);
        border-color: #00b4d8;
    }
    
    /* Tabs - dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        color: #a8b8d4 !important;
        background-color: #1a1a2e;
        border: 1px solid #2a3f5e;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: #00b4d8 !important;
        border-color: #00b4d8;
    }
    
    /* Text area - dark background */
    .stTextArea textarea {
        background-color: #1a1a2e !important;
        color: #c8d6e5 !important;
        border: 1px solid #2a3f5e !important;
        border-radius: 10px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00b4d8 !important;
    }
    
    /* Info/Warning/Success boxes - dark theme */
    .stAlert {
        background-color: #1a1a2e !important;
        border-left: 4px solid;
    }
    
    /* Dataframe - dark theme */
    .stDataFrame {
        background-color: #1a1a2e !important;
        border: 1px solid #2a3f5e;
        border-radius: 10px;
    }
    
    /* Expander - dark theme */
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        color: #00b4d8 !important;
        border: 1px solid #2a3f5e;
        border-radius: 10px;
    }
    
    .streamlit-expanderContent {
        background-color: #16213e !important;
        border: 1px solid #2a3f5e;
        border-top: none;
    }
    
    /* Language badges */
    .lang-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 25px;
        margin: 0.2rem;
        background-color: #0f3460;
        color: #00b4d8 !important;
        border: 1px solid #2a3f5e;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    
    /* Paragraph text */
    p, li, span, label {
        color: #c8d6e5 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c7a89;
        border-top: 1px solid #2a3f5e;
        margin-top: 2rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00b4d8, #0077b6);
    }
    
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1a1a2e !important;
        border-color: #2a3f5e !important;
        color: #c8d6e5 !important;
    }
    
    /* Metrics row - remove the blocks */
    div[data-testid="column"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CELL 4: TEXT PREPROCESSING
# ============================================================================

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF\u00C0-\u017F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================================
# CELL 5: LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    try:
        with open('models/language_detection_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return (
            model_data['model'],
            model_data['vectorizer'],
            model_data['label_encoder'],
            model_data['accuracy'],
            model_data['model_name']
        )
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        return None, None, None, None, None

# ============================================================================
# CELL 6: PREDICTION FUNCTION
# ============================================================================

def predict_language(text, model, vectorizer, label_encoder, top_n=3):
    if not text or len(text.strip()) == 0:
        return {
            'language': 'Unknown',
            'confidence_percent': '0%',
            'confidence': 0,
            'message': 'Empty text',
            'top_predictions': []
        }
    
    cleaned = clean_text(text)
    
    if len(cleaned) < 5:
        if any(ord(c) > 0x0600 and ord(c) < 0x06FF for c in text):
            return {
                'language': 'Urdu/Arabic',
                'confidence': 0.6,
                'confidence_percent': '60%',
                'message': 'Short text - based on script',
                'top_predictions': []
            }
        else:
            return {
                'language': 'Uncertain',
                'confidence': 0.3,
                'confidence_percent': 'Low',
                'message': 'Text too short (min 5 chars)',
                'top_predictions': []
            }
    
    features = vectorizer.transform([cleaned])
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        top_predictions = []
        for idx in top_indices:
            lang = label_encoder.inverse_transform([idx])[0]
            conf = probabilities[idx]
            top_predictions.append({
                'language': lang,
                'confidence': conf,
                'confidence_percent': f"{conf*100:.1f}%"
            })
        
        is_mixed = False
        if len(top_predictions) > 1:
            if top_predictions[0]['confidence'] - top_predictions[1]['confidence'] < 0.15:
                is_mixed = True
        
        return {
            'language': top_predictions[0]['language'],
            'confidence': top_predictions[0]['confidence'],
            'confidence_percent': top_predictions[0]['confidence_percent'],
            'top_predictions': top_predictions,
            'message': 'Mixed language detected' if is_mixed else 'Success',
            'is_mixed': is_mixed
        }
    else:
        prediction = model.predict(features)[0]
        lang = label_encoder.inverse_transform([prediction])[0]
        return {
            'language': lang,
            'confidence': 1.0,
            'confidence_percent': "100%",
            'top_predictions': [],
            'message': 'Success',
            'is_mixed': False
        }

# ============================================================================
# CELL 7: MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Language Detection System</h1>
        <p>Advanced multilingual text classification using Character-level TF-IDF & Ensemble Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, vectorizer, label_encoder, accuracy, model_name = load_model()
    
    if model is None:
        return
    
    # Main tabs - 4 tabs including Model Performance
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Real-time Detection", "📊 Batch Prediction", "📈 Model Performance", "ℹ️ Language Info"])
    
    # ========================================================================
    # TAB 1: Real-time Detection
    # ========================================================================
    with tab1:
        col_input, col_examples = st.columns([2, 1])
        
        with col_input:
            user_input = st.text_area(
                "Enter text:",
                height=120,
                placeholder="Type or paste text in any language...\n\nExamples:\n• Hello, how are you today?\n• آپ کیسے ہیں؟\n• ¿Cómo estás?\n• Bonjour le monde",
                label_visibility="collapsed"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                detect_clicked = st.button("🔍 Detect Language", type="primary", use_container_width=True)
            with col_btn2:
                clear_clicked = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_clicked:
                user_input = ""
                st.rerun()
        
        with col_examples:
            st.markdown("#### 📝 Quick Examples")
            examples = {
                "🇬🇧 English": "The weather is beautiful today",
                "🇵🇰 Urdu": "آج موسم بہت اچھا ہے",
                "🇪🇸 Spanish": "¿Cómo estás amigo?",
                "🇫🇷 French": "Bonjour, comment allez-vous?",
                "🇩🇪 German": "Guten Morgen, wie geht es dir?",
                "🇸🇦 Arabic": "كيف حالك اليوم؟"
            }
            
            for lang, example in examples.items():
                if st.button(f"{lang}", key=f"ex_{lang}", use_container_width=True):
                    user_input = example
                    st.rerun()
        
        # Detection logic
        if detect_clicked and user_input:
            if len(user_input.strip()) < 3:
                st.warning("⚠️ Please enter at least 3 characters for reliable detection")
            else:
                with st.spinner("🧠 Analyzing text..."):
                    result = predict_language(user_input, model, vectorizer, label_encoder)
                    
                    # Display prediction box
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 {result['language']}</h2>
                        <p>Confidence: {result['confidence_percent']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    if result['confidence'] > 0 and result['confidence_percent'] != 'Low':
                        conf_percent = result['confidence'] * 100
                        st.markdown(f"""
                        <div class="confidence-bar-container">
                            <div class="confidence-bar" style="width: {conf_percent}%;">
                                {conf_percent:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Message
                    if result.get('message') and result['message'] != 'Success':
                        st.info(result['message'])
                    
                    # Top predictions
                    if result['top_predictions'] and len(result['top_predictions']) > 1:
                        st.markdown("#### 🏆 Top Language Predictions")
                        cols = st.columns(min(len(result['top_predictions']), 3))
                        for i, (col, pred) in enumerate(zip(cols, result['top_predictions'][:3])):
                            with col:
                                st.markdown(f"""
                                <div class="card" style="text-align: center;">
                                    <h3 style="margin: 0; color: #00b4d8;">#{i+1}</h3>
                                    <h4 style="margin: 0.5rem 0; color: #ffffff;">{pred['language']}</h4>
                                    <p style="font-size: 1.2rem; font-weight: bold; color: #00b4d8;">{pred['confidence_percent']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Text statistics
                    with st.expander("📊 Text Statistics"):
                        st.markdown(f"""
                        - **Characters:** {len(user_input)}
                        - **Words:** {len(user_input.split())}
                        - **Unique characters:** {len(set(user_input))}
                        - **Cleaned length:** {len(clean_text(user_input))}
                        """)
        
        elif detect_clicked and not user_input:
            st.info("💡 Enter some text above to detect its language")
    
    # ========================================================================
    # TAB 2: Batch Prediction
    # ========================================================================
    with tab2:
        st.markdown("### 📁 Batch Prediction from CSV File")
        st.markdown("Upload a CSV file with a text column to detect languages for multiple entries at once")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="batch_upload")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Found {len(batch_df)} rows")
            
            st.markdown("#### 📄 Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Find text column
            text_column = None
            possible_columns = ['text', 'Text', 'sentence', 'Sentence', 'content', 'Content', 'review', 'Review']
            for col in possible_columns:
                if col in batch_df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                text_column = st.selectbox("Select the column containing text:", batch_df.columns)
            
            st.info(f"📌 Using column: **{text_column}** for language detection")
            
            if st.button("🚀 Start Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    predictions = []
                    confidences = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(batch_df[text_column].iloc[:500]):
                        status_text.text(f"Processing {i+1}/{min(len(batch_df), 500)}...")
                        
                        if pd.isna(text) or len(str(text).strip()) < 3:
                            predictions.append("Unknown")
                            confidences.append("0%")
                        else:
                            result = predict_language(str(text), model, vectorizer, label_encoder)
                            predictions.append(result['language'])
                            confidences.append(result['confidence_percent'])
                        
                        progress_bar.progress((i + 1) / min(len(batch_df), 500))
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Create results dataframe
                    result_df = batch_df.iloc[:500].copy()
                    result_df['Predicted_Language'] = predictions
                    result_df['Confidence'] = confidences
                    
                    st.success(f"✅ Successfully processed {len(result_df)} texts")
                    
                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download button
                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="language_detection_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown("#### 📈 Summary Statistics")
                    summary_df = result_df['Predicted_Language'].value_counts().reset_index()
                    summary_df.columns = ['Language', 'Count']
                    st.dataframe(summary_df, use_container_width=True)
    
    # ========================================================================
    # TAB 3: Model Performance
    # ========================================================================
    with tab3:
        st.markdown("### 📊 Model Performance Metrics")
        
        st.markdown(f"""
        <div class="card">
            <h3>📌 Model Overview</h3>
            <p><strong>Best Model:</strong> {model_name}</p>
            <p><strong>Accuracy:</strong> {accuracy*100:.2f}%</p>
            <p><strong>Languages Supported:</strong> {len(label_encoder.classes_)}</p>
            <p><strong>Feature Type:</strong> Character-level TF-IDF (n-gram range: 2-5)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 All Supported Languages")
        languages = sorted(label_encoder.classes_)
        
        # Display languages in a grid
        cols = st.columns(5)
        for i, lang in enumerate(languages):
            with cols[i % 5]:
                st.markdown(f'<span class="lang-badge">🌐 {lang}</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📋 Model Architecture Details")
        
        col_desc1, col_desc2 = st.columns(2)
        with col_desc1:
            st.markdown("""
            <div class="card">
                <h3>🔧 Feature Engineering</h3>
                <ul>
                    <li>Character-level TF-IDF features</li>
                    <li>N-gram range: (2, 5)</li>
                    <li>Max features: 30,000</li>
                    <li>Min document frequency: 2</li>
                    <li>Max document frequency: 0.95</li>
                    <li>Unicode normalization (NFKC)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_desc2:
            st.markdown("""
            <div class="card">
                <h3>🤖 Model Training</h3>
                <ul>
                    <li>Train-test split: 80/20</li>
                    <li>Stratified sampling</li>
                    <li>Cross-validation: 5-fold</li>
                    <li>Best model selected based on test accuracy</li>
                    <li>Probability calibration for confidence scores</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 💡 Tips for Best Results")
        st.markdown("""
        <div class="card">
            <ul>
                <li>✅ Use sentences with 5+ characters for reliable detection</li>
                <li>✅ Single-language text gives most accurate results</li>
                <li>✅ Mixed-language texts may show uncertain results with lower confidence</li>
                <li>✅ The model works best with natural, complete sentences</li>
                <li>✅ Very short texts (2-4 characters) will show low confidence warnings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Confusion Matrix if we have test data
        st.markdown("#### 🎨 Confusion Matrix Analysis")
        st.info("📊 For detailed confusion matrix and classification report, please refer to the training notebook where test data is available.")
    
    # ========================================================================
    # TAB 4: Language Info
    # ========================================================================
    with tab4:
        st.markdown("### 🌍 Language Detection Information")
        
        st.markdown("""
        <div class="card">
            <h3>📖 About This System</h3>
            <p>
                This language detection system uses advanced machine learning techniques to identify languages from text input.
                It's trained on multilingual data and can detect multiple languages with high accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🧪 Sample Predictions by Language")
        
        sample_texts = {
            "English": "The quick brown fox jumps over the lazy dog",
            "Urdu": "آج کا دن بہت خوبصورت ہے",
            "Spanish": "Me encanta aprender nuevos idiomas cada día",
            "French": "J'apprends le français chaque jour avec plaisir",
            "German": "Ich lerne Deutsch mit viel Spaß und Freude",
            "Arabic": "أتعلم اللغة العربية كل يوم بشكل مستمر"
        }
        
        for lang, sample in sample_texts.items():
            if lang in languages:
                with st.expander(f"🌐 {lang}"):
                    st.markdown(f"**Example:** \"{sample}\"")
                    result = predict_language(sample, model, vectorizer, label_encoder)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Predicted:** {result['language']}")
                    with col2:
                        st.markdown(f"**Confidence:** {result['confidence_percent']}")
    
    # ========================================================================
    # Compare with langdetect (optional section)
    # ========================================================================
    if LANGDETECT_AVAILABLE:
        with st.expander("🔄 Compare with langdetect Library"):
            st.markdown("Test a sentence against both our model and the langdetect library")
            compare_text = st.text_input("Enter text to compare:", key="compare_input", placeholder="Type a sentence...")
            
            if compare_text:
                our_result = predict_language(compare_text, model, vectorizer, label_encoder)
                try:
                    ld_result = detect(compare_text)
                    lang_map = {'en': 'English', 'ur': 'Urdu', 'es': 'Spanish', 'fr': 'French', 
                               'de': 'German', 'ar': 'Arabic', 'hi': 'Hindi', 'zh-cn': 'Chinese',
                               'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'it': 'Italian',
                               'pt': 'Portuguese', 'nl': 'Dutch', 'tr': 'Turkish'}
                    ld_lang = lang_map.get(ld_result, ld_result)
                except:
                    ld_lang = "Error"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <h3>🤖 Our Model</h3>
                        <h2 style="color: #00b4d8;">{our_result['language']}</h2>
                        <p>Confidence: {our_result['confidence_percent']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <h3>📚 langdetect</h3>
                        <h2 style="color: #00b4d8;">{ld_lang}</h2>
                        <p>Library-based detection</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if our_result['language'].lower()[:2] == ld_lang.lower()[:2]:
                    st.success("✅ Both models agree on the language!")
                else:
                    st.info("🤔 Models disagree - different approaches may give different results")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌐 Powered by Machine Learning | Character-level TF-IDF Features | Ensemble Learning</p>
        <p style="font-size: 0.8rem;">Language Detection System | Professional ML Model</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()