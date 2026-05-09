"""
📝 AI Text Summarizer - Professional Edition
"""

import re
import numpy as np
import pandas as pd
import nltk
import streamlit as st
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# DARK PROFESSIONAL THEME - LARGER FONTS
# ============================================

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: #0a0a0a;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #e94560;
        margin: 0.8rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    /* Settings Container */
    .settings-container {
        background: #111111;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #333;
    }
    
    /* Input Container */
    .input-container {
        background: #111111;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    .input-label {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        display: block;
    }
    
    /* Summary Box */
    .summary-box {
        background: #111111;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #e94560;
        margin: 0.8rem 0;
        color: #e0e0e0;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    
    /* Highlight Box */
    .highlight-box {
        background: #111111;
        padding: 1.5rem;
        border-radius: 12px;
        color: #e0e0e0;
        border-left: 4px solid #e94560;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    .highlight {
        background-color: #e94560;
        color: white;
        padding: 5px 15px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Professional Stats Cards */
    .stats-container {
        background: #111111;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    .stats-title {
        color: #e94560;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
        letter-spacing: 1px;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #333;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #e94560;
    }
    
    .stat-card h2 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .stat-card p {
        color: #888;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Results Title */
    .results-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e94560;
        display: inline-block;
    }
    
    /* Buttons */
    .stButton > button {
        background: #e94560;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(233,69,96,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #111111;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1a1a1a;
        color: #888;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #e94560;
        color: white;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #0a0a0a;
        color: #e0e0e0;
        border: 1px solid #333;
        border-radius: 10px;
        font-size: 1rem;
        min-height: 150px;
    }
    
    .stTextArea textarea:focus {
        border-color: #e94560;
        box-shadow: 0 0 5px rgba(233,69,96,0.3);
    }
    
    /* Selectbox & Slider */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #0a0a0a;
        border-color: #333;
        font-size: 1rem;
    }
    
    .stSlider label {
        color: #e0e0e0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #e94560;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%);
        border-radius: 12px;
        color: #666;
        font-size: 0.85rem;
        border-top: 1px solid #333;
    }
    
    /* Headers */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #e94560 !important;
    }
    
    /* Caption */
    .stCaption {
        font-size: 0.9rem;
        color: #888;
    }
    
    /* Divider */
    hr {
        border-color: #333;
        margin: 1.5rem 0;
    }
    
    /* Success message */
    .stAlert {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-size: 0.9rem;
        border-left-color: #e94560;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_bart_model():
    with st.spinner("🔄 Loading BART model..."):
        return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_bart_model()

# ============================================
# FUNCTIONS
# ============================================

def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text, list(range(len(sentences)))
    
    cleaned = [re.sub(r'[^a-zA-Z0-9 ]', '', s.lower()) for s in sentences]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(cleaned)
    similarity = cosine_similarity(tfidf)
    scores = similarity.sum(axis=1)
    ranked = np.argsort(scores)[::-1]
    selected = sorted(ranked[:num_sentences])
    summary = " ".join([sentences[i] for i in selected])
    return summary, selected

def abstractive_summary(text, max_len=130, min_len=40):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) < 800:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(result[0]["summary_text"])
    
    combined = " ".join(summaries)
    unique = []
    for s in sent_tokenize(combined):
        if s not in unique:
            unique.append(s)
    return " ".join(unique)

def get_length_config(option):
    configs = {
        "short": {"max_len": 60, "min_len": 25, "sentences": 2},
        "medium": {"max_len": 130, "min_len": 40, "sentences": 3},
        "long": {"max_len": 200, "min_len": 80, "sentences": 5}
    }
    return configs.get(option, configs["medium"])

def highlight_sentences(text, indices):
    sentences = sent_tokenize(text)
    result = []
    for i, s in enumerate(sentences):
        if i in indices:
            result.append(f'<mark class="highlight">⭐ SELECTED SENTENCE</mark><br><br>{s}')
        else:
            result.append(s)
    return "<br><br>".join(result)

def calculate_rouge(ref, gen):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ref, gen)
        return {
            'ROUGE-1': round(scores['rouge1'].fmeasure * 100, 2),
            'ROUGE-2': round(scores['rouge2'].fmeasure * 100, 2),
            'ROUGE-L': round(scores['rougeL'].fmeasure * 100, 2)
        }
    except:
        return None

def save_results_csv(original, extractive, abstractive):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame({
        'Original': [original],
        'Extractive': [extractive],
        'Abstractive': [abstractive]
    })
    filename = f"summary_{timestamp}.csv"
    df.to_csv(filename, index=False)
    return filename

def save_results_txt(original, extractive, abstractive):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TEXT SUMMARIZER OUTPUT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write("ORIGINAL TEXT:\n")
        f.write("-"*50 + "\n")
        f.write(original + "\n\n")
        f.write("EXTRACTIVE SUMMARY:\n")
        f.write("-"*50 + "\n")
        f.write(extractive + "\n\n")
        f.write("ABSTRACTIVE SUMMARY:\n")
        f.write("-"*50 + "\n")
        f.write(abstractive + "\n")
    return filename

# ============================================
# SAMPLE TEXTS
# ============================================

SAMPLES = {
    "AI Technology": "Artificial Intelligence is transforming industries across the globe. Machine learning algorithms can now process vast amounts of data in seconds. Healthcare, finance, and transportation are being revolutionized by AI technologies. Doctors use AI to detect diseases earlier and more accurately.",
    "Climate Change": "Climate change is a pressing challenge facing our planet. Rising temperatures are causing glaciers to melt at unprecedented rates. Sea levels have risen, threatening coastal communities. Extreme weather events are becoming more frequent.",
    "Business News": "The global economy is showing signs of recovery after recent challenges. Major stock markets have rebounded with technology sectors leading growth. Companies are adopting remote work policies, changing office culture forever."
}

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 AI Text Summarizer</h1>
        <p>EXTRACTIVE | ABSTRACTIVE | COMPARE BOTH MODELS</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings Row
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown('<div class="settings-container">', unsafe_allow_html=True)
        length = st.select_slider(
            "📏 Length",
            options=["short", "medium", "long"],
            value="medium",
            format_func=lambda x: {"short": "Short", "medium": "Medium", "long": "Long"}[x]
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="settings-container">', unsafe_allow_html=True)
        with st.expander("🧠 Models"):
            st.markdown("**Extractive:** TF-IDF + Cosine Similarity")
            st.markdown("**Abstractive:** Facebook BART-large-CNN")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="settings-container">', unsafe_allow_html=True)
        sample = st.selectbox("📚 Sample", list(SAMPLES.keys()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Section
    st.markdown("""
    <div class="input-container">
        <span class="input-label">📝 INPUT TEXT</span>
    </div>
    """, unsafe_allow_html=True)
    
    input_type = st.radio("", ["✏️ Type", "📖 Sample", "📁 Upload"], horizontal=True)
    
    text = ""
    if input_type == "✏️ Type":
        text = st.text_area("", height=150, placeholder="Paste your text here...", label_visibility="collapsed")
    elif input_type == "📖 Sample":
        text = SAMPLES[sample]
        st.success("✅ Sample loaded successfully!")
    else:
        uploaded = st.file_uploader("", type=['txt', 'csv'], label_visibility="collapsed")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                text = pd.read_csv(uploaded).iloc[0, 0]
            else:
                text = uploaded.read().decode("utf-8")
            st.success("✅ File loaded successfully!")
    
    # Generate Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate = st.button("🚀 GENERATE SUMMARIES", use_container_width=True)
    
    # Professional Statistics Section
    if text and len(text.strip()) > 0:
        st.markdown("""
        <div class="stats-container">
            <div class="stats-title">📊 TEXT STATISTICS</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{len(text.split())}</h2>
                <p>Total Words</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{len(sent_tokenize(text))}</h2>
                <p>Total Sentences</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{len(text)}</h2>
                <p>Total Characters</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate Results
    if generate and text:
        config = get_length_config(length)
        
        with st.spinner("📌 Generating extractive summary..."):
            ext_summary, indices = extractive_summary(text, config["sentences"])
        
        with st.spinner("🤖 Generating abstractive summary..."):
            abs_summary = abstractive_summary(text, config["max_len"], config["min_len"])
        
        st.markdown("---")
        st.markdown('<h2 class="results-title">📋 SUMMARIZATION RESULTS</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📌 EXTRACTIVE", "🤖 ABSTRACTIVE", "✨ HIGHLIGHTED", "📊 ANALYSIS"])
        
        with tab1:
            st.markdown(f'<div class="summary-box">{ext_summary}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", f"{len(ext_summary.split())} words")
            with col2:
                comp = (1 - len(ext_summary.split()) / len(text.split())) * 100
                st.metric("Compression Ratio", f"{comp:.1f}%")
            st.caption("📌 Extractive summarization using TF-IDF + Cosine Similarity")
        
        with tab2:
            st.markdown(f'<div class="summary-box">{abs_summary}</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", f"{len(abs_summary.split())} words")
            with col2:
                comp = (1 - len(abs_summary.split()) / len(text.split())) * 100
                st.metric("Compression Ratio", f"{comp:.1f}%")
            st.caption("🤖 Abstractive summarization using Facebook BART-large-CNN")
        
        with tab3:
            highlighted = highlight_sentences(text, indices)
            st.markdown(f'<div class="highlight-box">{highlighted}</div>', unsafe_allow_html=True)
            st.caption("✨ Sentences marked with 'SELECTED' were chosen for the extractive summary")
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                ext_comp = (1 - len(ext_summary.split()) / len(text.split())) * 100
                st.metric("Extractive Compression", f"{ext_comp:.1f}%", delta="From original text")
            with col2:
                abs_comp = (1 - len(abs_summary.split()) / len(text.split())) * 100
                st.metric("Abstractive Compression", f"{abs_comp:.1f}%", delta="From original text")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            rouge = calculate_rouge(ext_summary, abs_summary)
            if rouge:
                st.markdown("### 🎯 ROUGE EVALUATION SCORES")
                st.markdown("*Comparing Abstractive vs Extractive Summary*")
                
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("ROUGE-1", f"{rouge['ROUGE-1']}%", help="Unigram overlap - Word choice similarity")
                with r2:
                    st.metric("ROUGE-2", f"{rouge['ROUGE-2']}%", help="Bigram overlap - Phrase similarity")
                with r3:
                    st.metric("ROUGE-L", f"{rouge['ROUGE-L']}%", help="Longest common sequence - Structure similarity")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download Buttons
            st.markdown("### 💾 EXPORT RESULTS")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                csv_file = save_results_csv(text, ext_summary, abs_summary)
                with open(csv_file, 'rb') as f:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=f,
                        file_name=csv_file,
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_d2:
                txt_file = save_results_txt(text, ext_summary, abs_summary)
                with open(txt_file, 'rb') as f:
                    st.download_button(
                        label="📄 Download as TXT",
                        data=f,
                        file_name=txt_file,
                        mime="text/plain",
                        use_container_width=True
                    )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="font-size: 0.9rem;">✨ <strong>AI Text Summarizer</strong> | Extractive (TF-IDF) vs Abstractive (BART) | Compare Both Models</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">✅ Length Control • Sentence Highlighting • ROUGE Scores • CSV/TXT Export</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()