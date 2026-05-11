import streamlit as st
import pandas as pd
import time

# Import from nlp_pipeline.py
from nlp_pipeline import (
    text_preprocessing_pipeline,
    preprocess_batch,
    compare_stem_vs_lemma,
    test_edge_cases
)

st.set_page_config(
    page_title="NLP Text Processing Pipeline",
    page_icon="🧠",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Tab ribbon background */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 0.5rem;
        border-radius: 12px;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Center button */
    .center-button {
        display: flex;
        justify-content: center;
        width: 100%;
        margin: 0.5rem 0;
    }
    
    /* Output card styling */
    .output-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    .output-card pre {
        background: #2d2d2d;
        color: #00ff9d;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 250px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    
    /* PROFESSIONAL STATISTICS CARDS - WIDE, LARGE TEXT, PERFECT FIT */
    .stat-card-professional {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: center;
        min-width: 120px;
        width: 100%;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
    }
    
    .stat-card-professional:hover {
        transform: translateY(-3px);
    }
    
    .stat-card-professional h4 {
        color: rgba(255,255,255,0.9);
        font-size: 0.75rem;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .stat-card-professional .value {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    
    .stat-card-professional .unit {
        color: rgba(255,255,255,0.8);
        font-size: 0.7rem;
        margin-top: 0.3rem;
    }
    
    /* Raw text card */
    .raw-text-card {
        background: #2d2d44;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        color: #ffffff;
    }
    
    .processed-text-card {
        background: #1e2d2d;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff9d;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2d2d44;
        border-radius: 8px;
    }
    
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.75rem 0 0.5rem 0;
        color: #ccc;
    }
    
    /* Column spacing */
    .stColumn {
        padding: 0 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 NLP Text Preprocessing Pipeline</h1>
    <p>Clean, tokenize, and preprocess text for Machine Learning with edge case handling</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs([
    "📝 Single Text",
    "📊 Raw vs Processed",
    "⚖️ Stem vs Lemma",
    "📁 Batch Processing"
])

# ==================== TAB 1: Single Text Processing ====================
with tabs[0]:
    st.markdown("### 📝 Single Text Processing")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Input Text",
            height=120,
            placeholder="Example: I'm loving the new AI tools! They are running amazingly well.",
            key="single_text"
        )
        
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        process_clicked = st.button("🚀 Process Text", key="process_single", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>🔧 Pipeline Steps:</strong><br>
            1️⃣ Edge Case Handling<br>
            2️⃣ Contraction Expansion<br>
            3️⃣ Text Cleaning<br>
            4️⃣ Custom Tokenization<br>
            5️⃣ Stopword Removal<br>
            6️⃣ Lemmatization
        </div>
        """, unsafe_allow_html=True)
    
    if process_clicked:
        if text_input.strip():
            with st.spinner("Processing..."):
                result = text_preprocessing_pipeline(text_input)
            
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            
            original_words = len(text_input.split())
            processed_tokens = len(result)
            reduction = original_words - processed_tokens if original_words > processed_tokens else 0
            reduction_pct = (reduction / original_words * 100) if original_words > 0 else 0
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.markdown(f"""
                <div class="stat-card-professional">
                    <h4>📝 ORIGINAL</h4>
                    <div class="value">{original_words}</div>
                    <div class="unit">words</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="stat-card-professional">
                    <h4>✨ PROCESSED</h4>
                    <div class="value">{processed_tokens}</div>
                    <div class="unit">tokens</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                <div class="stat-card-professional">
                    <h4>🗑️ REMOVED</h4>
                    <div class="value">{reduction}</div>
                    <div class="unit">tokens</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_d:
                st.markdown(f"""
                <div class="stat-card-professional">
                    <h4>📉 REDUCTION</h4>
                    <div class="value">{reduction_pct:.1f}%</div>
                    <div class="unit">percentage</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📤 Processed Output")
            st.markdown(f"""
            <div class="output-card">
                <pre>{result}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 View as String"):
                st.code(" ".join(result), language="text")
        else:
            st.warning("⚠️ Please enter some text to process.")

# ==================== TAB 2: Raw vs Processed ====================
with tabs[1]:
    st.markdown("### 📊 Compare Raw vs Processed Text")
    
    compare_text = st.text_area(
        "Enter Text to Compare",
        height=120,
        placeholder="Enter any text to see before/after comparison...",
        key="compare_text"
    )
    
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    compare_clicked = st.button("🔄 Compare", key="compare_btn", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if compare_clicked:
        if compare_text.strip():
            with st.spinner("Processing..."):
                processed = text_preprocessing_pipeline(compare_text)
            
            st.markdown("---")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 📄 Raw Text")
                st.markdown(f"""
                <div class="raw-text-card">
                    {compare_text}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<p class="section-header">📊 RAW STATISTICS</p>', unsafe_allow_html=True)
                
                raw_chars = len(compare_text)
                raw_words = len(compare_text.split())
                raw_lines = compare_text.count('\n') + 1
                
                col_r1, col_r2, col_r3 = st.columns(3)
                
                with col_r1:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>📝 CHARACTERS</h4>
                        <div class="value">{raw_chars}</div>
                        <div class="unit">total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_r2:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>📊 WORDS</h4>
                        <div class="value">{raw_words}</div>
                        <div class="unit">total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_r3:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>📏 LINES</h4>
                        <div class="value">{raw_lines}</div>
                        <div class="unit">total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 View Raw as String"):
                    st.code(compare_text, language="text")
            
            with col_right:
                st.markdown("### ✨ Processed Text")
                st.markdown(f"""
                <div class="processed-text-card">
                    <pre style="background:#1a1a1a; color:#00ff9d; padding:0.5rem; border-radius:8px; overflow-x:auto;">{processed}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<p class="section-header">📊 PROCESSED STATISTICS</p>', unsafe_allow_html=True)
                
                proc_tokens = len(processed)
                proc_string_len = len(' '.join(processed))
                reduction_pct = ((raw_words - proc_tokens) / raw_words * 100) if raw_words > 0 else 0
                
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>🎯 TOKENS</h4>
                        <div class="value">{proc_tokens}</div>
                        <div class="unit">total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_p2:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>🔤 STRING LEN</h4>
                        <div class="value">{proc_string_len}</div>
                        <div class="unit">characters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_p3:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>📉 REDUCTION</h4>
                        <div class="value">{reduction_pct:.1f}%</div>
                        <div class="unit">saved</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 View Processed as String"):
                    st.code(" ".join(processed), language="text")
        else:
            st.warning("⚠️ Please enter text to compare.")

# ==================== TAB 3: Stem vs Lemma ====================
with tabs[2]:
    st.markdown("### ⚖️ Stemming vs Lemmatization Comparison")
    
    stem_text = st.text_area(
        "Enter Text for Comparison",
        height=100,
        placeholder="Example: The children were running quickly to the stores",
        key="stem_text"
    )
    
    st.info("💡 **Tip:** For faster processing, keep text under 500 characters")
    
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    compare_methods = st.button("⚖️ Compare Methods", key="stem_compare", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if compare_methods:
        if stem_text.strip():
            if len(stem_text) > 1000:
                st.warning("⚠️ Text is long. This may take 10-15 seconds.")
            
            with st.spinner("Processing comparison..."):
                start_time = time.time()
                
                from nlp_pipeline import preprocess_edge_cases, clean_text, expand_contractions, custom_tokenizer, remove_stopwords, stemmer
                
                text_processed = preprocess_edge_cases(stem_text)
                
                # Stemming
                text_clean = clean_text(expand_contractions(text_processed))
                tokens = custom_tokenizer(text_clean)
                tokens = remove_stopwords(tokens)
                stemmed = [stemmer.stem(t) for t in tokens]
                
                # Lemmatization
                lemmatized = text_preprocessing_pipeline(stem_text)
                
                elapsed_time = time.time() - start_time
            
            st.markdown("---")
            st.success(f"✅ Comparison completed in {elapsed_time:.2f} seconds")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 🔪 Stemming")
                st.markdown("*Aggressive, rule-based reduction*")
                st.markdown(f"""
                <div class="output-card">
                    <pre>{stemmed}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<p class="section-header">📊 STEMMING STATISTICS</p>', unsafe_allow_html=True)
                
                stem_tokens = len(stemmed)
                stem_unique = len(set(stemmed))
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>🎯 TOTAL TOKENS</h4>
                        <div class="value">{stem_tokens}</div>
                        <div class="unit">tokens</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_s2:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>🔄 UNIQUE TOKENS</h4>
                        <div class="value">{stem_unique}</div>
                        <div class="unit">tokens</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 View Stemmed as String"):
                    st.code(" ".join(stemmed), language="text")
            
            with col_right:
                st.markdown("### 🌿 Lemmatization")
                st.markdown("*Context-aware dictionary form*")
                st.markdown(f"""
                <div class="output-card">
                    <pre>{lemmatized}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<p class="section-header">📊 LEMMATIZATION STATISTICS</p>', unsafe_allow_html=True)
                
                lemma_tokens = len(lemmatized)
                lemma_unique = len(set(lemmatized))
                
                col_l1, col_l2 = st.columns(2)
                
                with col_l1:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>🎯 TOTAL TOKENS</h4>
                        <div class="value">{lemma_tokens}</div>
                        <div class="unit">tokens</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_l2:
                    st.markdown(f"""
                    <div class="stat-card-professional">
                        <h4>🔄 UNIQUE TOKENS</h4>
                        <div class="value">{lemma_unique}</div>
                        <div class="unit">tokens</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 View Lemmatized as String"):
                    st.code(" ".join(lemmatized), language="text")
            
            with st.expander("🔍 View Differences"):
                if stemmed != lemmatized:
                    diff_data = []
                    max_len = max(len(stemmed), len(lemmatized))
                    
                    for i in range(max_len):
                        stem = stemmed[i] if i < len(stemmed) else "(none)"
                        lemma = lemmatized[i] if i < len(lemmatized) else "(none)"
                        if stem != lemma:
                            diff_data.append({
                                "Position": i + 1, 
                                "Stemmed": stem, 
                                "Lemmatized": lemma
                            })
                    
                    if diff_data:
                        st.dataframe(pd.DataFrame(diff_data), use_container_width=True)
                    else:
                        st.info("✅ No differences found in the token list!")
                else:
                    st.info("✅ No differences found! Both methods produced the same result.")
        else:
            st.warning("⚠️ Please enter text to compare.")

# ==================== TAB 4: Batch Processing ====================
with tabs[3]:
    st.markdown("### 📁 Batch CSV Processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            text_column = st.selectbox("Select text column", df.columns, key="text_col")
        
        with col2:
            method = st.radio("Processing Method", ["lemma", "stem"], horizontal=True, key="method")
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            process_batch = st.button("🚀 Process Dataset", key="process_batch", use_container_width=True)
        
        if process_batch:
            total_rows = len(df)
            
            if total_rows > 100:
                st.warning(f"⚠️ Processing {total_rows} rows may take several minutes.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            from nlp_pipeline import preprocess_edge_cases, clean_text, expand_contractions, custom_tokenizer, remove_stopwords, stemmer
            
            for idx, row in df.iterrows():
                status_text.text(f"Processing row {idx + 1}/{total_rows}...")
                
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                
                if method == "stem":
                    if text.strip():
                        text_processed = preprocess_edge_cases(text)
                        if text_processed.strip():
                            text_clean = clean_text(expand_contractions(text_processed))
                            tokens = custom_tokenizer(text_clean)
                            tokens = remove_stopwords(tokens)
                            processed = [stemmer.stem(t) for t in tokens]
                        else:
                            processed = []
                    else:
                        processed = []
                else:
                    if text.strip():
                        processed = text_preprocessing_pipeline(text)
                    else:
                        processed = []
                
                results.append(" ".join(processed) if processed else "")
                progress_bar.progress((idx + 1) / total_rows)
            
            df["processed_text"] = results
            status_text.text("✅ Processing complete!")
            
            st.markdown("### ✅ Processing Complete")
            st.dataframe(df.head(), use_container_width=True)
            
            with st.expander("🔍 Preview Processed Results"):
                preview_df = df[[text_column, "processed_text"]].head(10)
                st.dataframe(preview_df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Processed CSV",
                data=csv,
                file_name="processed_output.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            non_empty = df["processed_text"].str.len() > 0
            st.success(f"✅ Processed {total_rows} rows | {non_empty.sum()} non-empty outputs")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>NLP Text Preprocessing Pipeline | Production-ready with edge case handling</p>",
    unsafe_allow_html=True
)