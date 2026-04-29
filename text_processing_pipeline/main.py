import streamlit as st
import pandas as pd


from nlp_pipeline import (
    text_preprocessing_pipeline,
    preprocess_batch,
    compare_stem_vs_lemma
)

st.set_page_config(
    page_title="NLP Text Processing Pipeline",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NLP Text Processing Pipeline")
st.markdown("Clean, tokenize, and preprocess text for Machine Learning")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Functionality",
    [
        "Single Text Processing",
        "Compare Raw vs Processed",
        "Stem vs Lemma",
        "Batch CSV Processing"
    ]
)

# -------------------------
# 1. Single Text Processing
# -------------------------
if option == "Single Text Processing":
    st.subheader("🔹 Single Text Processing")

    text = st.text_area("Enter text")

    if st.button("Process"):
        if text.strip():
            result = text_preprocessing_pipeline(text)
            st.success("Processed Text")
            st.code(result)
        else:
            st.warning("Please enter text")

# -------------------------
# 2. Raw vs Processed
# -------------------------
elif option == "Compare Raw vs Processed":
    st.subheader("🔹 Comparison")

    text = st.text_area("Enter text")

    if st.button("Compare"):
        if text.strip():
            processed = text_preprocessing_pipeline(text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Raw Text")
                st.write(text)

            with col2:
                st.markdown("### Processed Text")
                st.code(processed)
        else:
            st.warning("Please enter text")

# -------------------------
# 3. Stem vs Lemma
# -------------------------
elif option == "Stem vs Lemma":
    st.subheader("🔹 Stemming vs Lemmatization")

    text = st.text_area("Enter text")

    if st.button("Compare Methods"):
        if text.strip():
            result = compare_stem_vs_lemma(text)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Stemmed")
                st.code(result["stemmed"])

            with col2:
                st.markdown("### Lemmatized")
                st.code(result["lemmatized"])
        else:
            st.warning("Please enter text")

# -------------------------
# 4. Batch CSV Processing
# -------------------------
elif option == "Batch CSV Processing":
    st.subheader("🔹 Batch CSV Processing")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        column = st.selectbox("Select text column", df.columns)

        # ✅ FORCE NEW UI STATE
        method = st.radio("Choose Processing Method", ["lemma", "stem"])

        st.write("Selected method:", method)  # 🔥 DEBUG LINE

        if st.button("Process Dataset"):
            with st.spinner("Processing..."):
                df["processed_text"] = preprocess_batch(df, column, method)

            st.success("Done!")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Processed CSV",
                data=csv,
                file_name="processed_output.csv",
                mime="text/csv"
            )