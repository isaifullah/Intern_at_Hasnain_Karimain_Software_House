import streamlit as st
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

nltk.download("punkt")

# Load model
summarizer = pipeline("summarization", model="t5-small")



# preprocess text: lowercase, remove special chars, split into sentences

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9. ]', '', text)
    sentences = sent_tokenize(text)
    return text, sentences


# extractive summary: score sentences using TF-IDF and select top N sentences

def extractive_summary(text, num_sentences=2):

    _, sentences = preprocess_text(text)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)

    scores = np.array(tfidf.sum(axis=1)).flatten()
    ranked = np.argsort(scores)[::-1]

    selected = sorted(ranked[:num_sentences])

    summary = " ".join([sentences[i] for i in selected])

    return summary, selected


# abstractive summary: use transformer model to generate summary from text

def abstractive_summary(text, max_len=80, min_len=30):

    text = text[:1000]

    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return result[0]["summary_text"]


# streamlit app, UI for input text, summary length, display results and deploy on Streamlit Cloud

st.title("🧠 Text Summarizer (Extractive + Abstractive)")

text = st.text_area("Enter text here", height=250)

option = st.selectbox("Summary length", ["short", "medium", "long"])

length_map = {
    "short": (40, 15),
    "medium": (80, 30),
    "long": (120, 50)
}

max_len, min_len = length_map[option]

num_sentences = st.slider("Extractive sentences", 1, 5, 2)

if st.button("Generate Summary"):

    if text.strip() == "":
        st.warning("Please enter text")
    else:

        ext_summary, selected_idx = extractive_summary(text, num_sentences)
        abs_summary = abstractive_summary(text, max_len, min_len)

        st.subheader("Original Text")
        st.write(text)

        st.subheader("Extractive Summary")
        st.success(ext_summary)

        st.subheader("Abstractive Summary")
        st.info(abs_summary)

        st.subheader("Comparison")
        st.write("Extractive → factual, sentence-based")
        st.write("Abstractive → human-like, generated")