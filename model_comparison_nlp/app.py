# =========================
# 📊 Sentiment Analysis Streamlit App
# =========================

import streamlit as st
import pickle
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# -------------------------
# Download NLTK resources
# -------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# -------------------------
# Load saved models
# -------------------------
models = pickle.load(open("all_models.pkl", "rb"))
best_model = pickle.load(open("best_model.pkl", "rb"))
best_model_name = pickle.load(open("best_model_name.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# -------------------------
# Text preprocessing (same as training)
# -------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\$\s?\d+(\.\d+)?', ' money ', text)
    text = re.sub(r'\d+(\.\d+)?', ' number ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = word_tokenize(text)
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

    cleaned = []
    for word, tag in pos_tags:
        if word not in stop_words and len(word) > 2:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            cleaned.append(lemma)

    return " ".join(cleaned)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("📊 Sentiment Analysis Model Comparison")
st.write("Enter text and compare predictions from multiple ML models.")

user_input = st.text_area("✍️ Enter your text here:")

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # preprocess
        cleaned_text = preprocess_text(user_input)
        vector = vectorizer.transform([cleaned_text]).toarray()

        st.subheader("🔍 Model Predictions:")

        # predictions from all models
        for name, model in models.items():
            pred = model.predict(vector)[0]
            st.write(f"**{name}:** {pred}")

        # best model prediction
        best_pred = best_model.predict(vector)[0]

        st.success("🏆 Final Result")
        st.write(f"**Best Model Used:** {best_model_name}")
        st.write(f"**Prediction:** {best_pred}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | NLP Sentiment Classification Project")