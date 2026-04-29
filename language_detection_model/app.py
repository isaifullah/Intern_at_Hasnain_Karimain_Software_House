import streamlit as st
import pickle
import numpy as np
import re
import unicodedata
from langdetect import detect

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("language_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------------- PREPROCESS ----------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F]', '', text)

    return text

# ---------------- PREDICT ----------------
def predict_language(text):
    text = preprocess_text(text)

    if len(text.strip()) < 2:
        return "Unknown / Too Short Text", {}

    proba = model.predict_proba([text])[0]
    classes = model.classes_

    idx = np.argmax(proba)
    confidence = proba[idx]

    language = label_encoder.inverse_transform([classes[idx]])[0]

    return f"{language} ({round(confidence,2)})", proba

# ---------------- TOP-2 ----------------
def predict_top2(text):
    text = preprocess_text(text)

    proba = model.predict_proba([text])[0]
    classes = model.classes_

    top2 = np.argsort(proba)[-2:][::-1]

    return [
        (label_encoder.inverse_transform([classes[i]])[0], round(proba[i], 3))
        for i in top2
    ]

# ---------------- UI ----------------
st.set_page_config(page_title="Language Detection", layout="centered")

st.title("🌍 Multilingual Language Detection App")
st.write("Detect language from text using Machine Learning + NLP")

text = st.text_area("Enter your text here")

if st.button("Predict Language"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result, _ = predict_language(text)
        st.success(f"Predicted Language: {result}")

# ---------------- TOP 2 ----------------
if st.button("Show Top-2 Predictions"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        results = predict_top2(text)

        st.write("Top 2 Predictions:")
        for lang, prob in results:
            st.write(f"{lang} → {prob}")

# ---------------- LANGDETECT COMPARISON ----------------
if st.button("Compare with LangDetect"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        try:
            ld = detect(text)
        except:
            ld = "Error"

        ml_result, _ = predict_language(text)

        st.write("Model Prediction:", ml_result)
        st.write("LangDetect:", ld)