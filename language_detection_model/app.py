import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import unicodedata
from langdetect import detect

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("language_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# =========================
# PREPROCESS
# =========================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# CORE FUNCTIONS
# =========================
def predict(text):
    processed = preprocess_text(text)
    return model.predict([processed])[0]


def predict_top2(text):
    processed = preprocess_text(text)

    if hasattr(model.named_steps["model"], "predict_proba"):
        probs = model.predict_proba([processed])[0]
        classes = model.named_steps["model"].classes_

        top2_idx = np.argsort(probs)[-2:][::-1]

        return [
            (classes[top2_idx[0]], probs[top2_idx[0]]),
            (classes[top2_idx[1]], probs[top2_idx[1]])
        ]
    return None


def compare(text):
    my_pred = predict(text)

    try:
        ld = detect(text)
    except:
        ld = "Error"

    return my_pred, ld


def batch_predict(df):
    results = []

    for text in df["Text"]:
        pred = predict(text)

        top2 = predict_top2(text)
        if top2:
            top1, p1 = top2[0]
            top2_lang, p2 = top2[1]
        else:
            top1, p1, top2_lang, p2 = None, None, None, None

        try:
            ld = detect(text)
        except:
            ld = "Error"

        results.append([pred, p1, top2_lang, p2, ld])

    df[["Prediction", "Top1_Conf", "Second_Lang", "Second_Conf", "Langdetect"]] = results
    return df

# =========================
# STREAMLIT UI (CLI STYLE)
# =========================
st.title("🌍 Language Detection model")
st.write("Use commands like: predict, top2, compare, batch")

command = st.text_input("Enter command:")

# =========================
# SINGLE TEXT INPUT
# =========================
if command in ["predict", "top2", "compare"]:

    text = st.text_area("Enter text")

    if st.button("Run"):

        if command == "predict":
            st.success(f"Predicted Language: {predict(text)}")

        elif command == "top2":
            result = predict_top2(text)
            if result:
                st.subheader("Top 2 Predictions")
                for lang, prob in result:
                    st.write(f"{lang}: {prob:.4f}")

        elif command == "compare":
            my, ld = compare(text)
            st.write("Your Model:", my)
            st.write("Langdetect:", ld)

# =========================
# BATCH MODE
# =========================
elif command == "batch":

    file = st.file_uploader("Upload CSV (must contain 'Text' column)", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        if "Text" not in df.columns:
            st.error("CSV must contain 'Text' column")
        else:
            if st.button("Run Batch Prediction"):
                result_df = batch_predict(df)
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )

# =========================
# HELP MENU
# =========================
elif command == "help" or command == "":
    st.info("""
    📌 Available Commands:

    - predict → single prediction
    - top2 → top 2 languages
    - compare → compare with langdetect
    - batch → CSV batch prediction

    Example:
    Enter: predict
    """)