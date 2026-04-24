# ================= IMPORT LIBRARIES =================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    auc
)

from sklearn.preprocessing import LabelEncoder, label_binarize

from utils import preprocess_text

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
st.title("📊 Model Evaluation Dashboard")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

# =========================================================
# MAIN APP
# =========================================================
if uploaded_file is not None:

    # ================= LOAD DATA =================
    df = pd.read_csv(uploaded_file)

    st.write("### 📂 Dataset Preview")
    st.dataframe(df.head())

    # ================= DATASET ANALYSIS REPORT =================
    st.write("### 📊 Dataset Analysis Report")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())

    # Detailed breakdown
    st.write("#### Missing Values Per Column")
    st.dataframe(df.isnull().sum())

    st.write("#### Duplicate Rows")
    st.write(df[df.duplicated()])

    # ================= SMART COLUMN DETECTION =================
    st.write("### ⚙️ Column Selection")

    cols = df.columns

    text_keywords = ["text", "message", "content", "email", "review", "comment"]
    label_keywords = ["label", "class", "target", "category", "sentiment", "spam"]

    possible_text_cols = [c for c in cols if any(k in c.lower() for k in text_keywords)]
    possible_label_cols = [c for c in cols if any(k in c.lower() for k in label_keywords)]

    if not possible_text_cols:
        possible_text_cols = list(cols)

    if not possible_label_cols:
        possible_label_cols = list(cols)

    text_col = st.selectbox("Select TEXT column", possible_text_cols)
    label_col = st.selectbox("Select LABEL column", possible_label_cols)

    if text_col == label_col:
        st.error("Text and Label columns must be different")
        st.stop()

    ignored_cols = [c for c in cols if c not in [text_col, label_col]]

    st.info(f"Ignored Columns: {ignored_cols}")

    # ================= PREPROCESS =================
    df[text_col] = df[text_col].astype(str)
    df[text_col] = df[text_col].apply(preprocess_text)

    X = df[text_col]
    y = df[label_col]

    # ================= LABEL ENCODING =================
    le = LabelEncoder()
    y = le.fit_transform(y)

    # ================= CLASS DISTRIBUTION =================
    st.write("### 📊 Class Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)

    # ================= TRAIN TEST SPLIT =================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ================= TF-IDF =================
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ================= MODELS =================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Naive Bayes": MultinomialNB()
    }

    results = []
    preds_dict = {}

    # ================= TRAIN & EVALUATE =================
    for name, model in models.items():

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

    results_df = pd.DataFrame(results, columns=[
        "Model", "Accuracy", "Precision", "Recall", "F1-score"
    ])

    # ================= TABLE =================
    st.subheader("📋 Model Comparison")
    st.dataframe(results_df)

    # ================= BAR CHART =================
    st.subheader("📊 Metrics Comparison")

    fig, ax = plt.subplots()
    results_df.set_index("Model").plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # ================= MODEL ANALYSIS =================
    st.subheader("🔍 Model Analysis")

    model_name = st.selectbox("Select Model", results_df["Model"])
    selected = results_df[results_df["Model"] == model_name]

    st.write(selected)

    # ================= CONFUSION MATRIX =================
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_test, preds_dict[model_name])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ================= ROC CURVE =================
    st.subheader("📈 ROC Curve")

    model = models[model_name]
    num_classes = len(np.unique(y_test))

    if num_classes == 2:

        probs = model.predict_proba(X_test_vec)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend()
        st.pyplot(fig)

    else:

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        probs = model.predict_proba(X_test_vec)

        fig, ax = plt.subplots()

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend()
        st.pyplot(fig)

    # ================= SIDE BY SIDE =================
    st.subheader("⚖ Compare Models Side-by-Side")

    m1 = st.selectbox("Model 1", results_df["Model"], key="m1")
    m2 = st.selectbox("Model 2", results_df["Model"], key="m2")

    col1, col2 = st.columns(2)

    with col1:
        st.write(results_df[results_df["Model"] == m1])

    with col2:
        st.write(results_df[results_df["Model"] == m2])

else:
    st.info("📂 Upload a dataset to start analysis.")