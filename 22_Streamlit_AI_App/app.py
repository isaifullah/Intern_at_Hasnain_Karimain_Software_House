# ============================================================
# AI MODEL DEPLOYMENT DASHBOARD USING STREAMLIT
# Final Professional UI with Interactive Plotly Analytics
# ============================================================

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Model Deployment Dashboard",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD PROJECT FILES
# ============================================================

@st.cache_resource
def load_project_files():
    small_model = joblib.load("models/small_model.pkl")
    large_model = joblib.load("models/large_model.pkl")
    target_names = joblib.load("models/target_names.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    model_results = joblib.load("models/model_results.pkl")

    return small_model, large_model, target_names, feature_names, model_results


small_model, large_model, target_names, feature_names, model_results = load_project_files()


# ============================================================
# CUSTOM PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(14,165,233,0.18), transparent 28%),
            radial-gradient(circle at 90% 5%, rgba(168,85,247,0.20), transparent 32%),
            linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 36px 40px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.90));
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 24px 80px rgba(0,0,0,0.45);
        margin-bottom: 26px;
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 900;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 17px;
        line-height: 1.7;
        max-width: 880px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 27px;
        font-weight: 850;
        color: #f8fafc;
        margin-top: 14px;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .model-info {
        padding: 15px 18px;
        border-radius: 16px;
        background: rgba(37,99,235,0.15);
        border: 1px solid rgba(96,165,250,0.35);
        color: #bfdbfe;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 24px;
    }

    .result-box {
        padding: 24px;
        border-radius: 18px;
        background: linear-gradient(135deg, #047857, #065f46);
        border: 1px solid rgba(52,211,153,0.55);
        box-shadow: 0 16px 40px rgba(16,185,129,0.22);
        margin-top: 22px;
        margin-bottom: 22px;
    }

    .result-box h3 {
        color: #ecfdf5;
        margin: 0;
        font-size: 28px;
        font-weight: 900;
    }

    .warning-box {
        padding: 16px 18px;
        border-radius: 14px;
        background: rgba(120,53,15,0.90);
        color: #fef3c7;
        border: 1px solid rgba(245,158,11,0.55);
        font-weight: 650;
        margin-top: 15px;
    }

    label {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        font-size: 16px;
        font-weight: 850;
    }

    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(90deg, #16a34a, #22c55e);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        font-size: 16px;
        font-weight: 850;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid rgba(148,163,184,0.18);
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 14px 14px 0 0;
        padding: 12px 24px;
        color: #cbd5e1;
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.68);
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 16px;
        padding: 14px 18px;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8;
        font-weight: 900;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    .divider {
        height: 1px;
        background: rgba(148,163,184,0.18);
        margin: 32px 0 24px 0;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 24px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_selected_model(model_name):
    if model_name == "Small Model - Logistic Regression":
        return (
            small_model,
            "Small Logistic Regression Model",
            "Small model selected: lightweight, fast, and suitable for low-memory systems."
        )

    return (
        large_model,
        "Large Random Forest Model",
        "Large model selected: stronger performance and better prediction stability."
    )


def predict_single_record(model, input_data, target_names):
    prediction = model.predict(input_data)[0]
    predicted_class = target_names[prediction]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

    return predicted_class, probabilities


def predict_batch_records(model, dataframe, feature_names, target_names):
    predictions = model.predict(dataframe[feature_names])

    output_df = dataframe.copy()
    output_df["Predicted Class"] = [target_names[p] for p in predictions]

    return output_df


def show_prediction_gauge(probabilities):
    confidence = max(probabilities) * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Prediction Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"}
            }
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_feature_radar_chart(feature_names, input_values):
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=input_values,
            theta=feature_names,
            fill="toself",
            name="Input Features"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        template="plotly_dark",
        height=350,
        title="Input Feature Profile",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_probability_chart(target_names, probabilities):
    probability_df = pd.DataFrame({
        "Class": target_names,
        "Probability": probabilities
    })

    fig = px.bar(
        probability_df,
        x="Class",
        y="Probability",
        text="Probability",
        title="Class Probability Distribution"
    )

    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")

    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_batch_analytics(result_df):
    distribution = result_df["Predicted Class"].value_counts().reset_index()
    distribution.columns = ["Class", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            distribution,
            names="Class",
            values="Count",
            hole=0.45,
            title="Prediction Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            distribution,
            x="Class",
            y="Count",
            text="Count",
            title="Class Count Summary"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)


def show_accuracy_dashboard(model_results):
    accuracy_df = pd.DataFrame({
        "Model": list(model_results.keys()),
        "Accuracy": list(model_results.values())
    })

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Model Accuracy Comparison", "Accuracy Share")
    )

    fig.add_trace(
        go.Bar(
            x=accuracy_df["Model"],
            y=accuracy_df["Accuracy"],
            text=[f"{acc * 100:.2f}%" for acc in accuracy_df["Accuracy"]],
            textposition="outside",
            name="Accuracy"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Pie(
            labels=accuracy_df["Model"],
            values=accuracy_df["Accuracy"],
            hole=0.45,
            name="Accuracy Share"
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    fig.update_yaxes(range=[0, 1], row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI Model Deployment Dashboard</h1>
        <p>
            A professional Streamlit dashboard for deploying machine learning models.
            Choose a model, run manual predictions, upload CSV files for batch predictions,
            and explore interactive analytics in one clean interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL SELECTION
# ============================================================

model_col, acc_col = st.columns([3, 1])

with model_col:
    model_name = st.selectbox(
        "Choose Model",
        [
            "Small Model - Logistic Regression",
            "Large Model - Random Forest"
        ]
    )

selected_model, selected_model_key, model_description = get_selected_model(model_name)
selected_accuracy = model_results.get(selected_model_key, 0)

with acc_col:
    st.metric("Accuracy", f"{selected_accuracy * 100:.2f}%")

st.markdown(
    f"""
    <div class="model-info">
        {model_description}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN SECTIONS
# ============================================================

manual_tab, batch_tab = st.tabs(["🌸 Manual Prediction", "📁 Batch Prediction"])


# ============================================================
# MANUAL PREDICTION
# ============================================================

with manual_tab:
    st.markdown('<div class="section-title">🌸 Manual Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter values and generate a real-time prediction with interactive confidence analytics.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1, 0.1)
        petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4, 0.1)

    with col2:
        sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.5, 0.1)
        petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2, 0.1)

    if st.button("Predict Flower Class"):
        input_data = np.array([
            [sepal_length, sepal_width, petal_length, petal_width]
        ])

        predicted_class, probabilities = predict_single_record(
            selected_model,
            input_data,
            target_names
        )

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Predicted Class: {predicted_class.upper()}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if probabilities is not None:
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                show_prediction_gauge(probabilities)

            with chart_col2:
                show_feature_radar_chart(
                    feature_names,
                    [sepal_length, sepal_width, petal_length, petal_width]
                )

            show_probability_chart(target_names, probabilities)


# ============================================================
# BATCH PREDICTION
# ============================================================

with batch_tab:
    st.markdown('<div class="section-title">📁 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload a CSV file and generate predictions with visual class distribution analytics.</div>',
        unsafe_allow_html=True
    )

    st.write("Required CSV columns:")
    st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        st.markdown("#### Uploaded Data Preview")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        missing_columns = [
            col for col in feature_names
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.markdown(
                f"""
                <div class="warning-box">
                    Missing required columns: {missing_columns}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            if st.button("Run Batch Prediction"):
                result_df = predict_batch_records(
                    selected_model,
                    uploaded_df,
                    feature_names,
                    target_names
                )

                st.markdown("#### Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                show_batch_analytics(result_df)

                csv_data = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Interactive comparison between the lightweight and advanced models.</div>',
    unsafe_allow_html=True
)

show_accuracy_dashboard(model_results)

performance_df = pd.DataFrame({
    "Model": list(model_results.keys()),
    "Accuracy": [f"{acc * 100:.2f}%" for acc in model_results.values()]
})

st.dataframe(performance_df, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Python, Scikit-learn, Joblib, Streamlit, and Plotly
    </div>
    """,
    unsafe_allow_html=True
)