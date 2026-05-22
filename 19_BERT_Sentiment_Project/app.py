import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.24), transparent 32%),
        radial-gradient(circle at top right, rgba(20, 184, 166, 0.18), transparent 30%),
        linear-gradient(135deg, #020617 0%, #08111f 55%, #0f172a 100%);
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, span {
    color: #cbd5e1;
}

.hero {
    text-align: center;
    padding: 30px 20px 20px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 999px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #7dd3fc;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 52px;
    font-weight: 950;
    letter-spacing: -1.6px;
    background: linear-gradient(90deg, #ffffff, #93c5fd, #5eead4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 760px;
    margin: 12px auto 0;
    color: #94a3b8;
    font-size: 18px;
    line-height: 1.7;
}

.info-strip {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 24px;
}

.info-pill {
    padding: 9px 16px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.24);
    color: #dbeafe;
    font-size: 14px;
    font-weight: 700;
}

.stTextArea textarea {
    background: rgba(2, 6, 23, 0.92) !important;
    color: #f8fafc !important;
    border-radius: 18px !important;
    border: 1px solid rgba(148, 163, 184, 0.32) !important;
    padding: 18px !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.16) !important;
}

.stButton button {
    background: linear-gradient(135deg, #2563eb, #0891b2, #0d9488);
    color: white;
    border: none;
    border-radius: 16px;
    height: 52px;
    font-weight: 800;
    transition: all 0.25s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 34px rgba(8, 145, 178, 0.35);
}

[data-testid="stMetric"] {
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 900;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

.status-card {
    padding: 18px 20px;
    border-radius: 18px;
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.16);
    margin-bottom: 14px;
}

.status-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
}

.status-value {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 900;
}

.result-positive, .result-negative, .result-neutral {
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    margin-top: 10px;
}

.result-positive {
    background: linear-gradient(135deg, #064e3b, #059669);
    border: 1px solid rgba(110, 231, 183, 0.35);
}

.result-negative {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    border: 1px solid rgba(252, 165, 165, 0.35);
}

.result-neutral {
    background: linear-gradient(135deg, #164e63, #0284c7);
    border: 1px solid rgba(125, 211, 252, 0.35);
}

.result-label {
    font-size: 34px;
    font-weight: 950;
    color: white;
}

.result-confidence {
    font-size: 18px;
    color: rgba(255,255,255,0.88);
    font-weight: 700;
}

.footer {
    text-align: center;
    padding: 30px 0 5px;
    color: #64748b;
    font-size: 14px;
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = "models/distilbert_sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    predicted_id = torch.argmax(probabilities).item()

    sentiment = model.config.id2label[predicted_id]
    confidence = probabilities[predicted_id].item() * 100

    scores = {
        model.config.id2label[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(probabilities))
    }

    return sentiment, confidence, scores


if "input_text" not in st.session_state:
    st.session_state.input_text = ""


st.markdown(f"""
<div class="hero">
    <div class="hero-badge">Transformer-Powered NLP System</div>
    <div class="hero-title">Twitter Sentiment Analysis</div>
    <div class="hero-subtitle">
        Analyze tweets, reviews, and short text using a fine-tuned DistilBERT model trained for
        positive, negative, and neutral sentiment classification.
    </div>
    <div class="info-strip">
        <div class="info-pill">🤖 DistilBERT</div>
        <div class="info-pill">⚡ PyTorch</div>
        <div class="info-pill">💬 3-Class Sentiment</div>
        <div class="info-pill">🖥️ Runtime: {str(device).upper()}</div>
    </div>
</div>
""", unsafe_allow_html=True)


left_col, right_col = st.columns([1.45, 0.75], gap="large")

with left_col:
    st.subheader("Sentiment Input")

    user_text = st.text_area(
        "Enter text below",
        value=st.session_state.input_text,
        height=230,
        placeholder="Example: I really love this product. The quality is excellent and delivery was fast.",
        label_visibility="collapsed"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Characters", len(user_text))
    m2.metric("Words", len(user_text.split()))
    m3.metric("Token Limit", "128")

    analyze = st.button("Analyze Sentiment", use_container_width=True)

with right_col:
    st.subheader("Model Status")

    st.markdown("""
    <div class="status-card">
        <div class="status-label">Model</div>
        <div class="status-value">DistilBERT</div>
    </div>
    <div class="status-card">
        <div class="status-label">Task</div>
        <div class="status-value">Sentiment Classification</div>
    </div>
    <div class="status-card">
        <div class="status-label">Classes</div>
        <div class="status-value">Positive · Neutral · Negative</div>
    </div>
    """, unsafe_allow_html=True)


if analyze:
    if not user_text.strip():
        st.warning("Please enter text before analyzing sentiment.")
    else:
        sentiment, confidence, scores = predict_sentiment(user_text)

        st.markdown("## Analysis Result")

        st.markdown(
            f"""
            <div class="result-{sentiment.lower()}">
                <div class="result-label">{sentiment.upper()}</div>
                <div class="result-confidence">Confidence Score: {confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        score_df = pd.DataFrame({
            "Sentiment": list(scores.keys()),
            "Confidence (%)": list(scores.values())
        }).sort_values("Confidence (%)", ascending=False)

        chart_col, table_col = st.columns([1.25, 0.75], gap="large")

        with chart_col:
            st.subheader("Probability Distribution")
            st.bar_chart(score_df.set_index("Sentiment"))

        with table_col:
            st.subheader("Confidence Scores")
            st.dataframe(score_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.subheader("Quick Test Examples")

ex1, ex2, ex3 = st.columns(3, gap="large")

with ex1:
    if st.button("Positive Sample", use_container_width=True):
        st.session_state.input_text = "I absolutely love this product. The quality is amazing and the delivery was fast."
        st.rerun()
    st.caption("I absolutely love this product. The quality is amazing and the delivery was fast.")

with ex2:
    if st.button("Negative Sample", use_container_width=True):
        st.session_state.input_text = "This is the worst experience I have ever had. I am very disappointed."
        st.rerun()
    st.caption("This is the worst experience I have ever had. I am very disappointed.")

with ex3:
    if st.button("Neutral Sample", use_container_width=True):
        st.session_state.input_text = "The service was okay. Nothing special, but it was acceptable."
        st.rerun()
    st.caption("The service was okay. Nothing special, but it was acceptable.")


st.markdown("---")

summary_col, pipeline_col = st.columns(2, gap="large")

with summary_col:
    st.subheader("Project Summary")
    st.write(
        "This application uses a fine-tuned DistilBERT transformer model to classify short text "
        "into positive, negative, or neutral sentiment. It is suitable for tweets, reviews, "
        "customer feedback, and social media monitoring."
    )

with pipeline_col:
    st.subheader("Inference Pipeline")
    st.write(
        "Input text is tokenized with the DistilBERT tokenizer, passed through the fine-tuned "
        "classification model, converted into probabilities using softmax, and returned with "
        "the highest-confidence sentiment label."
    )


st.markdown("""
<div class="footer">
    Developed by Saif Ullah | Fine-Tuned DistilBERT Twitter Sentiment Analysis
</div>
""", unsafe_allow_html=True)