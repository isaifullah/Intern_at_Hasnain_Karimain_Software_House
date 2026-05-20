import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flower Classification System",
    page_icon="🌸",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flower_classification_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #312e81 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 2rem 1rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.8rem;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #cbd5e1;
    max-width: 760px;
    margin: auto;
    line-height: 1.7;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(18px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    margin-bottom: 18px;
}

.metric-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 20px;
    padding: 22px;
    text-align: center;
    margin-bottom: 18px;
}

.metric-label {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin-bottom: 8px;
}

.metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
}

.result-success {
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(5,150,105,0.18));
    border: 1px solid rgba(52,211,153,0.45);
    border-radius: 24px;
    padding: 28px;
    text-align: center;
}

.result-warning {
    background: linear-gradient(135deg, rgba(245,158,11,0.25), rgba(234,88,12,0.18));
    border: 1px solid rgba(251,191,36,0.45);
    border-radius: 24px;
    padding: 28px;
    text-align: center;
}

.result-title {
    font-size: 0.95rem;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 10px;
}

.result-class {
    font-size: 2.3rem;
    color: #ffffff;
    font-weight: 900;
    margin-bottom: 8px;
}

.result-confidence {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 2.5rem;
}

[data-testid="stSidebar"] {
    display: none;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.35);
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stCameraInput"] {
    background: rgba(15, 23, 42, 0.35);
    border-radius: 18px;
    padding: 18px;
}

h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #f8fafc;
}

.stAlert {
    background-color: rgba(15, 23, 42, 0.65);
    color: #f8fafc;
}

button {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND CLASS NAMES
# ============================================================

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not CLASS_NAMES_PATH.exists():
    st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
    st.stop()


model = load_trained_model()
class_names = load_class_names()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_flower(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_class = class_names[predicted_index]

    return prediction, predicted_class, confidence


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🌸 Flower Image Classification</div>
    <div class="hero-subtitle">
        A professional deep learning system powered by transfer learning.
        Upload a flower image or use live camera capture to classify flowers with confidence scores and prediction insights.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# METRIC CARDS
# ============================================================

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Architecture</div>
        <div class="metric-value">MobileNetV2</div>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Learning Method</div>
        <div class="metric-value">Transfer</div>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Flower Classes</div>
        <div class="metric-value">{len(class_names)}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PREDICTION CONTROL PANEL
# ============================================================

st.markdown("""
<div class="glass-card">
    <h3>Prediction Control Panel</h3>
    <p>
        Choose between image upload and live camera capture. For best results, use a clear flower image with good lighting and minimum background noise.
    </p>
</div>
""", unsafe_allow_html=True)

mode = st.radio(
    "Choose Prediction Mode",
    ["Upload Image", "Live Camera"],
    horizontal=True
)

selected_img = None


# ============================================================
# UPLOAD IMAGE MODE
# ============================================================

upload_col, guide_col = st.columns([1.1, 0.9])

if mode == "Upload Image":
    with upload_col:
        st.subheader("Upload Flower Image")

        uploaded_file = st.file_uploader(
            "Choose a flower image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.caption("Supported formats: JPG, JPEG, PNG")

        if uploaded_file is not None:
            selected_img = Image.open(uploaded_file).convert("RGB")

    with guide_col:
        st.markdown("""
        <div class="glass-card">
            <h3>Upload Mode</h3>
            <p>
                Upload mode is recommended for accurate predictions because uploaded images are usually clearer than webcam frames.
            </p>
            <p>
                The model will process the image and return the predicted flower class with confidence score.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# LIVE CAMERA MODE
# ============================================================

if mode == "Live Camera":
    with upload_col:
        st.subheader("Live Camera Flower Capture")

        camera_file = st.camera_input(
            "Capture a flower image",
            label_visibility="collapsed"
        )

        st.caption("Use good lighting and keep the flower centered in the camera frame.")

        if camera_file is not None:
            selected_img = Image.open(camera_file).convert("RGB")

    with guide_col:
        st.markdown("""
        <div class="glass-card">
            <h3>Live Camera Mode</h3>
            <p>
                Live camera mode allows you to capture a flower image directly through the webcam and classify it instantly.
            </p>
            <p>
                This mode is stable on Streamlit Cloud and works better than heavy real-time video dependencies.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PREDICTION RESULTS
# ============================================================

if selected_img is not None:
    prediction, predicted_class, confidence = predict_flower(selected_img)

    st.markdown("""
    <div class="glass-card">
        <h3>Prediction Analysis</h3>
        <p>
            Image received successfully. The model has processed the flower image and generated the prediction results below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    image_col, prediction_col = st.columns([1, 1])

    with image_col:
        st.subheader("Selected Image")
        st.image(selected_img, use_container_width=True)

    with prediction_col:
        st.subheader("Prediction Result")

        if confidence < 0.60:
            st.markdown(f"""
            <div class="result-warning">
                <div class="result-title">Low Confidence Prediction</div>
                <div class="result-class">{predicted_class}</div>
                <div class="result-confidence">Confidence Score: {confidence:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.warning("Try uploading or capturing a clearer flower image.")

        else:
            st.markdown(f"""
            <div class="result-success">
                <div class="result-title">Predicted Flower</div>
                <div class="result-class">{predicted_class}</div>
                <div class="result-confidence">Confidence Score: {confidence:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.success("Prediction completed successfully.")

    prob_col, top_col = st.columns([1.2, 0.8])

    with prob_col:
        st.subheader("Top Prediction Probabilities")

        top_indices = np.argsort(prediction[0])[-5:][::-1]

        probability_data = {
            class_names[i]: float(prediction[0][i])
            for i in top_indices
        }

        st.bar_chart(probability_data)

    with top_col:
        st.subheader("Top 3 Predictions")

        top_3_indices = np.argsort(prediction[0])[-3:][::-1]

        for rank, index in enumerate(top_3_indices, start=1):
            st.write(f"**{rank}. {class_names[index]}**")
            st.progress(float(prediction[0][index]))
            st.caption(f"Confidence: {prediction[0][index]:.2f}")


# ============================================================
# READY MESSAGE
# ============================================================

else:
    st.markdown("""
    <div class="glass-card">
        <h3>Ready for Prediction</h3>
        <p>
            Select upload mode or live camera mode, then provide a flower image.
            The system will display the predicted class, confidence score, top prediction probabilities, and top 3 model guesses.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Flower Image Classification System using Transfer Learning | Developed by Saif Ullah
</div>
""", unsafe_allow_html=True)