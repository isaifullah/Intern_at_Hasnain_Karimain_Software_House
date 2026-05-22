# ============================================================
# IMAGE SIMILARITY SEARCH ENGINE - STREAMLIT APP
# Developed by Khalid Saifullah
# ============================================================

import os
import cv2
import pickle
import faiss
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import torch
from transformers import CLIPProcessor, CLIPModel

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Similarity Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM DARK UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1b2440 0%, #090d16 45%, #05070d 100%);
        color: #f5f7fa;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #f5f7fa !important;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(25, 35, 65, 0.95), rgba(10, 15, 28, 0.95));
        border: 1px solid rgba(120, 160, 255, 0.25);
        border-radius: 28px;
        padding: 38px;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.45);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #9cc9ff, #e6f0ff, #b78cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #b7c7e6 !important;
        max-width: 850px;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #9cc9ff !important;
    }

    .metric-label {
        color: #b7c7e6 !important;
        font-size: 0.9rem;
    }

    .control-panel {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        padding: 26px;
        margin-bottom: 26px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }

    .result-card {
        background: rgba(255, 255, 255, 0.065);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 22px;
        padding: 14px;
        margin-bottom: 20px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
        transition: 0.25s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
        border-color: rgba(156, 201, 255, 0.65);
        box-shadow: 0 25px 70px rgba(70, 120, 255, 0.18);
    }

    .product-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-top: 10px;
        min-height: 42px;
    }

    .tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin: 4px 4px 4px 0;
        background: rgba(156, 201, 255, 0.13);
        color: #cfe4ff !important;
        border: 1px solid rgba(156, 201, 255, 0.25);
    }

    .score {
        color: #9cffc7 !important;
        font-weight: 800;
        font-size: 0.9rem;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #5b8cff, #8e5bff);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.85rem 1.2rem;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 15px 35px rgba(91, 140, 255, 0.35);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #7aa4ff, #a87cff);
        color: white;
        border: none;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.18) !important;
        color: white !important;
        border-radius: 14px !important;
    }

    input {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
        border-radius: 14px !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.055);
        border: 1px dashed rgba(156, 201, 255, 0.35);
        border-radius: 20px;
        padding: 18px;
    }

    [data-testid="stImage"] img {
        border-radius: 18px;
    }

    .footer {
        text-align: center;
        color: #8fa3c7 !important;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = "models"
UPLOADS_DIR = "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)

CNN_FEATURES_PATH = os.path.join(MODELS_DIR, "cnn_features.pkl")
IMAGE_PATHS_PATH = os.path.join(MODELS_DIR, "image_paths.pkl")

METADATA_PATH_1 = os.path.join(MODELS_DIR, "metadata.pkl")
METADATA_PATH_2 = os.path.join(MODELS_DIR, "valid_metadata.pkl")

FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")

CLIP_FEATURES_PATH_1 = os.path.join(MODELS_DIR, "clip_features_array.pkl")
CLIP_FEATURES_PATH_2 = os.path.join(MODELS_DIR, "clip_features.pkl")

CLIP_IMAGE_PATHS_PATH = os.path.join(MODELS_DIR, "clip_image_paths.pkl")

CLIP_METADATA_PATH_1 = os.path.join(MODELS_DIR, "clip_metadata.pkl")
CLIP_METADATA_PATH_2 = os.path.join(MODELS_DIR, "valid_clip_metadata.pkl")

CLIP_FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "clip_faiss_index.bin")


# ============================================================
# HELPER FUNCTIONS FOR FILE LOADING
# ============================================================

def get_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


METADATA_PATH = get_existing_path([METADATA_PATH_1, METADATA_PATH_2])
CLIP_FEATURES_PATH = get_existing_path([CLIP_FEATURES_PATH_1, CLIP_FEATURES_PATH_2])
CLIP_METADATA_PATH = get_existing_path([CLIP_METADATA_PATH_1, CLIP_METADATA_PATH_2])


@st.cache_data(show_spinner=False)
def load_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    return model


@st.cache_resource(show_spinner=False)
def load_faiss_index(path):
    return faiss.read_index(path)


@st.cache_resource(show_spinner=False)
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor, device


def check_required_files():
    required_files = {
        "CNN Features": CNN_FEATURES_PATH,
        "Image Paths": IMAGE_PATHS_PATH,
        "Metadata": METADATA_PATH,
        "FAISS Index": FAISS_INDEX_PATH,
        "CLIP Features": CLIP_FEATURES_PATH,
        "CLIP Image Paths": CLIP_IMAGE_PATHS_PATH,
        "CLIP Metadata": CLIP_METADATA_PATH,
        "CLIP FAISS Index": CLIP_FAISS_INDEX_PATH
    }

    missing_files = []

    for name, path in required_files.items():
        if path is None or not os.path.exists(path):
            missing_files.append(name)

    return missing_files


missing_files = check_required_files()

if missing_files:
    st.error("Some required model files are missing. Please run the notebook first and generate all model files.")
    st.write(missing_files)
    st.stop()


# ============================================================
# LOAD MODELS AND DATA
# ============================================================

with st.spinner("Loading AI models and vector databases..."):
    cnn_model = load_cnn_model()

    cnn_features = np.array(load_pickle_file(CNN_FEATURES_PATH), dtype=np.float32)
    image_paths = load_pickle_file(IMAGE_PATHS_PATH)
    metadata = load_pickle_file(METADATA_PATH)

    faiss_index = load_faiss_index(FAISS_INDEX_PATH)

    clip_features = np.array(load_pickle_file(CLIP_FEATURES_PATH), dtype=np.float32)
    clip_image_paths = load_pickle_file(CLIP_IMAGE_PATHS_PATH)
    clip_metadata = load_pickle_file(CLIP_METADATA_PATH)

    clip_index = load_faiss_index(CLIP_FAISS_INDEX_PATH)

    clip_model, clip_processor, device = load_clip_model()


# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def preprocess_image_for_cnn(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image


def extract_cnn_features(image_path):
    image = preprocess_image_for_cnn(image_path)

    if image is None:
        return None

    features = cnn_model.predict(image, verbose=0).flatten()

    norm = np.linalg.norm(features)

    if norm == 0:
        return None

    features = features / norm

    return features.astype(np.float32)


def extract_clip_image_features(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features.cpu().numpy().flatten()

    norm = np.linalg.norm(features)

    if norm == 0:
        return None

    features = features / norm

    return features.astype(np.float32)


def extract_clip_text_features(text_query):
    inputs = clip_processor(
        text=[text_query],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

    features = features.cpu().numpy().flatten()

    norm = np.linalg.norm(features)

    if norm == 0:
        return None

    features = features / norm

    return features.astype(np.float32)


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def filter_metadata_by_category(category):
    if category == "All":
        return cnn_features, image_paths, metadata.reset_index(drop=True)

    filtered_metadata = metadata[metadata["masterCategory"] == category]
    filtered_indices = filtered_metadata.index.tolist()

    filtered_features = cnn_features[filtered_indices]
    filtered_paths = [image_paths[index] for index in filtered_indices]
    filtered_metadata = filtered_metadata.reset_index(drop=True)

    return filtered_features, filtered_paths, filtered_metadata


def build_results(indices, scores, paths, metadata_table):
    results = []

    for index, score in zip(indices, scores):
        if index < 0 or index >= len(paths):
            continue

        row = metadata_table.iloc[index]

        results.append({
            "image_path": paths[index],
            "score": float(score),
            "product_name": row.get("productDisplayName", "Unknown Product"),
            "category": row.get("masterCategory", "Unknown"),
            "sub_category": row.get("subCategory", "Unknown"),
            "article_type": row.get("articleType", "Unknown"),
            "colour": row.get("baseColour", "Unknown")
        })

    return results


def search_by_cosine(query_image_path, top_k=5, category="All"):
    query_features = extract_cnn_features(query_image_path)

    if query_features is None:
        return []

    filtered_features, filtered_paths, filtered_metadata = filter_metadata_by_category(category)

    similarities = cosine_similarity([query_features], filtered_features)[0]
    indices = np.argsort(similarities)[::-1][:top_k]
    scores = similarities[indices]

    return build_results(indices, scores, filtered_paths, filtered_metadata)


def search_by_euclidean(query_image_path, top_k=5, category="All"):
    query_features = extract_cnn_features(query_image_path)

    if query_features is None:
        return []

    filtered_features, filtered_paths, filtered_metadata = filter_metadata_by_category(category)

    distances = euclidean_distances([query_features], filtered_features)[0]
    indices = np.argsort(distances)[:top_k]
    scores = distances[indices]

    return build_results(indices, scores, filtered_paths, filtered_metadata)


def search_by_faiss(query_image_path, top_k=5, category="All"):
    query_features = extract_cnn_features(query_image_path)

    if query_features is None:
        return []

    query_features = np.array([query_features], dtype=np.float32)

    search_k = min(top_k * 8, len(image_paths))
    scores, indices = faiss_index.search(query_features, search_k)

    raw_results = build_results(indices[0], scores[0], image_paths, metadata)

    if category != "All":
        raw_results = [item for item in raw_results if item["category"] == category]

    return raw_results[:top_k]


def search_clip_image(query_image_path, top_k=5, category="All"):
    query_features = extract_clip_image_features(query_image_path)

    if query_features is None:
        return []

    query_features = np.array([query_features], dtype=np.float32)

    search_k = min(top_k * 8, len(clip_image_paths))
    scores, indices = clip_index.search(query_features, search_k)

    raw_results = build_results(indices[0], scores[0], clip_image_paths, clip_metadata)

    if category != "All":
        raw_results = [item for item in raw_results if item["category"] == category]

    return raw_results[:top_k]


def search_clip_text(text_query, top_k=5, category="All"):
    query_features = extract_clip_text_features(text_query)

    if query_features is None:
        return []

    query_features = np.array([query_features], dtype=np.float32)

    search_k = min(top_k * 8, len(clip_image_paths))
    scores, indices = clip_index.search(query_features, search_k)

    raw_results = build_results(indices[0], scores[0], clip_image_paths, clip_metadata)

    if category != "All":
        raw_results = [item for item in raw_results if item["category"] == category]

    return raw_results[:top_k]


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def save_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    save_path = os.path.join(UPLOADS_DIR, "query_image.jpg")
    image.save(save_path)
    return save_path


def display_result_card(result):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.image(result["image_path"], use_container_width=True)

    product_name = result["product_name"]

    if len(product_name) > 55:
        product_name = product_name[:55] + "..."

    st.markdown(f'<div class="product-name">{product_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="score">Score: {result["score"]:.4f}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <span class="tag">{result["category"]}</span>
        <span class="tag">{result["sub_category"]}</span>
        <span class="tag">{result["article_type"]}</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN USER INTERFACE
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Image Similarity Search Engine</div>
        <p class="hero-subtitle">
            A deep learning powered visual search system that finds visually and semantically similar products using CNN embeddings, FAISS vector search, and CLIP multimodal intelligence.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(image_paths)}</div>
            <div class="metric-label">Indexed Images</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{cnn_features.shape[1]}</div>
            <div class="metric-label">CNN Vector Size</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{clip_features.shape[1]}</div>
            <div class="metric-label">CLIP Vector Size</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with metric_col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{device.upper()}</div>
            <div class="metric-label">Runtime Device</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)

control_col1, control_col2, control_col3 = st.columns([1.4, 1, 1])

with control_col1:
    search_mode = st.selectbox(
        "Choose Search Mode",
        [
            "CNN - Cosine Similarity",
            "CNN - Euclidean Distance",
            "CNN - FAISS Fast Search",
            "CLIP - Image Search",
            "CLIP - Text Search"
        ]
    )

with control_col2:
    top_k = st.slider("Number of Results", min_value=4, max_value=20, value=8)

with control_col3:
    if "masterCategory" in metadata.columns:
        categories = ["All"] + sorted(metadata["masterCategory"].dropna().unique().tolist())
    else:
        categories = ["All"]

    selected_category = st.selectbox("Category Filter", categories)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SEARCH AREA
# ============================================================

if search_mode == "CLIP - Text Search":
    st.markdown("### Enter a natural language query")

    text_query = st.text_input(
        "Text Query",
        value="black sports shoes",
        placeholder="Example: black sports shoes, red casual shirt, blue denim jeans"
    )

    search_button = st.button("Search Similar Products")

    if search_button:
        with st.spinner("Searching products using CLIP semantic understanding..."):
            results = search_clip_text(
                text_query=text_query,
                top_k=top_k,
                category=selected_category
            )

        st.markdown(f"## Results for: `{text_query}`")

        if not results:
            st.warning("No results found. Try a different query or category.")
        else:
            result_columns = st.columns(4)

            for index, result in enumerate(results):
                with result_columns[index % 4]:
                    display_result_card(result)

else:
    upload_col, preview_col = st.columns([1, 1])

    with upload_col:
        st.markdown("### Upload Query Image")

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"]
        )

    query_image_path = None

    if uploaded_file is not None:
        query_image_path = save_uploaded_image(uploaded_file)

        with preview_col:
            st.markdown("### Query Preview")
            st.image(query_image_path, use_container_width=True)

        search_button = st.button("Find Similar Images")

        if search_button:
            with st.spinner("Extracting features and searching similar products..."):

                if search_mode == "CNN - Cosine Similarity":
                    results = search_by_cosine(
                        query_image_path=query_image_path,
                        top_k=top_k,
                        category=selected_category
                    )

                elif search_mode == "CNN - Euclidean Distance":
                    results = search_by_euclidean(
                        query_image_path=query_image_path,
                        top_k=top_k,
                        category=selected_category
                    )

                elif search_mode == "CNN - FAISS Fast Search":
                    results = search_by_faiss(
                        query_image_path=query_image_path,
                        top_k=top_k,
                        category=selected_category
                    )

                elif search_mode == "CLIP - Image Search":
                    results = search_clip_image(
                        query_image_path=query_image_path,
                        top_k=top_k,
                        category=selected_category
                    )

            st.markdown("## Similar Image Results")

            if not results:
                st.warning("No results found. Try another image or category.")
            else:
                result_columns = st.columns(4)

                for index, result in enumerate(results):
                    with result_columns[index % 4]:
                        display_result_card(result)

    else:
        st.info("Upload an image to start visual similarity search.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <p>Developed by <b>Saif ullah</b> | Deep Learning | Computer Vision | Semantic Search</p>
    </div>
    """,
    unsafe_allow_html=True
)