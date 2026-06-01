# ============================================================
# PROFESSIONAL STREAMLIT UI FOR TRANSFORMER NER SYSTEM
# No Sidebar | No Empty Blocks | Clean Text + Batch Sections
# Fixed DATE/TIME Duplicate Regex Issue
# ============================================================

import os
import re
import torch
import pandas as pd
import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# ============================================================
# CONFIGURATION
# ============================================================

SMALL_MODEL_NAME = "dslim/bert-base-NER"
LARGE_MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
OUTPUT_CSV_PATH = "outputs/extracted_entities.csv"

LABEL_MAPPING = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "MISC": "MISCELLANEOUS",
    "DATE": "DATE",
    "TIME": "TIME"
}

ENTITY_COLORS = {
    "PERSON": "#ff4d6d",
    "ORGANIZATION": "#38bdf8",
    "LOCATION": "#22c55e",
    "DATE": "#facc15",
    "TIME": "#a78bfa",
    "MISCELLANEOUS": "#fb923c"
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_ner_model(model_type):
    """
    Loads selected Hugging Face Transformer NER model.
    """

    model_name = LARGE_MODEL_NAME if model_type == "Large Model" else SMALL_MODEL_NAME
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipeline = pipeline(
        task="ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device
    )

    return ner_pipeline, model_name, device


# ============================================================
# DATE AND TIME EXTRACTION
# ============================================================

def is_overlapping(new_entity, existing_entities):
    """
    Checks whether a new entity overlaps with an already extracted entity.

    This prevents duplicate time matches such as:
    - 10:30 AM
    - 30 AM
    """

    new_start = new_entity["start"]
    new_end = new_entity["end"]

    for entity in existing_entities:
        existing_start = entity["start"]
        existing_end = entity["end"]

        if new_start < existing_end and new_end > existing_start:
            return True

    return False


def extract_date_time_entities(text):
    """
    Extracts DATE and TIME entities using regex patterns.

    This is added because most CoNLL-based Transformer NER models mainly detect:
    PERSON, ORGANIZATION, LOCATION, and MISC.

    This updated version removes duplicate overlapping matches.
    """

    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
    ]

    time_patterns = [
        r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b",
        r"(?<!:)\b\d{1,2}\s?(?:AM|PM|am|pm)\b"
    ]

    entities = []

    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            entity = {
                "entity": match.group(),
                "label": "DATE",
                "score": 1.0,
                "start": match.start(),
                "end": match.end(),
                "source": "Regex"
            }

            if not is_overlapping(entity, entities):
                entities.append(entity)

    for pattern in time_patterns:
        for match in re.finditer(pattern, text):
            entity = {
                "entity": match.group(),
                "label": "TIME",
                "score": 1.0,
                "start": match.start(),
                "end": match.end(),
                "source": "Regex"
            }

            if not is_overlapping(entity, entities):
                entities.append(entity)

    return entities


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text, ner_pipeline, confidence_threshold):
    """
    Extracts named entities using Transformer model and regex DATE/TIME extraction.
    """

    transformer_results = ner_pipeline(text)
    entities = []

    for item in transformer_results:
        label = LABEL_MAPPING.get(item["entity_group"], item["entity_group"])
        score = round(float(item["score"]), 4)

        if score >= confidence_threshold:
            entities.append({
                "entity": item["word"],
                "label": label,
                "score": score,
                "start": item["start"],
                "end": item["end"],
                "source": "Transformer"
            })

    regex_entities = extract_date_time_entities(text)
    entities.extend(regex_entities)

    df = pd.DataFrame(entities)

    if not df.empty:
        df = df.sort_values(by=["start", "end"]).reset_index(drop=True)

    return df


# ============================================================
# HIGHLIGHT ENTITIES
# ============================================================

def highlight_entities(text, entities_df):
    """
    Highlights detected entities inside the original text.
    """

    if entities_df.empty:
        return text

    highlighted_text = ""
    last_index = 0

    valid_df = entities_df.dropna(subset=["start", "end"]).sort_values(by="start")

    for _, row in valid_df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        label = row["label"]
        color = ENTITY_COLORS.get(label, "#94a3b8")

        highlighted_text += text[last_index:start]
        highlighted_text += (
            f"<span class='entity-chip' style='background:{color};'>"
            f"{text[start:end]} <b>{label}</b>"
            f"</span>"
        )
        last_index = end

    highlighted_text += text[last_index:]

    return highlighted_text


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

def save_entities_to_csv(df, text):
    """
    Saves extracted entities to a CSV file.
    """

    if df.empty:
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    save_df = df.copy()
    save_df["input_text"] = text
    save_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(OUTPUT_CSV_PATH):
        old_df = pd.read_csv(OUTPUT_CSV_PATH)
        save_df = pd.concat([old_df, save_df], ignore_index=True)

    save_df.to_csv(OUTPUT_CSV_PATH, index=False)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NER Intelligence System",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(168,85,247,0.14), transparent 28%),
        linear-gradient(135deg, #020617 0%, #0f172a 50%, #111827 100%);
    color: #f8fafc;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero {
    padding: 34px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.85));
    border: 1px solid rgba(148,163,184,0.22);
    box-shadow: 0 25px 80px rgba(0,0,0,0.35);
    margin-bottom: 28px;
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    letter-spacing: -1.2px;
    line-height: 1.05;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 14px;
    color: #cbd5e1;
    font-size: 18px;
    line-height: 1.7;
    max-width: 900px;
}

.metric-card {
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(148,163,184,0.20);
    text-align: center;
    margin-bottom: 18px;
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #38bdf8;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 4px;
}

.entity-chip {
    display: inline-block;
    color: #020617;
    padding: 5px 10px;
    margin: 3px;
    border-radius: 10px;
    font-weight: 700;
}

.highlight-box {
    font-size: 18px;
    line-height: 2.15;
    padding: 24px;
    border-radius: 22px;
    background: rgba(2,6,23,0.65);
    border: 1px solid rgba(148,163,184,0.22);
    margin-top: 10px;
}

.stTextArea textarea {
    background-color: rgba(15,23,42,0.95) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148,163,184,0.32) !important;
    border-radius: 18px !important;
    font-size: 16px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(15,23,42,0.95) !important;
    color: #f8fafc !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,0.32) !important;
}

.stButton button {
    width: 100%;
    border-radius: 16px;
    border: none;
    padding: 14px 22px;
    font-weight: 800;
    font-size: 16px;
    color: #020617;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    box-shadow: 0 14px 35px rgba(56,189,248,0.24);
}

.stDownloadButton button {
    width: 100%;
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.25);
    padding: 13px 20px;
    font-weight: 800;
    color: #f8fafc;
    background: rgba(30,41,59,0.95);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(15,23,42,0.95);
    border-radius: 16px;
    color: #cbd5e1;
    padding: 12px 22px;
    border: 1px solid rgba(148,163,184,0.20);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    color: #020617 !important;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">NER Intelligence System</div>
    <div class="hero-subtitle">
        A professional Transformer-based Named Entity Recognition dashboard for detecting
        Person, Organization, Location, Date, and Time entities with confidence filtering,
        highlighted output, CSV export, and batch processing.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TWO MAIN SECTIONS
# ============================================================

text_tab, batch_tab = st.tabs([
    "📝 Text Entity Extraction",
    "📂 Batch CSV Processing"
])


# ============================================================
# SECTION 1: TEXT ENTITY EXTRACTION
# ============================================================

with text_tab:

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["Small Model", "Large Model"],
            help="Small model is faster. Large model gives stronger predictions but needs more memory."
        )

    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.70,
            step=0.05
        )

    input_text = st.text_area(
        "Input Text",
        height=220,
        placeholder="Example: Elon Musk founded Tesla in California on 12 March 2024 at 10:30 AM."
    )

    run_button = st.button("🚀 Analyze Text", use_container_width=True)

    if run_button:
        if not input_text.strip():
            st.warning("Please enter text before analysis.")
        else:
            with st.spinner("Loading model and extracting entities..."):
                ner_pipeline, loaded_model_name, device = load_ner_model(model_type)
                entities_df = extract_entities(input_text, ner_pipeline, confidence_threshold)
                save_entities_to_csv(entities_df, input_text)

            total_entities = len(entities_df)
            person_count = len(entities_df[entities_df["label"] == "PERSON"]) if not entities_df.empty else 0
            org_count = len(entities_df[entities_df["label"] == "ORGANIZATION"]) if not entities_df.empty else 0
            loc_count = len(entities_df[entities_df["label"] == "LOCATION"]) if not entities_df.empty else 0
            date_time_count = len(entities_df[entities_df["label"].isin(["DATE", "TIME"])]) if not entities_df.empty else 0

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_entities}</div>
                    <div class="metric-label">Total Entities</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{person_count}</div>
                    <div class="metric-label">Persons</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{org_count}</div>
                    <div class="metric-label">Organizations</div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{date_time_count}</div>
                    <div class="metric-label">Date / Time</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Highlighted Text")
            highlighted_text = highlight_entities(input_text, entities_df)
            st.markdown(f"<div class='highlight-box'>{highlighted_text}</div>", unsafe_allow_html=True)

            st.markdown("### Extracted Entities")

            if entities_df.empty:
                st.info("No entities found. Try lowering the confidence threshold.")
            else:
                st.dataframe(entities_df, use_container_width=True)

                csv_data = entities_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Current Results",
                    data=csv_data,
                    file_name="ner_results.csv",
                    mime="text/csv"
                )

            device_name = "GPU" if device == 0 else "CPU"

            st.markdown("### Model Information")
            st.markdown(f"""
            **Loaded Model:** `{loaded_model_name}`  
            **Running Device:** `{device_name}`  
            **Confidence Threshold:** `{confidence_threshold}`
            """)


# ============================================================
# SECTION 2: BATCH CSV PROCESSING
# ============================================================

with batch_tab:

    batch_col1, batch_col2 = st.columns(2)

    with batch_col1:
        batch_model_type = st.selectbox(
            "Select Batch Model",
            ["Small Model", "Large Model"],
            key="batch_model_type"
        )

    with batch_col2:
        batch_confidence = st.slider(
            "Batch Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.70,
            step=0.05,
            key="batch_confidence"
        )

    uploaded_file = st.file_uploader(
        "Upload CSV File With A Column Named text",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "text" not in batch_df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            st.success(f"{len(batch_df)} records loaded successfully.")
            st.dataframe(batch_df.head(), use_container_width=True)

            process_batch = st.button("⚡ Process Batch File", use_container_width=True)

            if process_batch:
                ner_pipeline, loaded_model_name, device = load_ner_model(batch_model_type)

                all_results = []

                with st.spinner("Processing batch file..."):
                    for text in batch_df["text"].dropna():
                        temp_df = extract_entities(str(text), ner_pipeline, batch_confidence)

                        if not temp_df.empty:
                            temp_df["input_text"] = text
                            all_results.append(temp_df)

                if all_results:
                    final_batch_df = pd.concat(all_results, ignore_index=True)

                    st.success("Batch processing completed successfully.")
                    st.dataframe(final_batch_df, use_container_width=True)

                    batch_csv = final_batch_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Batch Results",
                        data=batch_csv,
                        file_name="batch_ner_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No entities found in uploaded CSV.")