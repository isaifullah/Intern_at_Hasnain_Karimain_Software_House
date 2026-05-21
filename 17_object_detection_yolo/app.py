# ============================================================
# PROFESSIONAL STREAMLIT APP: YOLOv8 OBJECT + KIDNEY STONE DETECTION
# ============================================================

import os
import cv2
import tempfile
import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image
from datetime import datetime
from collections import Counter
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="YOLOv8 Detection System",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# DARK UI CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #e5e7eb;
    }

    header {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .main-title {
        padding: 34px;
        border-radius: 26px;
        background: linear-gradient(135deg, #111827, #1e293b, #312e81);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        text-align: center;
        margin-bottom: 30px;
    }

    .main-title h1 {
        font-size: 46px;
        color: #f8fafc;
        margin-bottom: 10px;
        font-weight: 800;
    }

    .main-title p {
        font-size: 18px;
        color: #cbd5e1;
        margin: 0;
    }

    .glass-card {
        padding: 24px;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 16px 40px rgba(0,0,0,0.35);
        margin-bottom: 24px;
    }

    .section-heading {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .section-text {
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.6;
    }

    .warning-note {
        padding: 14px 18px;
        border-radius: 16px;
        background: rgba(127, 29, 29, 0.35);
        border: 1px solid rgba(248, 113, 113, 0.35);
        color: #fecaca;
        margin-top: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 16px;
        border-radius: 18px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.75);
        border-radius: 18px;
        padding: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .stButton > button {
        width: 100%;
        border-radius: 16px;
        padding: 14px 20px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #6d28d9);
        color: white;
        border: none;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 16px;
        padding: 13px 20px;
        background: linear-gradient(135deg, #059669, #0f766e);
        color: white;
        border: none;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 23, 42, 0.85);
        padding: 12px;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 14px;
        padding: 12px 20px;
        color: #cbd5e1;
        font-weight: 700;
        background: rgba(30, 41, 59, 0.7);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
    }

    label, .stRadio label, .stSlider label, .stSelectbox label, .stMultiSelect label {
        color: #e5e7eb !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATHS AND OUTPUT FOLDERS
# ============================================================

OBJECT_MODEL_PATH = "yolov8n.pt"

# Correct Streamlit Cloud path
KIDNEY_MODEL_PATH = "17_object_detection_yolo/runs/detect/kidney_stone_yolo-7/weights/best.pt"

OUTPUT_IMAGE_DIR = "17_object_detection_yolo/output/streamlit_images"
OUTPUT_VIDEO_DIR = "17_object_detection_yolo/output/streamlit_videos"
LOG_DIR = "17_object_detection_yolo/output/logs"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_object_model():
    return YOLO(OBJECT_MODEL_PATH)


@st.cache_resource
def load_kidney_model():
    if not os.path.exists(KIDNEY_MODEL_PATH):
        st.error(f"Kidney model not found at: {KIDNEY_MODEL_PATH}")
        st.stop()

    return YOLO(KIDNEY_MODEL_PATH)


object_model = load_object_model()
kidney_model = load_kidney_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_ids(model, selected_classes):
    class_ids = []

    for class_id, class_name in model.names.items():
        if class_name in selected_classes:
            class_ids.append(class_id)

    return class_ids if class_ids else None


def count_objects(results, model):
    detected_classes = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detected_classes.append(model.names[class_id])

    return dict(Counter(detected_classes))


def extract_detection_table(results, model):
    rows = []

    for result in results:
        if len(result.boxes) == 0:
            rows.append({
                "class_name": "no_detection",
                "confidence": 0.0,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None
            })
        else:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                rows.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

    return pd.DataFrame(rows)


def save_detection_logs(results, model, source_name, log_file):
    rows = []

    for result in results:
        if len(result.boxes) == 0:
            rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source_name,
                "class_name": "no_detection",
                "confidence": 0.0,
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None
            })
        else:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                rows.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": source_name,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

    log_path = os.path.join(LOG_DIR, log_file)
    df = pd.DataFrame(rows)

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", index=False, header=False)
    else:
        df.to_csv(log_path, index=False)

    return log_path


def run_image_detection(model, image, confidence, selected_class_ids=None):
    image_np = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_np,
        conf=confidence,
        classes=selected_class_ids,
        verbose=False
    )

    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return results, annotated_image


def run_kidney_detection(model, image, confidence):
    image_np = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_np,
        conf=confidence,
        save=False,
        verbose=False
    )

    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return results, annotated_image


def run_video_detection(model, video_file, confidence, selected_class_ids, output_name, log_name):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 25

    output_path = os.path.join(OUTPUT_VIDEO_DIR, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=confidence,
            classes=selected_class_ids,
            verbose=False
        )

        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        save_detection_logs(
            results=results,
            model=model,
            source_name=f"{video_file.name}_frame_{frame_number}",
            log_file=log_name
        )

        if total_frames > 0:
            progress = min(frame_number / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.write(f"Processing frame {frame_number} of {total_frames}")

    cap.release()
    writer.release()

    progress_bar.empty()
    status_text.empty()

    return output_path


def run_local_webcam_detection(model, confidence, selected_class_ids=None):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not found. This feature works locally on your computer.")
        return

    st.info("Webcam window opened. Press 'q' on the webcam window to stop.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            classes=selected_class_ids,
            verbose=False
        )

        annotated_frame = results[0].plot()
        object_counts = count_objects(results, model)

        y_position = 30

        for obj, count in object_counts.items():
            cv2.putText(
                annotated_frame,
                f"{obj}: {count}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y_position += 30

        cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def display_summary(results, model):
    counts = count_objects(results, model)
    detection_df = extract_detection_table(results, model)

    if "no_detection" in detection_df["class_name"].values:
        total_detections = 0
    else:
        total_detections = len(detection_df)

    best_confidence = detection_df["confidence"].max() if not detection_df.empty else 0.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Detections", total_detections)

    with col2:
        st.metric("Best Confidence", f"{best_confidence:.2f}")

    with col3:
        st.metric("Detected Classes", len(counts))

    st.markdown("### Detection Details")
    st.dataframe(detection_df, use_container_width=True)

    st.markdown("### Object Count")
    st.write(counts)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        <h1>🎯 YOLOv8 Intelligent Detection System</h1>
        <p>General Object Detection + Custom Kidney Stone X-ray Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP NAVIGATION
# ============================================================

main_tab_1, main_tab_2, main_tab_3 = st.tabs(
    [
        "🌍 Object Detection",
        "🏥 Kidney Stone X-ray",
        "📁 Detection Logs"
    ]
)


# ============================================================
# OBJECT DETECTION SECTION
# ============================================================

with main_tab_1:

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">General Object Detection</div>
            <div class="section-text">
                Detect common objects such as people, cars, animals, laptops, bottles, chairs, buses, and more using pretrained YOLOv8.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Choose Detection Mode",
        ["Image Detection", "Video Detection", "Real-Time Webcam"],
        horizontal=True
    )

    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        confidence = st.slider(
            "Object Detection Confidence",
            min_value=0.10,
            max_value=1.00,
            value=0.25,
            step=0.05
        )

    with top_col2:
        object_classes = list(object_model.names.values())

        selected_classes = st.multiselect(
            "Filter Classes",
            options=object_classes,
            default=[]
        )

    selected_class_ids = get_class_ids(object_model, selected_classes)

    if mode == "Image Detection":

        uploaded_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            key="object_image"
        )

        if uploaded_image is not None:

            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)

            if st.button("Run Object Image Detection"):

                results, annotated_image = run_image_detection(
                    model=object_model,
                    image=image,
                    confidence=confidence,
                    selected_class_ids=selected_class_ids
                )

                save_detection_logs(
                    results=results,
                    model=object_model,
                    source_name=uploaded_image.name,
                    log_file="object_detection_logs.csv"
                )

                output_path = os.path.join(
                    OUTPUT_IMAGE_DIR,
                    f"object_detected_{uploaded_image.name}"
                )

                cv2.imwrite(
                    output_path,
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                )

                with col2:
                    st.markdown("### Detection Result")
                    st.image(annotated_image, use_container_width=True)

                st.success("Object detection completed.")
                display_summary(results, object_model)

                with open(output_path, "rb") as file:
                    st.download_button(
                        "Download Detected Image",
                        file,
                        file_name=f"object_detected_{uploaded_image.name}"
                    )

    elif mode == "Video Detection":

        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov"],
            key="object_video"
        )

        if uploaded_video is not None:

            st.markdown("### Original Video")
            st.video(uploaded_video)

            if st.button("Run Object Video Detection"):

                with st.spinner("Processing video..."):
                    output_path = run_video_detection(
                        model=object_model,
                        video_file=uploaded_video,
                        confidence=confidence,
                        selected_class_ids=selected_class_ids,
                        output_name="object_detected_video.mp4",
                        log_name="object_detection_logs.csv"
                    )

                st.success("Video detection completed.")

                st.markdown("### Detected Video Preview")
                st.video(output_path)

                with open(output_path, "rb") as file:
                    st.download_button(
                        "Download Detected Video",
                        file,
                        file_name="object_detected_video.mp4"
                    )

    else:

        st.markdown(
            """
            <div class="glass-card">
                <div class="section-text">
                    Webcam detection opens a local OpenCV window. This works best when running the app on your own computer.
                    Press <b>q</b> in the webcam window to stop.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Start Real-Time Webcam Detection"):

            run_local_webcam_detection(
                model=object_model,
                confidence=confidence,
                selected_class_ids=selected_class_ids
            )


# ============================================================
# KIDNEY STONE X-RAY SECTION
# ============================================================

with main_tab_2:

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">Kidney Stone X-ray Detection</div>
            <div class="section-text">
                Upload X-ray or medical scan images and detect kidney stone regions using your custom fine-tuned YOLOv8 model.
            </div>
            <div class="warning-note">
                This system is for educational and demonstration purposes only. It is not a replacement for professional medical diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    kidney_mode = st.radio(
        "Choose X-ray Detection Mode",
        ["Single X-ray Detection", "Batch X-ray Detection"],
        horizontal=True
    )

    kidney_confidence = st.slider(
        "X-ray Detection Confidence",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01
    )

    if kidney_mode == "Single X-ray Detection":

        uploaded_xray = st.file_uploader(
            "Upload X-ray image",
            type=["jpg", "jpeg", "png"],
            key="kidney_single"
        )

        if uploaded_xray is not None:

            image = Image.open(uploaded_xray)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original X-ray")
                st.image(image, use_container_width=True)

            if st.button("Run Kidney Stone Detection"):

                results, annotated_image = run_kidney_detection(
                    model=kidney_model,
                    image=image,
                    confidence=kidney_confidence
                )

                save_detection_logs(
                    results=results,
                    model=kidney_model,
                    source_name=uploaded_xray.name,
                    log_file="kidney_stone_logs.csv"
                )

                output_path = os.path.join(
                    OUTPUT_IMAGE_DIR,
                    f"kidney_detected_{uploaded_xray.name}"
                )

                cv2.imwrite(
                    output_path,
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                )

                with col2:
                    st.markdown("### Detection Result")
                    st.image(annotated_image, use_container_width=True)

                st.success("Kidney stone detection completed.")
                display_summary(results, kidney_model)

                with open(output_path, "rb") as file:
                    st.download_button(
                        "Download X-ray Result",
                        file,
                        file_name=f"kidney_detected_{uploaded_xray.name}"
                    )

    else:

        st.markdown(
            """
            <div class="glass-card">
                <div class="section-text">
                    Upload multiple X-ray images. The app will show detection results directly on the website and save logs automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        batch_images = st.file_uploader(
            "Upload multiple X-ray images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="kidney_batch"
        )

        if batch_images:

            max_images = st.slider(
                "Number of batch images to display",
                min_value=1,
                max_value=len(batch_images),
                value=min(6, len(batch_images))
            )

            if st.button("Run Batch X-ray Detection"):

                st.success(f"Processing {max_images} X-ray images...")

                for index, uploaded_file in enumerate(batch_images[:max_images], start=1):

                    image = Image.open(uploaded_file)

                    results, annotated_image = run_kidney_detection(
                        model=kidney_model,
                        image=image,
                        confidence=kidney_confidence
                    )

                    save_detection_logs(
                        results=results,
                        model=kidney_model,
                        source_name=uploaded_file.name,
                        log_file="kidney_stone_logs.csv"
                    )

                    st.markdown(f"### X-ray Result {index}: {uploaded_file.name}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            image,
                            caption="Original X-ray",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            annotated_image,
                            caption="Detected X-ray",
                            use_container_width=True
                        )

                    display_summary(results, kidney_model)


# ============================================================
# DETECTION LOGS SECTION
# ============================================================

with main_tab_3:

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-heading">Detection Logs</div>
            <div class="section-text">
                Review saved detection logs from both object detection and kidney stone detection modules.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    log_file = st.selectbox(
        "Select log file",
        ["object_detection_logs.csv", "kidney_stone_logs.csv"]
    )

    log_path = os.path.join(LOG_DIR, log_file)

    if os.path.exists(log_path):

        logs_df = pd.read_csv(log_path)

        st.metric("Total Log Entries", len(logs_df))

        st.dataframe(
            logs_df.tail(100),
            use_container_width=True
        )

        with open(log_path, "rb") as file:
            st.download_button(
                "Download Log CSV",
                file,
                file_name=log_file
            )

    else:
        st.info("No logs found yet. Run detection first.")