# 🎯 YOLOv8 Object Detection System

A professional real-time Object Detection System built using YOLOv8, OpenCV, and PyTorch.  
This project supports image detection, video detection, webcam detection, object counting, CSV logging, Streamlit deployment, and custom dataset fine-tuning using a Kidney Stone Detection dataset.

Dataset link = https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images

---

# 📌 Project Overview

This project demonstrates how modern deep learning models such as YOLOv8 can perform fast and accurate object detection in real time.

The system can:
- Detect multiple objects in images and videos
- Draw bounding boxes around detected objects
- Display class labels and confidence scores
- Perform real-time webcam detection
- Save detected outputs
- Count detected objects
- Save detection logs in CSV format
- Fine-tune YOLOv8 on a custom medical dataset

The project is divided into two main parts:

1. Pretrained YOLOv8 Object Detection
2. Custom Kidney Stone Detection using Fine-Tuning

---

# 🚀 Features

## ✅ Pretrained YOLOv8 Detection
- Real-time object detection
- Multiple object classification
- Bounding box visualization
- Confidence score display
- Image detection
- Video detection
- Webcam detection

## ✅ Advanced Features
- Save detected images
- Save detected videos
- Filter specific classes
- Count detected objects
- Save detection logs in CSV
- Detection visualization graphs

## ✅ Streamlit Web Application
- Upload image/video
- Run object detection
- Download detected output
- Interactive UI

## ✅ Custom Dataset Fine-Tuning
- YOLOv8 custom training
- Kidney stone detection
- Medical image object detection
- Custom model evaluation

---

# 🛠 Technologies Used

- Python
- OpenCV
- YOLOv8
- PyTorch
- Ultralytics
- Streamlit
- Pandas
- Matplotlib

---

# 📂 Project Structure

```text
17_object_detection_yolo/
│
├── input/
│   ├── images/
│   └── videos/
│
├── output/
│   ├── detected_images/
│   ├── detected_videos/
│   └── logs/
│
├── kidney_stone_dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── runs/
│
├── app.py
├── object_detection.ipynb
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/isaifullah/object_detection_yolo.git
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
object_detection.ipynb
```

---

# 🖼 Image Detection

The system can detect multiple objects from images using YOLOv8 pretrained weights.

Example detections:
- Person
- Car
- Dog
- Bicycle
- Truck

---

# 🎥 Video Detection

The project supports real-time video processing using OpenCV.

Features:
- Frame-by-frame detection
- Bounding box visualization
- Video output saving

---

# 📷 Webcam Detection

Real-time webcam detection is supported using OpenCV and YOLOv8.

Press:

```text
q
```

to stop webcam detection.

---

# 📊 Detection Logs

All detections are automatically saved into CSV format.

Saved information:
- Class name
- Confidence score
- Bounding box coordinates
- Timestamp
- Source file

---

# 📈 Object Counting

The system automatically counts detected objects inside images and video frames.

Example:

```text
Person: 3
Car: 2
Dog: 1
```

---

# 🧠 Custom Dataset Fine-Tuning

This project also includes fine-tuning YOLOv8 on a Kidney Stone Detection dataset.

The custom model can:
- Detect kidney stones in medical images
- Perform medical object detection
- Generate custom predictions using trained weights

---

# 🏥 Kidney Stone Dataset

Dataset structure follows YOLO format:

```text
train/images
train/labels

valid/images
valid/labels

test/images
test/labels
```

---

# 🌐 Streamlit Web Application

Run the Streamlit application:

```bash
streamlit run app.py
```

Features:
- Upload image/video
- Detect objects
- Download outputs
- Interactive UI

---

# 📌 Example Output

```text
Detected Objects:
- Person (0.92)
- Car (0.88)
- Dog (0.85)
```

---

# 📊 YOLOv8 Advantages

- Extremely fast
- High detection accuracy
- Real-time performance
- Detects multiple objects simultaneously
- Better performance than traditional methods

---

# 🔥 Future Improvements

- YOLOv8 segmentation
- Object tracking
- Medical report generation
- Cloud deployment
- GPU optimization
- Multi-class medical detection

---

# 👨‍💻 Author

Saif ullah

Github = https://github.com/isaifullah/


---

# ⭐ Conclusion

This project demonstrates a complete industry-level Object Detection System using YOLOv8 with both pretrained and custom fine-tuned models. The system provides real-time detection, medical image detection, Streamlit deployment, logging, visualization, and advanced object detection functionalities.