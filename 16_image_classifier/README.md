# 🌸 Flower Image Classification System using Transfer Learning

## 📖 Project Overview

This project is a deep learning-based Flower Image Classification System developed using Transfer Learning techniques. The system uses the pretrained MobileNetV2 architecture to classify flower images into multiple categories.

The project demonstrates the complete deep learning workflow including image preprocessing, data augmentation, transfer learning, model training, fine-tuning, evaluation, prediction, and deployment using Streamlit.

This project was developed as an industry-level Computer Vision and Deep Learning project using TensorFlow and Keras.

---

# 🚀 Features

- Flower Image Classification
- Transfer Learning using MobileNetV2
- Image Preprocessing and Resizing
- Data Augmentation
- Accuracy and Loss Visualization
- Confusion Matrix
- Classification Report
- Fine-Tuning of Pretrained Model
- Model Saving and Loading
- Single Image Prediction
- Real-Time Webcam Prediction
- Streamlit Web Application Deployment

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- Streamlit
- Pillow

---

# 📂 Project Structure

```text
16_image_classifier/
│
├── flowers dataset/
│   ├── train/
│   ├── val/
│   ├── train.csv
│   └── val.csv
│
├── models/
│   ├── flower_classification_model.h5
│   └── class_names.json
│
├── app.py
├── image_classifier.ipynb
├── requirements.txt
└── README.md
```

---

# 🌼 Dataset Information

The dataset contains flower images organized into training and validation folders. Each flower category is stored inside its own folder, and the folder names are automatically used as class labels during training.

Example dataset structure:

```text
flowers dataset/
│
├── train/
│   ├── astilbe/
│   ├── bellflower/
│   ├── iris/
│   └── ...
│
├── val/
│   ├── astilbe/
│   ├── bellflower/
│   ├── iris/
│   └── ...
```

The dataset also includes:

- `train.csv`
- `val.csv`

These CSV files are used for dataset inspection and analysis purposes.

---

# 🧠 Transfer Learning Model

This project uses MobileNetV2 as a pretrained transfer learning model.

The original ImageNet classification layers are removed and replaced with custom layers for flower classification.

The final model architecture includes:

- MobileNetV2 Base Model
- Global Average Pooling Layer
- Dropout Layers
- Dense Layer
- Softmax Output Layer

---

# 🔄 Training Workflow

The complete training pipeline includes:

1. Loading flower images from folders
2. Image preprocessing and resizing
3. Applying data augmentation
4. Building transfer learning model
5. Training classification head
6. Evaluating model performance
7. Fine-tuning pretrained layers
8. Saving trained model
9. Testing predictions on new images

---

# 📊 Model Evaluation

The model is evaluated using multiple evaluation techniques:

- Validation Accuracy
- Validation Loss
- Accuracy and Loss Graphs
- Confusion Matrix
- Classification Report

These evaluation methods help analyze model performance and class-wise prediction behavior.

---

# 🖼 Single Image Prediction

The project supports prediction on custom flower images by uploading or providing an image path.

The system displays:

- Predicted Flower Class
- Confidence Score
- Uploaded Image Preview

---

# 📷 Real-Time Webcam Prediction

A real-time webcam prediction system is also included as a bonus feature using OpenCV.

The webcam feature performs:

- Live Frame Capture
- Real-Time Prediction
- Confidence Display
- Live Classification Output

> Note: Webcam prediction may not always perform as accurately as image upload prediction because of lighting conditions, camera quality, background noise, and object positioning.

---

# 🌐 Streamlit Web Application

The project includes a complete Streamlit web application for deployment.

Users can:

- Upload flower images
- Get predictions instantly
- View confidence scores
- Interact with the model through a web interface

Run the Streamlit application using:

```bash
streamlit run app.py
```

---

# ⚙️ Installation

Install all required libraries using:

```bash
pip install -r requirements.txt
```

---

# ☁️ Streamlit Cloud Deployment

To deploy the project on Streamlit Cloud:

1. Upload the project to GitHub
2. Ensure these files are included:
   - `app.py`
   - `requirements.txt`
   - `models/flower_classification_model.h5`
   - `models/class_names.json`
3. Open Streamlit Cloud
4. Connect GitHub repository
5. Select `app.py`
6. Deploy the application

---

# 📌 Important Notes

- Folder names are used as actual class labels during training.
- CSV labels are only used for dataset inspection and analysis.
- Image upload prediction is more reliable than webcam prediction.
- Fine-tuning improves overall model performance.

---

# ✅ Project Status

This project successfully satisfies all required and bonus features of the Image Classification using Transfer Learning task.

Implemented Features:

- ✅ Transfer Learning
- ✅ Data Augmentation
- ✅ Fine-Tuning
- ✅ Accuracy and Loss Visualization
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Model Saving and Loading
- ✅ Streamlit Deployment
- ✅ Real-Time Webcam Prediction

---

# 👨‍💻 Developed By

## Saif Ullah

GitHub: https://github.com/isaifullah