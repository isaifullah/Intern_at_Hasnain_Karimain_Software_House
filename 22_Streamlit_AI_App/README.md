# 🤖 AI Model Deployment Dashboard using Streamlit

A professional Machine Learning deployment project that transforms trained AI models into an interactive web application using Streamlit. The dashboard allows users to perform real-time predictions through manual input or batch prediction using CSV files while providing interactive analytics and model performance visualization.

---

## 📌 Project Overview

This project demonstrates the complete machine learning deployment workflow, from model training to creating a production-ready web application. The system supports multiple machine learning models, real-time inference, batch prediction, and interactive visual analytics through a modern Streamlit dashboard.

The application is designed to provide a user-friendly interface where users can enter feature values manually, upload datasets for bulk predictions, compare model performance, and visualize prediction results using interactive charts.

---

## 🎯 Objectives

* Train and evaluate machine learning models.
* Save trained models for deployment.
* Build a professional Streamlit dashboard.
* Perform real-time predictions.
* Support batch predictions through CSV uploads.
* Compare multiple machine learning models.
* Visualize prediction confidence and model performance.
* Deploy the application locally or on Streamlit Cloud.

---

## 🚀 Features

### Core Features

* Real-time prediction system
* Interactive Streamlit dashboard
* Manual data entry interface
* Batch prediction using CSV files
* Model performance comparison
* Prediction confidence visualization
* Downloadable prediction results

### Advanced Features

* Multiple model selection
* Lightweight and advanced model options
* Interactive Plotly visualizations
* Prediction confidence gauge
* Radar chart for feature analysis
* Batch prediction analytics dashboard
* Pie chart distribution analysis
* Bar chart performance comparison
* Responsive modern dark-themed UI

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Model Persistence

* Joblib

### Visualization

* Plotly
* Matplotlib

### Deployment

* Streamlit

---

## 📂 Project Structure

```text
AI_Model_Deployment_Dashboard/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── iris_dataset.csv
│
├── models/
│   ├── small_model.pkl
│   ├── large_model.pkl
│   ├── feature_names.pkl
│   ├── target_names.pkl
│   └── model_results.pkl
│
├── training_model.ipynb
│
└── README.md
```

---

## 🤖 Machine Learning Models

### Small Model

**Logistic Regression**

* Lightweight model
* Fast inference
* Low memory consumption
* Suitable for deployment on limited-resource systems

### Large Model

**Random Forest Classifier**

* Higher predictive capability
* Better generalization
* More robust predictions
* Suitable for performance-focused environments

---

## 📊 Dashboard Analytics

The dashboard includes several interactive visualizations:

### Prediction Confidence Gauge

Displays the confidence score of the predicted class.

### Feature Radar Chart

Visualizes input feature values for better understanding of the prediction.

### Probability Distribution Chart

Shows prediction probabilities across all available classes.

### Batch Prediction Analytics

Provides:

* Class distribution pie chart
* Class count bar chart

### Model Performance Dashboard

Provides:

* Accuracy comparison chart
* Model accuracy distribution
* Performance summary table

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/isaifullah/22_Streamlit_AI_App.git
```

Navigate to the project folder:

```bash
cd AI_Model_Deployment_Dashboard
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically open in your default browser.

---

## 📁 Batch Prediction Format

The uploaded CSV file must contain the following columns:

```text
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
```

Example:

```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
6.2,3.4,5.4,2.3
5.8,2.7,4.1,1.0
```

---

## 📈 Model Performance

The dashboard automatically displays:

* Model accuracy
* Performance comparison
* Prediction confidence
* Interactive analytics
* Batch prediction statistics

---

## 🌐 Deployment

This application can be deployed on:

### Streamlit Cloud

1. Push the project to GitHub.
2. Login to Streamlit Cloud.
3. Connect your GitHub repository.
4. Select `app.py`.
5. Deploy the application.

### Local Deployment

```bash
streamlit run app.py
```

---

## 🎓 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Machine Learning Model Training
* Model Evaluation
* Model Serialization
* Interactive Dashboard Development
* Streamlit Deployment
* Data Visualization
* Real-Time Prediction Systems
* Batch Processing Pipelines
* Model Comparison Techniques

---

## 👨‍💻 Author

**Saif Ullah**

Artificial Intelligence Graduate

Specializations:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Natural Language Processing
* Generative AI

---

## ⭐ Project Highlights

* Professional Streamlit Dashboard
* Interactive Plotly Analytics
* Multiple Model Support
* Manual and Batch Prediction
* Modern Dark-Themed UI
* Production-Ready Architecture
* Clean and Modular Code Structure
* Real-Time AI Predictions

---

### If you found this project useful, consider giving it a ⭐ on GitHub.
