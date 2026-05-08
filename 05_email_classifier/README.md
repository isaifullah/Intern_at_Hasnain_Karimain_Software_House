# 📧 Email Classification System (AI Project)

## 🧠 Overview
This project is an AI-based Email Classification System that classifies emails into four categories:

- Spam 🚫  
- Important ⭐  
- Social 👥  
- Promotions 📢  

It uses Natural Language Processing (NLP) and Machine Learning to analyze email subjects and predict their category.

---

## 🚀 Features

### 🔹 Machine Learning Pipeline
- Text preprocessing (cleaning, lowercasing, stopword removal, lemmatization)
- TF-IDF vectorization
- Multiple ML models (Naive Bayes, SVM, Logistic Regression)
- Best model selection
- Model saved using pickle

### 🔹 Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix (Seaborn visualization)

### 🔹 Prediction System
- Single email prediction
- Confidence score for prediction

### 🔹 Batch Processing
- Upload CSV file
- Predict multiple emails at once
- Download results as CSV

### 🔹 Deployment
- Streamlit web application

---

## 📂 Project Structure

email-classifier/
├── app.py
├── email_classifier.ipynb
├── email_classifier_dataset.csv
├── models/
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   ├── label_encoder.pkl
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1. Clone repository
```bash
git clone https://github.com/isaifullah/email-classifier.git
cd email-classifier

first run this command
pip install -r requirements.txt

then this command
streamlit run app.py

📊 CSV Input Format

Your CSV file must contain a column named:

subject

Example:

subject
Win a free iPhone now
Meeting scheduled tomorrow
Your bank account update

📈 Output Example

Single Prediction:
Category: Spam
Confidence: 0.91

Batch Output:
subject | prediction | confidence
Win iPhone | Spam | 0.94
Meeting tomorrow | Important | 0.88

🔥 Future Improvements
Deep learning models (LSTM / BERT)
Email body classification
API integration for real emails
Dashboard analytics

👨‍💻 Autho
AI/ML project for email classification using NLP and machine learning.