# 📰 Fake News Detection System using Machine Learning

## 📌 Project Overview
This project is a Machine Learning-based Fake News Detection System that classifies news articles as Fake 🛑 or Real ✅ using Natural Language Processing (NLP) techniques. It uses TF-IDF for feature extraction and multiple ML models for classification, with a Streamlit web app for real-time predictions.

## 🎯 Objective
To build an intelligent system that can automatically detect fake news using text classification techniques.

## ⚙️ Features
- Data preprocessing (cleaning, stopwords removal, lemmatization)
- TF-IDF feature extraction
- Multiple ML models comparison:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Model evaluation (Accuracy, Precision, Recall, F1-score)
- Confusion matrix visualization
- Feature importance analysis
- Batch prediction using CSV files
- Real-time prediction using Streamlit
- Model saving using Joblib

## 🧠 Machine Learning Pipeline
1. Data Collection (Fake & Real News dataset)
2. Text Preprocessing:
   - Lowercasing
   - Removing punctuation & special characters
   - Stopword removal
   - Lemmatization
3. Feature Engineering (TF-IDF Vectorization)
4. Model Training
5. Model Evaluation
6. Deployment using Streamlit

## 📊 Best Model
Logistic Regression was selected as the final model due to:
- Strong generalization performance
- Efficiency with high-dimensional sparse data
- Stability compared to other models

## 📁 Project Structure
Fake-News-Detection/
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── news_dataset.csv
├── fake_news.ipynb
└── README.md

## 🚀 Installation & Setup

### Clone the repository
git clone https://github.com/isaifullah/fake-news-detection.git
cd fake-news-detection

### Install dependencies
pip install -r requirements.txt

### Run Jupyter Notebook
jupyter notebook

### Run Streamlit App
streamlit run app.py

## 🧪 Example Prediction

Input:
"Breaking: Scientists confirm water found on Mars"

Output:
Prediction: Real News ✅  
Confidence: 0.98

## 📈 Model Performance

Logistic Regression → 0.9919  
Naive Bayes → 0.9590  
Random Forest → 0.9989  

## ⚠️ Important Note
Although Random Forest achieved slightly higher accuracy, Logistic Regression was selected due to better generalization for NLP tasks and lower risk of overfitting.

## 🔥 Future Improvements
- Implement BERT / Transformer models for higher accuracy
- Real-time news scraping from APIs
- Cloud deployment (AWS / Render / HuggingFace Spaces)
- Improve dataset diversity

## 👨‍💻 Author
Saifullah  
BS Artificial Intelligence  
Passionate about AI, NLP, and Machine Learning

## ⭐ Support
If you like this project, give it a ⭐ on GitHub.