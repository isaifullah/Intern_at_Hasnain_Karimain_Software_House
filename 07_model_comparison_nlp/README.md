# 📊 Sentiment Analysis Model Comparison Project

## 📌 Overview
This project is a complete Natural Language Processing (NLP) system that performs sentiment analysis using multiple machine learning models and compares their performance to select the best one. It is built using a Twitter sentiment dataset and supports four sentiment classes.

The models used in this project are:
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Logistic Regression

The system compares all models and automatically selects the best-performing model based on evaluation metrics.

---

## 🏷️ Sentiment Classes
The dataset contains 4 sentiment categories:
- Positive 😊
- Negative 😡
- Neutral 😐
- Irrelevant 🚫

---

## 🚀 Features
- Advanced NLP text preprocessing (cleaning, tokenization, lemmatization)
- TF-IDF vectorization for feature extraction
- Multi-class classification (4 labels)
- Multiple ML model training and comparison
- Evaluation using Accuracy, Precision, Recall, and F1-score
- Automatic best model selection
- GridSearchCV hyperparameter tuning
- Model saving using Pickle
- Streamlit web app for real-time prediction

---

## ⚙️ Project Workflow

1. Load dataset
2. Data cleaning (missing values and duplicates handling)
3. Text preprocessing:
   - Lowercasing
   - Removing URLs and special characters
   - Tokenization
   - Stopword removal
   - Lemmatization
4. Feature extraction using TF-IDF
5. Train-test split
6. Model training:
   - Naive Bayes
   - SVM
   - Logistic Regression
7. Model evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-score
8. Model comparison
9. Best model selection
10. Hyperparameter tuning using GridSearchCV
11. Save models using Pickle
12. Deploy using Streamlit

---

## 🧠 Example Prediction

Input:
"This product is amazing and works perfectly!"

Output:
Naive Bayes: Positive  
SVM: Positive  
Logistic Regression: Positive  

Best Model: Logistic Regression  
Best Model Prediction: Positive  

---

## 📦 Installation
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit


## ▶️ How to Run

### 1. Run in Jupyter Notebook
Open your `.ipynb` file and run all cells step by step:
- Execute data preprocessing cells
- Train all models
- Evaluate performance
- Save best model

---

### 2. Run Streamlit App (after training)
```bash
streamlit run app.py


👨‍💻 Author

Saif ullah