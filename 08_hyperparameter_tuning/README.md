# 📰 Fake News Detection using Machine Learning + Hyperparameter Tuning

## 📌 Project Overview
This project is an NLP-based Fake News Detection system that classifies news articles as Fake or Real using Machine Learning techniques. It includes advanced text preprocessing, TF-IDF feature extraction, and hyperparameter tuning using GridSearchCV and RandomizedSearchCV to improve model performance.

---

## 🚀 Features
- Advanced NLP preprocessing (lowercasing, stopword removal, lemmatization, POS tagging)
- TF-IDF vectorization for feature extraction
- Baseline Logistic Regression model
- Hyperparameter tuning using:
  - GridSearchCV
  - RandomizedSearchCV
- Model performance comparison
- Confusion matrix visualization
- Real-time prediction system
- Model saving using Pickle

---

## 🧠 Machine Learning Model
- Logistic Regression (best performing model after tuning)

---

## 📊 Results

| Model | Accuracy |
|------|--------|
| Before Tuning | ~82% |
| After GridSearchCV Tuning | ~89% |

**Best Parameters Example:**
```python
{'C': 10, 'solver': 'liblinear'}
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
contractions
pickle-mixin
```

---

## 📂 Dataset
- Fake News Dataset (CSV format)
- Columns:
  - Text (news content)
  - Label (Fake / Real)

---

## 🔄 Workflow
1. Load dataset
2. Data preprocessing (cleaning, tokenization, lemmatization)
3. TF-IDF feature extraction
4. Train baseline Logistic Regression model
5. Hyperparameter tuning:
   - GridSearchCV
   - RandomizedSearchCV
6. Compare performance
7. Save best model using Pickle
8. Build prediction system

---

## 🧪 How to Run

### Train Model
```bash
python train.py
```

### Run Prediction System
```bash
python app.py
```

---

## 🔮 Prediction Example

```
Enter text: Government announces new policy for citizens

Prediction (Tuned Model): Real News
```

---

## 💾 Model Files
- best_model.pkl
- label_encoder.pkl

---

## 📈 Future Improvements
- Add multiple ML models (SVM, Naive Bayes)
- Deploy using Streamlit or Flask
- Add deep learning (LSTM / BERT)
- Add confidence score for predictions

---

## 👨‍💻 Author
Machine Learning + NLP project focused on hyperparameter tuning and real-world text classification.