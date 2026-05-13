# 🌍 Language Detection Model (Multilingual NLP Project)

A machine learning-based language detection system that identifies the language of a given text input. It supports multiple languages such as English, Urdu, Spanish, French, and German using character-level TF-IDF features and classical ML models.

The system also includes a Streamlit-based CLI-style web application, batch prediction system, and comparison with the langdetect library.

---

# 🚀 Features

- Detect language from text input
- Supports multiple languages (expandable)
- Character-level TF-IDF (2–4 n-grams)
- Automatic best model selection
- Top-2 language prediction with probabilities
- Batch prediction using CSV upload
- Comparison with langdetect library
- Streamlit CLI-style interface
- Downloadable prediction results

---

# 🧠 Machine Learning Approach

The model uses character-level TF-IDF vectorization, which is highly effective for multilingual and short-text classification.

Models used:
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

The best model is selected automatically based on accuracy.

---

# 📊 Dataset

The dataset contains multilingual text samples labeled with their respective languages.

Example:

Text | Language
----- | --------
How are you? | English
آپ کیسے ہیں؟ | Urdu
¿Como estás? | Spanish
Comment ça va? | French

---

# ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Langdetect
- Matplotlib
- Seaborn

---

# 📁 Project Structure

Language_Detection_Model/
│── language_detection_dataset.csv
│── language_model.pkl
│── tfidf_vectorizer.pkl
│── app.py
│── train.ipynb
│── requirements.txt
│── README.md

---

# 🚀 Installation

## 1. Clone Repository
git clone https://github.com/your-username/language-detection-model.git
cd language-detection-model

## 2. Install Dependencies
pip install -r requirements.txt

---

# ▶️ Run Application

streamlit run app.py

---

# 🖥️ Streamlit CLI Commands

Inside the app, use these commands:

predict   → Single language prediction  
top2      → Top 2 predicted languages  
compare   → Compare with langdetect  
batch     → Upload CSV for batch prediction  
help      → Show help menu  

---

# 📂 Batch Prediction Format

Upload a CSV file with this format:

Text
How are you?
Bonjour comment allez vous
¿Como estás?

Output includes:
- Predicted Language
- Top 1 Confidence
- Second Language Prediction
- Langdetect Result

---

# 📈 Model Performance

- High accuracy using character n-grams
- Works well on short text inputs
- Robust across multiple languages and scripts

---

# 🔍 Comparison with langdetect

The system compares predictions with Python's langdetect library to validate model performance.

---

# 💾 Saved Files

- language_model.pkl → trained ML model
- tfidf_vectorizer.pkl → feature extractor

---

# ⚠️ Important Notes

- LabelEncoder is NOT required during inference
- Model directly outputs language labels
- Ensure CSV contains a "Text" column for batch mode

---

# 📌 Future Improvements

- Add deep learning models (LSTM / Transformers)
- Expand to 50+ languages
- Deploy on cloud (Streamlit Cloud / HuggingFace)
- Add REST API using FastAPI
- Add confidence threshold for uncertainty handling

---

# 👨‍💻 Author

Developed as a multilingual NLP project using classical machine learning techniques for language detection.

---

# ⭐ Support

If you like this project, give it a star on GitHub ⭐