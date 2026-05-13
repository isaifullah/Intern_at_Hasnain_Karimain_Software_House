# 📧 Email Classification System using NLP and Machine Learning

An AI-powered email classification system that automatically classifies emails into four categories:

- 🚫 Spam
- ⭐ Important
- 👥 Social
- 📢 Promotions

The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and Machine Learning algorithms to automatically analyze and classify email text.

---

# 🧠 Project Overview

This project was developed to automate email organization and reduce manual email filtering. The system processes email text, extracts meaningful features, and predicts the most appropriate email category using trained machine learning models.

The project includes:

- Advanced NLP preprocessing
- TF-IDF feature engineering
- Multiple machine learning models
- Hyperparameter tuning
- Model comparison and evaluation
- Prediction system with confidence scores
- Batch CSV prediction system
- Interactive CLI menu
- Visualization of model performance
- Model saving using Pickle

---

# 📌 Categories

| Category | Description |
|---|---|
| 🚫 Spam | Fraudulent emails, scams, fake offers, phishing |
| ⭐ Important | Meetings, work updates, deadlines, approvals |
| 👥 Social | Social media notifications and interactions |
| 📢 Promotions | Marketing emails, discounts, offers, newsletters |

---

# 📊 Dataset Information

| Property | Details |
|---|---|
| Dataset Source | Kaggle Email Spam Classification Dataset |
| Original Dataset Size | 1M+ emails |
| Used Dataset | 400,000 emails |
| Classes Used | Spam, Important, Social, Promotions |
| Dataset Link | https://www.kaggle.com/datasets/sharmajicoder/email-spam-classification |

The original dataset contained six categories, but only four categories were used because of limited storage and computational resources.

---

# 🚀 Features

## ✅ NLP Preprocessing Pipeline

The project uses advanced preprocessing techniques:

- Lowercasing
- URL removal
- HTML tag removal
- Currency normalization
- Number normalization
- Special character removal
- Tokenization using NLTK
- POS tagging
- Lemmatization
- Stopword removal
- Short token filtering
- Extra space cleaning

---

## ✅ Feature Engineering

TF-IDF Vectorization was applied using:

- Word-level features
- Unigrams and bigrams `(1,2)`
- Maximum 15,000 features

---

## ✅ Machine Learning Models

The following machine learning models were trained and compared:

- Logistic Regression
- Naive Bayes
- Linear SVM

Hyperparameter tuning was performed using:

- `RandomizedSearchCV`

---

## ✅ Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curves
- Per-Class Performance Metrics

---

## ✅ Prediction System

The prediction system supports:

- Single email prediction
- Confidence scores
- Probability breakdown
- Short text handling
- Long text handling
- Uncertain prediction handling

---

## ✅ Batch Prediction System

The project supports batch prediction using CSV files:

- Input CSV file processing
- Automatic prediction generation
- Output CSV generation
- Progress tracking
- Summary statistics

---

## ✅ Interactive CLI Menu

The project includes a complete command-line interface with:

1. Single Email Prediction
2. Batch CSV Prediction
3. View Model Performance
4. View Top Keywords
5. Change Confidence Threshold
6. Test Sample Emails
7. Exit System

---

# 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | pandas, numpy |
| NLP | nltk |
| Machine Learning | scikit-learn |
| Visualization | matplotlib, seaborn |
| Model Saving | pickle |
| Environment | Jupyter Notebook |

---

# 📂 Project Structure

```text
email-classifier/
│
├── email_classifier.ipynb
├── email_classifier_dataset.csv
├── README.md
│
├── models/
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
│
├── figures/
│   ├── label_distribution.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── per_class_performance.png
│   ├── roc_curves.png
│   └── text_length_distribution.png
│
├── batch_input.csv
├── batch_output.csv
│
└── requirements.txt
```

---

# ⚙️ Installation Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/isaifullah/email-classifier.git
cd email-classifier
```

---

## Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

---

## Step 3: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
email_classifier.ipynb
```

Run all notebook cells sequentially.

---

# 🔬 NLP Preprocessing Pipeline

The preprocessing pipeline includes:

## Step 1: Lowercasing

Convert all text into lowercase.

Example:

```text
HELLO WORLD → hello world
```

---

## Step 2: Remove URLs

Example:

```text
https://example.com → removed
```

---

## Step 3: Remove HTML Tags

Example:

```html
<p>Hello</p> → Hello
```

---

## Step 4: Normalize Currency

Example:

```text
$500 → money
```

---

## Step 5: Normalize Numbers

Example:

```text
12345 → number
```

---

## Step 6: Remove Special Characters

Example:

```text
hello!!! → hello
```

---

## Step 7: Tokenization

Split text into words using NLTK.

---

## Step 8: POS Tagging

Part-of-speech tagging for better lemmatization.

---

## Step 9: Lemmatization

Example:

```text
running → run
```

---

## Step 10: Stopword Removal

Example:

```text
the, is, and → removed
```

---

# ⚙️ Feature Engineering

TF-IDF Vectorization settings:

```python
TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=15000,
    max_df=0.90,
    min_df=3,
    sublinear_tf=True,
    strip_accents='unicode'
)
```

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

- Balanced class weights
- Hyperparameter tuning using different `C` values

---

## 2. Naive Bayes

- Multinomial Naive Bayes
- Alpha tuning

---

## 3. Linear SVM

- Linear Support Vector Machine
- Wrapped using `CalibratedClassifierCV` for probability predictions

---

# 📈 Model Evaluation

The following metrics were used:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correct positive predictions |
| Recall | Ability to find all relevant samples |
| F1-Score | Balance between precision and recall |

---

# 📊 Visualizations Generated

The project automatically generates:

| File Name | Description |
|---|---|
| label_distribution.png | Distribution of categories |
| confusion_matrix.png | Prediction vs actual labels |
| model_comparison.png | Comparison of all models |
| per_class_performance.png | Precision, recall, F1 per class |
| roc_curves.png | ROC curves with AUC scores |
| text_length_distribution.png | Distribution of email lengths |

---

# 🔑 Top Keywords Per Category

The project extracts top important words for each category based on model coefficients.

Example:

| Category | Example Keywords |
|---|---|
| Spam | winner, click, free, money |
| Important | meeting, report, project |
| Social | comment, friend, photo |
| Promotions | sale, discount, offer |

---

# 📬 Single Email Prediction Example

## Input

```text
WINNER! You won $1000! Click here now!
```

## Output

```text
🚫 Spam
Confidence: 94%
```

---

# 📁 Batch Prediction System

## Input CSV Format

Your CSV file must contain a column named:

```text
text
```

Example:

```csv
text
Win a free iPhone now
Meeting scheduled tomorrow
Your friend liked your post
```

---

## Output CSV Format

```csv
text,prediction,confidence
Win a free iPhone now,Spam,0.94
Meeting tomorrow,Important,0.88
```

---

# 🖥️ CLI Menu System

The project contains an interactive CLI menu:

```text
1. Single Email Prediction
2. Batch CSV Prediction
3. View Model Performance
4. View Top Keywords
5. Change Confidence Threshold
6. Test Sample Emails
7. Exit
```

---

# ⚠️ Edge Case Handling

The prediction system handles:

| Scenario | Handling |
|---|---|
| Empty input | Returns ERROR |
| Very short text | Returns UNCERTAIN |
| Very long text | Automatically truncated |
| Low confidence prediction | Marked as UNCERTAIN |

---

# 💾 Model Saving

The project automatically saves:

- Trained model
- TF-IDF vectorizer
- Label encoder

Saved files:

```text
models/best_model.pkl
models/vectorizer.pkl
models/label_encoder.pkl
```

---

# 🧪 Sample Predictions

| Email Text | Predicted Category |
|---|---|
| Win a free iPhone now! | Spam |
| Team meeting tomorrow at 10 AM | Important |
| Your friend commented on your photo | Social |
| Flat 50% OFF on all products | Promotions |

---

# 💡 Future Improvements

Possible future enhancements:

- Deep Learning models (LSTM, BERT)
- Email body classification
- Real-time API integration
- Streamlit web deployment
- Dashboard analytics
- Multi-language support
- Gmail/Outlook integration

---

# 👨‍💻 Author
 This project is created by isaifullah, you can search on github
AI/ML Project — Email Classification System using NLP and Machine Learning.

This project demonstrates:

- Natural Language Processing
- Multi-class classification
- Feature engineering
- Hyperparameter tuning
- Model evaluation
- Production-style prediction systems

---

# 📄 License

This project is for educational and learning purposes.

---