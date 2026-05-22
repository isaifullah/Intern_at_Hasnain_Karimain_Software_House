# 🚀 BERT-Based Twitter Sentiment Analysis

A production-ready Natural Language Processing (NLP) project that fine-tunes a pretrained **DistilBERT Transformer** model for multi-class Twitter sentiment classification using **Hugging Face Transformers** and **PyTorch**.

The model classifies tweets and reviews into:

- ✅ Positive
- ❌ Negative
- ⚪ Neutral

---

# 📌 Project Overview

This project demonstrates the complete end-to-end workflow of transformer fine-tuning for sentiment analysis. A pretrained `distilbert-base-uncased` model is fine-tuned on a custom Twitter sentiment dataset to perform accurate multi-class sentiment prediction.

The project includes:

- Data preprocessing and cleaning
- Label encoding
- Hugging Face dataset pipeline
- Transformer tokenization
- DistilBERT fine-tuning
- Performance evaluation
- Confusion matrix visualization
- Model saving and reloading
- Real-time custom inference

This project is designed using industry-standard NLP workflows and can later be deployed using Streamlit or FastAPI.

---

# ✨ Features

- Fine-tunes pretrained DistilBERT transformer
- Multi-class sentiment classification
- Hugging Face Trainer API integration
- GPU / CUDA support
- Automatic tokenization pipeline
- Classification report generation
- Confusion matrix visualization
- Model persistence and reloading
- Custom sentiment prediction
- Clean and modular notebook structure
- Production-ready NLP workflow

---

# 🧠 Model Information

### Pretrained Transformer Used

```python
distilbert-base-uncased
```

DistilBERT is a lightweight and faster version of BERT that retains strong language understanding capabilities while reducing model size and training time.

---

# 📂 Project Structure

```text
19_BERT_Sentiment_Project/
│
├── dataset/
│   └── sentiment_dataset.csv
│
├── figures/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── label_distribution.png
│   └── Text_length_distribution.png
│
├── models/
│   ├── distilbert_sentiment_model/
│   └── training_results/
│
├── app.py
├── bert_sentiment_analysis.ipynb
├── README.md
└── requirements.txt
```

---

# 📊 Dataset Format

The dataset must contain the following columns:

```text
text,sentiment
```

### Example

```text
I love this product,positive
The service was okay,neutral
This is terrible,negative
```

### Supported Labels

```text
positive
negative
neutral
```

---

# ⚙️ Workflow

The project follows the following NLP pipeline:

1. Load Twitter sentiment dataset
2. Clean and preprocess text data
3. Encode sentiment labels
4. Split dataset into training and testing sets
5. Convert data into Hugging Face Dataset format
6. Tokenize text using DistilBERT tokenizer
7. Fine-tune DistilBERT for sequence classification
8. Evaluate model performance
9. Generate classification report
10. Generate confusion matrix
11. Save and reload trained model
12. Perform custom sentiment prediction

---

# 📈 Evaluation Metrics

The model is evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

# 🖥️ Installation

Clone the repository:

```bash
git clone https://github.com/isaifullah/19_BERT_Sentiment_Project.git
```

Move into the project directory:

```bash
cd 19_BERT_Sentiment_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
bert_sentiment_analysis.ipynb
```

Run all notebook cells sequentially.

---

# 💾 Model Output

After training, the fine-tuned model is automatically saved inside:

```text
models/distilbert_sentiment_model/
```

The saved model can later be used for:

- Streamlit deployment
- FastAPI deployment
- Batch inference
- Production APIs
- Real-time prediction systems

---

# 🔮 Example Prediction

### Input

```python
predict_sentiment("I really love this product.")
```

### Output

```python
{
    "text": "I really love this product.",
    "sentiment": "positive",
    "confidence": 98.45
}
```

---

# 🛠️ Technologies Used

### Languages & Frameworks

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

### Data & ML Libraries

- Pandas
- NumPy
- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

---

# 🚀 Future Improvements

- Streamlit web deployment
- FastAPI REST API deployment
- Batch sentiment prediction
- Real-time Twitter sentiment analysis
- Model comparison with BERT and RoBERTa
- Hyperparameter optimization
- Attention score visualization
- Live dashboard integration

---

# 📌 Key Learning Outcomes

This project demonstrates practical understanding of:

- Transformer-based NLP
- Hugging Face ecosystem
- Transfer learning
- Fine-tuning pretrained models
- Multi-class text classification
- Tokenization pipelines
- Deep learning model evaluation
- Production-ready NLP workflows

---

# 👨‍💻 Author

Developed by **Saif Ullah** as part of an advanced NLP and Transformer Fine-Tuning project focused on industry-level sentiment analysis systems.

---