# Intent Classification System using Machine Learning

## Overview

The Intent Classification System is a real-world Natural Language Processing (NLP) project developed to automatically identify user intent from textual input using Machine Learning techniques. The system can understand and classify user requests such as greetings, food orders, weather queries, payment issues, account assistance, complaints, and many other conversational intents.

Intent classification is one of the core components used in modern:
- AI Chatbots
- Virtual Assistants
- Customer Support Systems
- Conversational AI Applications
- Intelligent Help Desk Solutions

This project combines advanced NLP preprocessing using spaCy, TF-IDF feature extraction, and multiple Machine Learning algorithms to build a scalable and production-ready intent detection system.

---

# Key Features

- Advanced NLP preprocessing using spaCy
- TF-IDF feature extraction
- Multiple Machine Learning model comparison
- Linear SVM based high-performance intent classifier
- Confidence score prediction
- Unknown intent handling
- Classification report and confusion matrix
- Data visualization and model comparison graphs
- Modular and reusable code structure
- Model saving and loading using Joblib
- Chatbot-style testing interface
- Integration-ready architecture for chatbot systems

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data handling and preprocessing |
| numpy | Numerical operations |
| scikit-learn | Machine Learning models and utilities |
| spaCy | Advanced NLP preprocessing |
| matplotlib | Data visualization |
| seaborn | Statistical plotting |
| joblib | Model serialization |
| Jupyter Notebook | Development environment |

---

# Dataset Information

The dataset contains user queries mapped to their corresponding intents.

## Example Intents

- greeting
- goodbye
- food_order
- weather_query
- payment_issue
- order_status
- account_help
- complaint
- product_search
- appointment_booking

## Dataset Columns

| Column | Description |
|---|---|
| text | Original user input |
| intent | Target intent label |
| clean_text | Preprocessed text |

---

# NLP Preprocessing Pipeline

The project uses advanced NLP preprocessing techniques to improve model performance and text understanding.

## Preprocessing Steps

- Text lowercasing
- URL removal
- HTML tag removal
- Currency normalization
- Number normalization
- Special character removal
- Stopword removal
- Lemmatization using spaCy
- Extra whitespace removal

---

# Feature Extraction

TF-IDF Vectorization is used to convert textual data into numerical vectors for Machine Learning models.

## TF-IDF Configuration

- Maximum Features: 5000
- N-Gram Range: (1, 2)

---

# Machine Learning Models

The following Machine Learning algorithms were trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM
4. Random Forest Classifier

---

# Best Performing Model

The best results were achieved using:

```python
CalibratedClassifierCV(LinearSVC())
```

This approach provides:
- High NLP classification accuracy
- Better scalability for larger datasets
- Probability estimation using `predict_proba()`
- Confidence score prediction
- Better generalization on sparse TF-IDF features

---

# Visualizations and Analysis

The project includes multiple visualization and evaluation techniques:

- Intent Distribution Graph
- Text Length Distribution
- Model Accuracy Comparison
- Confusion Matrix Heatmap
- Per-Class Performance Graph
- Prediction Confidence Visualization

All graphs are automatically saved inside the `figures/` directory.

---

# Production Features

## Machine Learning Features
- Multiple model comparison
- Train-test split
- Accuracy evaluation
- Classification report
- Confidence score prediction
- Unknown intent handling

## NLP Features
- spaCy-based preprocessing
- Lemmatization
- Stopword removal
- Text normalization

## Production Features
- Modular reusable functions
- Saved trained models
- Saved TF-IDF vectorizer
- Saved label encoder
- Joblib serialization
- Chatbot-style testing loop

---

# Project Structure

```text
intent_classifier/
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── model_comparison.csv
│
├── figures/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── intent_distribution.png
│   ├── per_class_performance.png
│   ├── prediction_confidence.png
│   └── text_length_distribution.png
│
├── intent_data.csv
├── intent_classifier.ipynb
├── requirements.txt
└── README.md
```

---

# Model Saving

The trained model and preprocessing artifacts are saved using Joblib for future deployment and chatbot integration.

## Saved Files

- `best_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

---

# Example Prediction

## User Input

```text
I want to order pizza
```

## Predicted Intent

```text
food_order
```

## Confidence Score

```text
98.7%
```

---

# Chatbot Integration

This intent classification system is designed to be integrated with context-aware chatbot systems. The trained model can be loaded into chatbot applications to automatically detect user intent and generate appropriate responses.

---

# Future Improvements

Possible future upgrades include:

- Transformer-based models (BERT)
- Streamlit or Flask deployment
- FastAPI backend deployment
- Voice-enabled chatbot integration
- Real-time API inference
- Database integration
- Multilingual intent classification
- Deep Learning based intent detection

---

# Conclusion

This project demonstrates a complete end-to-end NLP pipeline for Intent Classification using Machine Learning. It includes advanced preprocessing, feature extraction, model training, evaluation, visualization, and deployment-ready model saving techniques.

The system is modular, scalable, production-ready, and designed for real-world chatbot and conversational AI applications.

---

# Author

**Saif Ullah**