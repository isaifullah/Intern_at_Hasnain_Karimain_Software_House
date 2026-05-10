# Sentiment Analysis on Reviews

This project is a **sentiment analysis system** that predicts whether a review is **Positive, Negative, Neutral, or Mixed**. The dataset was artificially created to include four labels, so it may have some inconsistencies, but the model works to the best of its training.

---

## Features

- Preprocessing includes:
  - Expanding contractions (e.g., "don't" → "do not")
  - Removing special characters and punctuation
  - Lowercasing
  - Tokenization
  - Lemmatization with POS tagging
  - Stopword removal
- Uses **CountVectorizer** and **TF-IDF** for text vectorization
- Implements three algorithms:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - Naive Bayes
- Selects the **best model** based on accuracy
- Can predict the sentiment of **new reviews** with preprocessing

---

## How to Use

1. Clone the repository:
```bash
git clone <[repository_url](https://github.com/isaifullah/Internship-at-HKSH-and-TC/tree/main/sentiment_classifier)>
cd <[sentiment_classifier]

