# Domain-Specific Q&A Chatbot using TF-IDF and Cosine Similarity

## Project Overview

This project is a domain-specific Q&A chatbot built using TF-IDF Vectorization and Cosine Similarity.

The chatbot answers user questions by searching a prepared question-answer dataset and returning the most relevant answer. It is a retrieval-based chatbot, which means it does not generate new answers. Instead, it retrieves the best matching answer from the available dataset.

This project is designed for AI and Data Science educational question-answering. The dataset contains questions and answers related to Artificial Intelligence, Machine Learning, Deep Learning, NLP, Computer Vision, Data Science, and related technical topics.

---

# Objective

The main objective of this project is to build a simple, professional, and reusable chatbot system that can:

- Load a domain-specific Q&A dataset
- Clean and preprocess user questions
- Convert questions into TF-IDF vectors
- Compare user input with stored questions using Cosine Similarity
- Return the best matching answer
- Display matched question and similarity score
- Handle unknown questions using a similarity threshold
- Provide a professional Streamlit chat interface

---

# Dataset Format

The dataset must be in CSV format and should contain the following columns:

```csv
domain,question,answer
education,What is artificial intelligence?,Artificial intelligence is the simulation of human intelligence by machines.
education,What is machine learning?,Machine learning is a subset of AI that allows systems to learn from data.
education,What is deep learning?,Deep learning is a branch of machine learning based on neural networks.
```

In this project, the `domain` column is filled with:

```text
education
```

because the dataset contains AI and Data Science-related educational content.

---

# Project Structure

```text
Domain-Specific-QA-Chatbot/
│
├── data/
│   └── qa_dataset.csv
│
├── models/
│   ├── clean_qa_dataset.csv
│   ├── tfidf_vectorizer.pkl
│   └── question_vectors.pkl
│
├── figures/
│   ├── domain_distribution.png
│   ├── question_length_distribution.png
│   ├── answer_length_distribution.png
│   ├── similarity_scores.png
│   └── top_tfidf_words.png
│
├── notebook/
│   └── domain_specific_qa_chatbot.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Matplotlib
- Seaborn
- Streamlit

---

# Features

- Domain-specific Q&A chatbot
- TF-IDF feature extraction
- N-gram support
- Cosine similarity matching
- Similarity threshold
- Unknown question handling
- Top matched questions
- Matched question display
- Similarity score display
- Domain-based filtering
- Dataset cleaning
- Duplicate removal
- Missing value handling
- Dataset visualization
- TF-IDF vocabulary visualization
- Model saving and loading
- Professional Streamlit chat interface
- Chat history
- CSV dataset upload option
- Modular and reusable code structure

---

# How the Chatbot Works

1. The dataset is loaded from a CSV file.
2. The dataset is cleaned by removing missing values and duplicate rows.
3. Questions are preprocessed by converting text to lowercase and removing unnecessary characters.
4. TF-IDF converts cleaned questions into numerical vectors.
5. The user question is also converted into a TF-IDF vector.
6. Cosine Similarity compares the user question with all dataset questions.
7. The chatbot selects the most similar question.
8. If the similarity score is above the threshold, the chatbot returns the matched answer.
9. If the similarity score is below the threshold, the chatbot returns an unknown response.

---

# Important Limitation

This chatbot does not generate new answers like large language models.

It only retrieves the most relevant answer from the dataset. Therefore, the quality of the chatbot depends heavily on the quality, size, and coverage of the Q&A dataset.

---

# Installation

First, clone the repository or download the project files.

Then install the required libraries:

```bash
pip install -r requirements.txt
```

---

# Run the Jupyter Notebook

Open the notebook file:

```text
notebook/domain_specific_qa_chatbot.ipynb
```

Run all cells step by step to:

- Load the dataset
- Clean the dataset
- Preprocess text
- Build the TF-IDF model
- Test the chatbot
- Save model files
- Generate visualizations

---

# Run the Streamlit App

After saving the model files from the notebook, run:

```bash
streamlit run app.py
```

---

# Streamlit App Features

The Streamlit app provides:

- Professional chat-style interface
- User and bot message bubbles
- Chat history
- Similarity threshold control
- Top-k matched questions
- Domain selection
- Dataset upload option
- Matched question display
- Similarity score display
- Clear chat history button

---

# Example Conversation

```text
User: What is machine learning?

Bot: Machine learning is a subset of AI that allows systems to learn from data.

Matched Question: What is machine learning?
Similarity Score: 0.92
```

```text
User: Tell me about football

Bot: Sorry, I don't understand your question. Please ask something related to AI, Machine Learning, Deep Learning, NLP, or Data Science.
```

---

# Implemented Modules

1. Dataset loading
2. Dataset exploration
3. Dataset cleaning
4. Text preprocessing
5. TF-IDF feature extraction
6. Cosine similarity matching
7. Top-k question retrieval
8. Similarity threshold handling
9. Unknown question handling
10. Model saving
11. Model loading
12. Chatbot-style testing
13. Streamlit deployment

---

# Future Improvements

- Add semantic embeddings
- Add transformer-based sentence similarity
- Add database support
- Add user feedback system
- Add admin dashboard for adding Q&A pairs
- Add voice input
- Add multilingual support
- Add API deployment using FastAPI

---

# Conclusion

This project demonstrates how a domain-specific retrieval-based chatbot can be built using traditional NLP techniques. It is lightweight, explainable, easy to modify, and suitable for FAQ systems, educational assistants, and customer support applications.

---

# Author

**Saif ullah**  

GitHub: https://github.com/isaifullah

---

# License

This project is developed for educational and learning purposes.