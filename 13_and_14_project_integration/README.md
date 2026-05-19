# 🤖 Hybrid Context-Aware AI Chatbot

## Overview

The Hybrid Context-Aware AI Chatbot is a real-world conversational AI system developed using Machine Learning, Natural Language Processing (NLP), and Transformer models. The chatbot intelligently detects user intent, maintains conversational context, handles follow-up queries, stores chat history, and provides dynamic responses through an interactive Streamlit interface.

This project combines the strengths of:
- Rule-Based NLP
- Machine Learning Intent Classification
- Transformer-Based Semantic Understanding
- Context Memory
- SQLite Database Storage
- Streamlit Web Application

The chatbot is designed to simulate modern AI-powered conversational systems used in:
- Customer Support Platforms
- AI Assistants
- Help Desk Systems
- Conversational AI Applications
- Intelligent Query Handling Systems

---

# 🚀 Key Features

## 🧠 Hybrid AI Architecture
The chatbot uses a professional multi-layer AI pipeline:

1. Rule-Based Intent Detection
2. Trained SVM Intent Classifier
3. Transformer Fallback Model
4. Context Memory System
5. SQLite Chat History

---

# ✨ Core Features

- Hybrid Intent Detection System
- Context-Aware Conversation Handling
- Transformer Fallback Architecture
- Real-Time Intent Prediction
- Confidence Score Prediction
- Dynamic Chat Responses
- SQLite Chat History Storage
- Streamlit Interactive Web Interface
- Voice Input Support
- Text-to-Speech Support
- Context Memory Handling
- Follow-Up Query Management
- Professional Modular Code Structure
- Joblib Model Serialization
- Production-Ready Architecture

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| pandas | Data handling |
| numpy | Numerical operations |
| scikit-learn | Machine Learning |
| spaCy | NLP preprocessing |
| Transformers | Transformer-based fallback model |
| PyTorch | Transformer backend |
| SQLite | Chat history database |
| Joblib | Model saving/loading |
| SpeechRecognition | Voice input |
| pyttsx3 | Text-to-speech |
| Jupyter Notebook | Model development |

---

# 🧠 Hybrid AI Architecture

The chatbot follows a layered hybrid architecture for better performance and intelligent conversation handling.

## Workflow

```text
User Message
      ↓
Rule-Based Intent Detection
      ↓
Trained Linear SVM Intent Classifier
      ↓
Low Confidence?
      ↓
Transformer Fallback Model
      ↓
Context Memory Handling
      ↓
Response Generation
      ↓
SQLite Chat Storage
      ↓
Streamlit User Interface
```

---

# 📌 Intent Categories

The chatbot supports multiple conversational intents including:

- Greeting
- Goodbye
- Thanks
- Food Ordering
- Weather Queries
- Product Search
- Order Tracking
- Payment Issues
- Account Help
- Password Reset
- Appointment Booking
- Cancellation Requests
- Return Requests
- Technical Support
- Complaint Handling
- Service Information

---

# 🧹 NLP Preprocessing Pipeline

The project uses advanced NLP preprocessing techniques including:

- Lowercasing
- URL Removal
- HTML Tag Removal
- Number Normalization
- Currency Normalization
- Special Character Removal
- Stopword Removal
- Lemmatization using spaCy
- Extra Whitespace Removal

---

# 🤖 Machine Learning Model

The primary intent classification model is based on:

```python
CalibratedClassifierCV(LinearSVC())
```

This model provides:
- High NLP classification accuracy
- Better scalability
- Fast inference speed
- Confidence score prediction
- Better handling of sparse TF-IDF features

---

# 🔥 Transformer Fallback Model

The chatbot uses:

```python
facebook/bart-large-mnli
```

as a transformer fallback model for:
- unseen queries
- semantic understanding
- generalized intent prediction
- low-confidence ML predictions

This creates a professional hybrid NLP system combining:
- classical Machine Learning
- modern transformer-based NLP

---

# 🧠 Context Memory System

The chatbot remembers conversational context and handles follow-up queries intelligently.

## Example

### User:
```text
Where is my package?
```

### Bot:
```text
Please provide your tracking number.
```

### User:
```text
12345
```

### Bot:
```text
Thank you. I will check your order status using this tracking number.
```

---

# 💾 SQLite Chat History

All chatbot conversations are automatically stored inside a SQLite database.

Stored Information:
- User Messages
- Bot Responses
- Predicted Intent
- Confidence Score
- Model Used
- Timestamp

---

# 🎤 Voice Features

The chatbot also supports:
- Voice Input
- Speech Recognition
- Text-to-Speech Responses

This makes the system closer to real-world virtual assistants.

---

# 🌐 Streamlit Web Application

The chatbot includes a professional Streamlit web interface with:
- Modern chat UI
- Interactive conversation flow
- Real-time predictions
- Confidence display
- Context handling
- Chat history viewer
- Sidebar controls

---

# 📂 Project Structure

```text
13_and_14_project_integration/
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── app.py
├── integrated_chatbot.ipynb
├── chat_history.db
├── requirements.txt
└── README.md
```

---

# ▶️ Installation

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

---

## Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

---

## Step 3: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
spacy
joblib
transformers
torch
speechrecognition
pyttsx3
pyaudio
```

---

# 📊 Example Conversation

## User
```text
Hello
```

## Bot
```text
Hello! How can I help you today?
```

---

## User
```text
I want to order pizza
```

## Bot
```text
Sure, I can help you with food ordering. What would you like to order?
```

---

## User
```text
Large chicken pizza
```

## Bot
```text
Your food order request has been noted: Large chicken pizza
```

---

# 🎯 Production Features

## Machine Learning Features
- Trained SVM Intent Classifier
- Transformer Fallback
- TF-IDF Vectorization
- Confidence Score Prediction
- Unknown Intent Handling

## NLP Features
- spaCy Preprocessing
- Lemmatization
- Stopword Removal
- Semantic Understanding

## Chatbot Features
- Context Awareness
- Follow-Up Handling
- Dynamic Responses
- Voice Support

## Deployment Features
- Streamlit Interface
- SQLite Storage
- Joblib Serialization
- Modular Architecture

---

# 🚀 Future Improvements

Possible future upgrades include:

- GPT-based conversational responses
- RAG (Retrieval-Augmented Generation)
- FastAPI deployment
- Docker containerization
- User authentication system
- Real-time API integration
- Multi-language chatbot support
- Deep Learning based intent classification
- Voice cloning and advanced speech synthesis

---

# 📌 Conclusion

This project demonstrates a complete real-world Hybrid Conversational AI System combining:
- Rule-Based NLP
- Machine Learning
- Transformer Models
- Context Memory
- Voice Processing
- Web Deployment

The chatbot is modular, scalable, production-ready, and designed for intelligent conversational AI applications.

---

# 👨‍💻 Author

## Saif Ullah

AI Engineer | Machine Learning Enthusiast | NLP Developer