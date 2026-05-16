# 🤖 Nexus Support AI — Context-Aware Customer Support Chatbot

<div align="center">

### *An Industry-Level AI Customer Support Assistant Powered by Transformer Models*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![SQLite](https://img.shields.io/badge/Database-SQLite-green.svg)
![License](https://img.shields.io/badge/License-Educational-orange.svg)

**Transformer-Based | Context-Aware | Voice Enabled | Production-Style Architecture**

</div>

---

# 📌 Project Overview

**Nexus Support AI** is a modern **context-aware AI chatbot** designed to simulate real-world customer support systems used in intelligent applications and enterprise support platforms.

Unlike traditional keyword-based chatbots, this system uses a **pretrained transformer model** for semantic intent understanding, enabling it to handle natural conversations more intelligently.

The chatbot is capable of:

- Understanding customer queries
- Detecting user intent
- Maintaining conversation context
- Handling follow-up interactions
- Providing intelligent support responses
- Supporting voice-based interaction
- Storing chat history in a database

---

# 🚀 Key Features

## ✅ AI-Powered Intent Detection

Uses the pretrained Hugging Face transformer model:

```python
facebook/bart-large-mnli
```

for **zero-shot intent classification**.

This allows the chatbot to:

- Understand natural language semantically
- Handle unseen user queries
- Work with small datasets
- Produce more intelligent predictions

---

## ✅ Context-Aware Conversations

The chatbot maintains memory of previous interactions.

### Example:

```text
User: Where is my order?
Bot: Please provide your tracking ID.

User: 12345
Bot: Thank you. I will check your order status.
```

---

## ✅ Voice Chat Support

The chatbot supports:

- 🎤 Speech-to-Text
- 🔊 Text-to-Speech

Users can communicate with the chatbot using voice commands.

---

## ✅ Professional Streamlit Interface

Features a modern UI with:

- Gradient-based professional design
- Live prediction dashboard
- Context monitoring
- Chat history tracking
- Real-time interaction
- Quick action buttons

---

## ✅ SQLite Database Integration

All conversations are stored in SQLite including:

- User messages
- Bot responses
- Predicted intents
- Confidence scores
- Input mode
- Timestamps
- Model information

---

# 🧠 Supported Intents

The chatbot supports multiple customer support operations:

| Intent | Description |
|---|---|
| `account_help` | General account assistance |
| `password_reset` | Password reset support |
| `cancellation` | Subscription cancellation |
| `order_status` | Order tracking |
| `return_request` | Product return requests |
| `payment_update` | Payment and billing help |
| `technical_support` | Technical issue assistance |
| `business_hours` | Company timings |
| `service_info` | Service information |

---

# 🏗️ System Architecture

```text
User Input
    ↓
Rule-Based NLP Layer
    ↓
Transformer Intent Detection
    ↓
Context Memory Engine
    ↓
Response Generator
    ↓
Voice Output (Optional)
    ↓
SQLite Database Storage
    ↓
Professional Streamlit UI
```

---

# ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Web application |
| Hugging Face Transformers | Pretrained NLP model |
| SQLite | Database storage |
| Pandas | Data handling |
| SpeechRecognition | Speech-to-text |
| pyttsx3 | Text-to-speech |
| Regular Expressions | NLP preprocessing |

---

# 📂 Project Structure

```text
13_AI_chatbot/
│
├── dataset/
│   └── customers.csv
│
├── app.py
├── chatbot.ipynb
├── chat_history.db
├── requirements.txt
└── README.md
```

---

# 🔥 Advanced AI Features

## ✅ Zero-Shot Intent Classification

The chatbot uses transformer-based zero-shot learning instead of traditional supervised classification.

### Benefits

- No large dataset required
- Better semantic understanding
- Handles dynamic queries
- More scalable architecture

---

## ✅ Rule-Based NLP Layer

A lightweight NLP layer handles:

- Greetings
- Short messages
- Common commands
- Quick responses

This improves response reliability and reduces transformer overhead.

---

## ✅ Context Memory

The chatbot stores temporary conversation state to handle follow-up questions naturally.

---

## ✅ Voice Interaction

Users can communicate using voice input.

### Supported Features

- Microphone voice input
- AI-generated voice responses
- Real-time speech recognition

---

# 💬 Example Conversations

## Greeting Example

```text
User: Hi
Bot: Hello! How can I help you today?
```

---

## Password Reset Example

```text
User: I forgot my password
Bot: You can reset your password using the Forgot Password option.
```

---

## Order Tracking Example

```text
User: Where is my package?
Bot: Please provide your order ID.

User: 12345
Bot: Thank you. I will check your order status using ID: 12345
```

---

## Cancellation Example

```text
User: Cancel my subscription
Bot: Please confirm if you want to continue.

User: yes
Bot: Your cancellation request has been confirmed and submitted.
```

---

# 📊 Streamlit Dashboard Features

The application dashboard provides:

- Live prediction information
- Intent confidence monitoring
- Context tracking
- Chat history viewer
- Voice interaction controls
- Quick support actions

---

# 🗄️ Database System

SQLite database stores:

```text
- User messages
- AI responses
- Predicted intents
- Confidence scores
- Input mode
- Timestamps
- Model information
```

---

# ▶️ Installation Guide

## Step 1: Clone Repository

```bash
git clone <your_repository_link>
cd 13_AI_chatbot
```

---

## Step 2: Create Virtual Environment

### Using Conda

```bash
conda create -n chatbot_env python=3.10 -y
conda activate chatbot_env
```

---

## Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

---

## Step 4: Install Voice Dependencies

```bash
pip install speechrecognition pyttsx3 pyaudio
```

### Windows Fix for PyAudio

```bash
pip install pipwin
pipwin install pyaudio
```

---

## Step 5: Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```txt
streamlit
pandas
transformers
torch
sentencepiece
speechrecognition
pyttsx3
pyaudio
scikit-learn
nltk
numpy
```

---

# 🧪 Pretrained Transformer Model

## Model Used

```text
facebook/bart-large-mnli
```

### Why This Model?

- Strong NLP understanding
- Excellent semantic classification
- Works without large training datasets
- Handles unseen queries effectively
- Industry-level performance

---

# 📈 Future Improvements

Potential future upgrades include:

- Multi-language support
- OpenAI/Gemini integration
- RAG-based document chatbot
- Vector database integration
- Customer analytics dashboard
- Sentiment analysis
- Live API integrations
- User authentication system
- Deployment on cloud platforms

---

# 👨‍💻 Author

## Khalid Saifullah

Bachelor of Science in Artificial Intelligence

### Interests

- Natural Language Processing
- Generative AI
- Deep Learning
- Computer Vision
- Intelligent AI Systems

---

# 📜 License

This project is developed for:

- Educational purposes
- AI learning
- NLP experimentation
- Research and portfolio development

---

# ⭐ Final Notes

This project demonstrates:

- Industry-level AI chatbot architecture
- Transformer-powered NLP systems
- Context-aware conversations
- Voice-enabled AI assistants
- Professional Streamlit deployment
- Database integration
- Real-world customer support simulation

---

<div align="center">

### 🚀 Nexus Support AI — Intelligent Conversations Powered by Transformers

</div>