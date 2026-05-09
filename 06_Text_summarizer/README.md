# 🧠 Automatic Text Summarizer (Extractive + Abstractive)

## 📌 Project Overview
This project is an Automatic Text Summarization system that compares two NLP approaches: Extractive Summarization (selecting important sentences using TF-IDF) and Abstractive Summarization (generating new human-like summaries using a pretrained Transformer model - T5). It also provides a comparison between both methods to analyze readability, meaning retention, and summary quality.

---

## 🎯 Objectives
- Implement extractive summarization using TF-IDF
- Implement abstractive summarization using Transformers (T5-small)
- Compare both approaches on real text data
- Provide interactive user interface using Streamlit
- Evaluate summaries using ROUGE scores (optional)

---

## ⚙️ Features
- Extractive summarization using TF-IDF sentence ranking
- Abstractive summarization using Hugging Face Transformers
- Sentence preprocessing using NLTK and spaCy
- Summary length control (short / medium / long)
- Sentence highlighting for extractive results
- ROUGE score evaluation support
- Streamlit web interface for user interaction

---

## 🧰 Tech Stack
Python, Streamlit, NLTK, spaCy, scikit-learn, Transformers (Hugging Face), PyTorch, ROUGE-score

---

## 📁 Project Structure
text_summarizer/
├── app.py                  # Streamlit application
├── summary_output.txt       # (optional) backend functions
├── requirements.txt       # dependencies
├── bbc_news_dataset.csv   # dataset (optional)
└── text_summarizer.ipynb     # main code file

---

## 📊 Dataset
The project works with any long text dataset such as BBC News articles, blogs, Wikipedia content, or custom user input.

Required format:
- text → full article/content
- summary (optional) → reference summary for evaluation

---

## 🚀 How to Run

Install dependencies:
pip install -r requirements.txt

Install spaCy model:
python -m spacy download en_core_web_sm

Run Streamlit app:
streamlit run app.py

---

## 🧠 How It Works

### Extractive Summarization
- Split text into sentences
- Convert sentences into TF-IDF vectors
- Score sentence importance
- Select top-ranked sentences as summary

### Abstractive Summarization
- Input text is passed to pretrained T5 model
- Model generates new compressed sentences
- Output is more natural and human-like

---

## 📈 Comparison

Extractive:
- Selects original sentences
- More factual
- Less fluent

Abstractive:
- Generates new sentences
- More readable and human-like
- May slightly change meaning

---

## 📌 Example Output

Original Text:
Long article...

Extractive Summary:
Selected key sentences from the text.

Abstractive Summary:
Generated simplified version of the content.

---

## 📏 ROUGE Evaluation (Optional)
ROUGE is used to evaluate similarity between generated and reference summaries:
- ROUGE-1 → word overlap
- ROUGE-2 → phrase overlap
- ROUGE-L → sequence matching

---

## 🔥 Future Improvements
- Add BART summarization model
- PDF and DOCX file upload support
- Download summary as TXT/PDF
- ROUGE score visualization dashboard
- Multi-document summarization

---

## 👨‍💻 Author
This project demonstrates both traditional NLP techniques and modern Transformer-based summarization for educational and portfolio purposes.

---

## 📌 Note
This project is built for learning and comparison of extractive vs abstractive summarization techniques in NLP.