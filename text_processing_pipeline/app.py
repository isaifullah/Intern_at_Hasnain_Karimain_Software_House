import streamlit as st
import pandas as pd
import spacy
import re
import string
from nltk.stem import PorterStemmer

# =========================
# LOAD MODELS
# =========================
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

# =========================
# YOUR CONTRACTIONS DICT
# =========================
# contractions dictionary, we will use this to expand contractions in the text, like "don't" to "do not", "can't" to "cannot"
contractions_dict = {
    "I'm": "I am",
    "I'm'a": "I am about to",
    "I'm'o": "I am going to",
    "I've": "I have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'd": "I would",
    "I'd've": "I would have",
    "Whatcha": "What are you",
    "amn't": "am not",
    "ain't": "are not",
    "aren't": "are not",
    "'cause": "because",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "daren't": "dare not",
    "daresn't": "dare not",
    "dasn't": "dare not",
    "didn't": "did not",
    "didn’t": "did not",
    "don't": "do not",
    "don’t": "do not",
    "doesn't": "does not",
    "e'er": "ever",
    "everyone's": "everyone is",
    "finna": "fixing to",
    "gimme": "give me",
    "gon't": "go not",
    "gonna": "going to",
    "gotta": "got to",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he've": "he have",
    "he's": "he is",
    "he'll": "he will",
    "he'll've": "he will have",
    "he'd": "he would",
    "he'd've": "he would have",
    "here's": "here is",
    "how're": "how are",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how's": "how is",
    "how'll": "how will",
    "isn't": "is not",
    "it's": "it is",
    "'tis": "it is",
    "'twas": "it was",
    "it'll": "it will",
    "it'll've": "it will have",
    "it'd": "it would",
    "it'd've": "it would have",
    "kinda": "kind of",
    "let's": "let us",
    "luv": "love",
    "ma'am": "madam",
    "may've": "may have",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "ne'er": "never",
    "o'": "of",
    "o'clock": "of the clock",
    "ol'": "old",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "o'er": "over",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shalln't": "shall not",
    "shan't've": "shall not have",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "she'd've": "she would have",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "somebody's": "somebody is",
    "someone's": "someone is",
    "something's": "something is",
    "sux": "sucks",
    "that're": "that are",
    "that's": "that is",
    "that'll": "that will",
    "that'd": "that would",
    "that'd've": "that would have",
    "'em": "them",
    "there're": "there are",
    "there's": "there is",
    "there'll": "there will",
    "there'd": "there would",
    "there'd've": "there would have",
    "these're": "these are",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they'd": "they would",
    "they'd've": "they would have",
    "this's": "this is",
    "this'll": "this will",
    "this'd": "this would",
    "those're": "those are",
    "to've": "to have",
    "wanna": "want to",
    "wasn't": "was not",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we'd": "we would",
    "we'd've": "we would have",
    "weren't": "were not",
    "what're": "what are",
    "what'd": "what did",
    "what've": "what have",
    "what's": "what is",
    "what'll": "what will",
    "what'll've": "what will have",
    "when've": "when have",
    "when's": "when is",
    "where're": "where are",
    "where'd": "where did",
    "where've": "where have",
    "where's": "where is",
    "which's": "which is",
    "who're": "who are",
    "who've": "who have",
    "who's": "who is",
    "who'll": "who will",
    "who'll've": "who will have",
    "who'd": "who would",
    "who'd've": "who would have",
    "why're": "why are",
    "why'd": "why did",
    "why've": "why have",
    "why's": "why is",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "you're": "you are",
    "you've": "you have",
    "you'll've": "you shall have",
    "you'll": "you will",
    "you'd": "you would",
    "you'd've": "you would have",

    "to cause": "to cause",
    "will cause": "will cause",
    "should cause": "should cause",
    "would cause": "would cause",
    "can cause": "can cause",
    "could cause": "could cause",
    "must cause": "must cause",
    "might cause": "might cause",
    "shall cause": "shall cause",
    "may cause": "may cause"
}

# =========================
# FUNCTIONS (PIPELINE)
# =========================

def expand_contractions(text):
    for word, replacement in contractions_dict.items():
        text = text.replace(word, replacement)
    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def custom_tokenizer(text):
    return re.findall(r'\b\w+\b', text)


def remove_stopwords(tokens):
    stop_words = set([
        "is", "the", "and", "a", "an", "in", "to", "of", "for", "on", "with"
    ])
    return [t for t in tokens if t.lower() not in stop_words]


def lemmatize_text(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


def processing_text(text):
    text = expand_contractions(text)
    text = clean_text(text)
    tokens = custom_tokenizer(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_text(tokens)


# =========================
# STEM vs LEMMA
# =========================
def compare_stem_vs_lemma(text):

    lemma = processing_text(text)

    text_clean = expand_contractions(text)
    text_clean = clean_text(text_clean)
    tokens = custom_tokenizer(text_clean)
    tokens = remove_stopwords(tokens)

    stemmed = [stemmer.stem(t) for t in tokens]

    return stemmed, lemma


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="NLP Pipeline", layout="wide")

st.title("🧠 NLP Text Preprocessing Pipeline")
st.write("Clean, tokenize, remove stopwords, lemmatize & compare NLP methods.")

menu = st.sidebar.selectbox(
    "Choose Option",
    ["Single Text Processing", "Stem vs Lemma", "Batch CSV Processing"]
)

# =========================
# 1. SINGLE TEXT
# =========================
if menu == "Single Text Processing":

    st.subheader("🔹 Process Single Text")

    text = st.text_area("Enter text")

    if st.button("Process"):

        if text.strip():

            result = processing_text(text)

            st.success("Processed Output:")
            st.write(" ".join(result))

        else:
            st.warning("Enter text first")


# =========================
# 2. STEM VS LEMMA
# =========================
elif menu == "Stem vs Lemma":

    st.subheader("🔹 Compare Stemming vs Lemmatization")

    text = st.text_area("Enter text")

    if st.button("Compare"):

        if text.strip():

            stemmed, lemma = compare_stem_vs_lemma(text)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🌱 Stemmed Output")
                st.write(" ".join(stemmed))

            with col2:
                st.subheader("🌿 Lemmatized Output")
                st.write(lemma)

        else:
            st.warning("Enter text first")


# =========================
# 3. BATCH CSV PROCESSING
# =========================
elif menu == "Batch CSV Processing":

    st.subheader("🔹 Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])
    column = st.text_input("Enter text column name")

    if st.button("Process Dataset"):

        if file and column:

            df = pd.read_csv(file)

            processed = []

            for text in df[column].fillna(""):
                processed.append(" ".join(processing_text(text)))

            df["processed_text"] = processed

            st.success("Processing Completed")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                csv,
                "processed_output.csv",
                "text/csv"
            )

        else:
            st.warning("Upload file and column name")