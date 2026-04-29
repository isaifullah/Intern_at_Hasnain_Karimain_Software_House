import re
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

#--------------------------
# Contractions Dictionary
#--------------------------
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

# ------------------------
# core functions
# ------------------------

def validate_input(text):
    return isinstance(text, str) and text.strip() != ""


def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, contractions_dict.keys())) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def custom_tokenizer(text):
    return re.findall(r'\b[a-zA-Z]+\b', text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]


def lemmatize_text(tokens):
    doc = nlp(" ".join(tokens), disable=["parser", "ner"])
    return [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]


def text_preprocessing_pipeline(text):
    if not validate_input(text):
        return []

    text = expand_contractions(text)
    text = clean_text(text)

    tokens = custom_tokenizer(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)

    return tokens


def preprocess_batch(data, text_column=None, method="lemma"):

    import pandas as pd
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()

    # -----------------------
    # Load / extract texts
    # -----------------------
    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if text_column is None:
            raise ValueError("text_column must be provided for DataFrame input")
        texts = data[text_column].fillna("").tolist()

    elif isinstance(data, list):
        texts = data

    else:
        raise ValueError("Unsupported input type")

    # -----------------------
    # Processing
    # -----------------------
    results = []

    for text in texts:

        if not validate_input(text):
            results.append("")
            continue

        # 🔥 STEM MODE
        if method == "stem":
            text_clean = clean_text(expand_contractions(text))
            tokens = custom_tokenizer(text_clean)
            tokens = remove_stopwords(tokens)
            processed = [stemmer.stem(t) for t in tokens]

        # 🔥 LEMMA MODE (DEFAULT)
        else:
            processed = text_preprocessing_pipeline(text)

        results.append(" ".join(processed))

    return results

def compare_stem_vs_lemma(text):
    stemmer = PorterStemmer()

    text = clean_text(expand_contractions(text))
    tokens = custom_tokenizer(text)
    tokens = remove_stopwords(tokens)

    stemmed = [stemmer.stem(t) for t in tokens]
    lemma = text_preprocessing_pipeline(text)

    return {"stemmed": stemmed, "lemmatized": lemma}