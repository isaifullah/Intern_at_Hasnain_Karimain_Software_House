# utils.py

# import necessary libraries
import re                         # For text cleaning using regex
import nltk                       # NLP toolkit

# Import NLP tools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# download necessary NLTK resources
# These downloads are required for tokenization, stopwords, POS tagging, etc.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize NLP tools such as stopwords and lemmatizer
stop_words = set(stopwords.words('english'))   # Load stopwords
lemmatizer = WordNetLemmatizer()               # Initialize lemmatizer

# create a function to preprocess text data
def preprocess_text(text):
    """
    This function cleans and preprocesses text data
    Steps:
    - Lowercasing
    - Remove URLs, HTML, numbers
    - Remove punctuation
    - Tokenization
    - POS tagging
    - Lemmatization
    - Stopword removal
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Replace currency values with 'money'
    text = re.sub(r'\$\s?\d+(\.\d+)?', ' money ', text)

    # Replace numbers with 'number'
    text = re.sub(r'\d+(\.\d+)?', ' number ', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Apply Part-of-Speech tagging
    pos_tags = pos_tag(tokens)

    # Helper function to convert POS tags for WordNet
    def get_wordnet_pos(tag):
        if tag.startswith('J'): return 'a'   # Adjective
        elif tag.startswith('V'): return 'v' # Verb
        elif tag.startswith('N'): return 'n' # Noun
        elif tag.startswith('R'): return 'r' # Adverb
        else: return 'n'

    cleaned = []

    # Lemmatization + stopword removal + short word filtering
    for word, tag in pos_tags:
        if word not in stop_words and len(word) > 2:
            cleaned.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    # Join cleaned words into final string
    return " ".join(cleaned)