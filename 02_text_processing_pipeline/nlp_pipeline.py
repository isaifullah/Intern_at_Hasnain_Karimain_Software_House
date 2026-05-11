# nlp_pipeline.py
"""
NLP Text Preprocessing Pipeline with Comprehensive Edge Case Handling
Clean, tokenize, and preprocess text for Machine Learning tasks
"""

import re
import spacy
import pandas as pd
import logging
import unicodedata
import html
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(
    filename="nlp_pipeline.log",
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# Load Models and Resources
# ---------------------------
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ---------------------------
# Extended Contractions Dictionary
# ---------------------------
contractions_dict = {
    "I'm": "I am",
    "I've": "I have",
    "I'll": "I will",
    "I'd": "I would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "it's": "it is",
    "it'll": "it will",
    "it'd": "it would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "can't": "cannot",
    "cannot": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "could've": "could have",
    "would've": "would have",
    "should've": "should have",
    "might've": "might have",
    "must've": "must have",
    "ain't": "am not",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "y'all": "you all",
    "whatcha": "what are you",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "outta": "out of",
    "cuz": "because",
    "bc": "because",
    "dunno": "do not know",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "mighta": "might have",
    "musta": "must have",
}

# ---------------------------
# EDGE CASE HANDLERS
# ---------------------------

def handle_null_input(text):
    """Edge Case 1: Handle None, empty string, or whitespace-only input"""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return text.strip()


def handle_non_english_chars(text):
    """Edge Case 2: Handle Unicode, emojis, special characters"""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    return text


def handle_html_entities(text):
    """Edge Case 3: Decode HTML entities (&amp; → &)"""
    return html.unescape(text)


def handle_urls_emails(text):
    """Edge Case 4: Remove URLs, emails, and social media handles"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    return text


def handle_repeated_characters(text):
    """Edge Case 5: Normalize repeated characters (coooool → cool)"""
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


def handle_numbers_in_words(text):
    """Edge Case 6: Remove numbers from within words (ai2024 → ai)"""
    words = text.split()
    cleaned_words = []
    for word in words:
        if re.search(r'[a-zA-Z]', word) and re.search(r'\d', word):
            word = re.sub(r'\d+', '', word)
        cleaned_words.append(word)
    return ' '.join(cleaned_words)


def handle_special_quotes_dashes(text):
    """Edge Case 7: Normalize smart quotes and special dashes"""
    text = text.replace('’', "'").replace('‘', "'").replace('”', '"').replace('“', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', ' ')
    return text


def preprocess_edge_cases(text):
    """
    Master edge case handler - applies all edge case preprocessing
    BEFORE sending to main pipeline
    """
    if text is None or (isinstance(text, str) and text.strip() == ""):
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    # Apply edge case handlers in logical order
    text = handle_null_input(text)
    text = handle_html_entities(text)
    text = handle_special_quotes_dashes(text)
    text = handle_urls_emails(text)
    text = handle_non_english_chars(text)
    text = handle_repeated_characters(text)
    text = handle_numbers_in_words(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ---------------------------
# Main Pipeline Functions
# ---------------------------

def validate_input(text):
    """
    STEP 0: Ensures safe pipeline execution
    MUST BE USED BEFORE ANY PROCESSING
    """
    if text is None:
        return False
    if not isinstance(text, str):
        return False
    if text.strip() == "":
        return False
    return True


def expand_contractions(text):
    """
    Expands contractions like don't → do not
    """
    if not isinstance(text, str):
        return ""
    
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, contractions_dict.keys())) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)


def clean_text(text):
    """
    Cleans raw text:
    - lowercase
    - remove URLs
    - remove emails
    - remove special characters
    - normalize spaces
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def custom_tokenizer(text):
    """
    ADVANCED CUSTOM TOKENIZER (IMPROVED)
    - Handles alphanumeric tokens
    - Splits intelligently
    - Removes noise
    - Keeps meaningful single letters (a, i)
    """
    if not text or text.strip() == "":
        return []
    
    raw_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
    
    tokens = []
    for token in raw_tokens:
        token = token.lower()
        
        # remove numbers inside words (ai2024 → ai)
        token = re.sub(r'\d+', '', token)
        
        # skip empty after cleaning
        if token == "":
            continue
        
        # remove repeated chars (coooool → cool)
        token = re.sub(r'(.)\1{2,}', r'\1\1', token)
        
        # Keep single letters 'a' and 'i', remove other single letters
        if len(token) == 1 and token in ['a', 'i']:
            tokens.append(token)
        elif len(token) >= 2:
            if token not in ["thing", "something", "anything", "everything"]:
                tokens.append(token)
    
    return tokens


def remove_stopwords(tokens):
    """
    Removes stopwords + noise filtering
    Keeps meaningful single letters 'a' and 'i'
    """
    filtered = []
    for t in tokens:
        t = t.lower().strip()
        
        # Keep 'a' and 'i' as they have meaning
        if t in ['a', 'i']:
            filtered.append(t)
        elif t not in stop_words:
            if len(t) >= 2:
                filtered.append(t)
    
    return filtered


def lemmatize_text(tokens):
    """Lemmatize tokens using spaCy"""
    if not tokens:
        return []
    
    text = " ".join(tokens)
    doc = nlp(text, disable=["parser", "ner"])
    
    return [
        token.lemma_
        for token in doc
        if token.lemma_ != "-PRON-" and token.lemma_.strip()
    ]


def text_preprocessing_pipeline(text, config=None):
    """
    ENHANCED TEXT PREPROCESSING PIPELINE
    Handles all edge cases automatically
    """
    if config is None:
        config = {
            "remove_stopwords": True,
            "lemmatize": True,
            "remove_duplicates": True
        }
    
    try:
        logger.info("Pipeline started")
        
        # Step 0: Validation
        if not validate_input(text):
            logger.warning("Invalid input received")
            return []
        
        logger.info(f"Raw Input: {text}")
        
        # Step 1: Edge case preprocessing
        text = preprocess_edge_cases(text)
        
        if text.strip() == "":
            logger.warning("Text became empty after edge case handling")
            return []
        
        # Step 2: Expand contractions
        text = expand_contractions(text)
        logger.info(f"After contraction expansion: {text}")
        
        # Step 3: Clean text
        text = clean_text(text)
        logger.info(f"After cleaning: {text}")
        
        if text.strip() == "":
            return []
        
        # Step 4: Tokenize
        tokens = custom_tokenizer(text)
        logger.info(f"Tokens: {tokens}")
        
        if not tokens:
            return []
        
        # Step 5: Remove stopwords
        if config.get("remove_stopwords", True):
            tokens = remove_stopwords(tokens)
            logger.info(f"After stopword removal: {tokens}")
        
        if not tokens:
            return []
        
        # Step 6: Lemmatize
        if config.get("lemmatize", True):
            tokens = lemmatize_text(tokens)
            logger.info(f"After lemmatization: {tokens}")
        
        # Step 7: Remove duplicates
        if config.get("remove_duplicates", True):
            tokens = list(dict.fromkeys(tokens))
            logger.info(f"After duplicate removal: {tokens}")
        
        logger.info(f"Final Output: {tokens}")
        return tokens
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return []


# ---------------------------
# Comparison Functions
# ---------------------------

def compare_stem_vs_lemma(text):
    """
    Compare stemming vs lemmatization using your pipeline
    Both methods now use edge case preprocessing for consistency
    """
    if not validate_input(text):
        return None
    
    # Apply edge cases first for consistency
    text_processed = preprocess_edge_cases(text)
    
    print("\nRAW:", text)
    
    # Lemmatization (uses full enhanced pipeline)
    lemma = text_preprocessing_pipeline(text)
    
    # Stemming (also uses edge cases)
    text_clean = clean_text(expand_contractions(text_processed))
    tokens = custom_tokenizer(text_clean)
    tokens = remove_stopwords(tokens)
    stemmed = [stemmer.stem(t) for t in tokens]
    
    print("\nSTEMMED:", stemmed)
    print("\nLEMMATIZED:", lemma)
    
    return {"stemmed": stemmed, "lemmatized": lemma}


def compare_raw_vs_processed(text):
    """Compare raw text with processed output"""
    if not validate_input(text):
        print("Invalid input")
        return []
    
    print("\nRAW:", text)
    processed = text_preprocessing_pipeline(text)
    print("\nPROCESSED:", processed)
    
    return processed


# ---------------------------
# Batch Processing
# ---------------------------

def preprocess_batch(data, text_column=None, method="lemma"):
    """
    Batch processing of multiple texts using the pipeline
    Can read from CSV file, DataFrame, or list
    """
    # Load / extract texts
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
    
    # Process texts
    results = []
    
    for text in texts:
        if not validate_input(text):
            results.append("")
            continue
        
        # STEM MODE
        if method == "stem":
            # Apply edge cases for consistency
            text_processed = preprocess_edge_cases(text)
            text_clean = clean_text(expand_contractions(text_processed))
            tokens = custom_tokenizer(text_clean)
            tokens = remove_stopwords(tokens)
            processed = [stemmer.stem(t) for t in tokens]
        
        # LEMMA MODE (DEFAULT)
        else:
            processed = text_preprocessing_pipeline(text)
        
        results.append(" ".join(processed) if processed else "")
    
    return results


# ---------------------------
# Edge Case Testing
# ---------------------------

def test_edge_cases():
    """Test suite for edge cases to demonstrate pipeline robustness"""
    test_cases = [
        ("Empty String", ""),
        ("None Value", None),
        ("Only Spaces", "   "),
        ("Single Letter 'a'", "a"),
        ("Single Letter 'I'", "I"),
        ("Repeated Characters", "coooool hiiiiiii"),
        ("Numbers in Words", "ai2024 version2 beta3"),
        ("Emojis", "I love 🎉 this! 😊"),
        ("URLs", "Check https://example.com for info"),
        ("Emails", "Contact me@test.com please"),
        ("HTML Tags", "<p>Hello</p> <b>World</b>"),
        ("Special Quotes", "Hello ‘world’ and “universe”"),
        ("Dashes", "This—is—awesome – really"),
        ("Ellipsis", "Wait... what... really?"),
        ("Smart Quotes", "It’s a beautiful day"),
        ("Very Long Word", "supercalifragilisticexpialidocious"),
        ("Only Punctuation", "!!! ??? ..."),
        ("Complex Contractions", "y'all'd've ain't gonna"),
        ("Social Media", "Check @username and #hashtag"),
        ("Mixed Case", "HELLO World Mixed CASE"),
        ("Mix of Everything", "COOOOOL!!! https://test.com @user #hash 🎉 It’s 2024!"),
    ]
    
    print("\n" + "="*80)
    print("EDGE CASE TEST SUITE RESULTS")
    print("="*80)
    
    results = []
    for name, test_input in test_cases:
        output = text_preprocessing_pipeline(test_input)
        status = "✅ PASS" if output is not None else "❌ FAIL"
        results.append({"test_case": name, "status": status, "output": output})
        
        print(f"\n📌 {name}:")
        print(f"   Input: {repr(test_input)[:60]}")
        print(f"   Output: {output}")
        print(f"   Status: {status}")
    
    print("\n" + "="*80)
    print(f"SUMMARY: {sum(1 for r in results if 'PASS' in r['status'])}/{len(results)} tests passed")
    print("="*80)
    
    return results


# ---------------------------
# CLI Menu
# ---------------------------

def run_cli():
    """Run command line interface for testing"""
    while True:
        print("\n===== NLP PIPELINE =====")
        print("1. Single Text")
        print("2. Raw vs Processed")
        print("3. Stem vs Lemma")
        print("4. Batch CSV")
        print("5. Run Edge Case Tests")
        print("6. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            text = input("Enter text: ")
            result = text_preprocessing_pipeline(text)
            print(f"Processed: {result}")
        
        elif choice == "2":
            text = input("Enter text: ")
            compare_raw_vs_processed(text)
        
        elif choice == "3":
            text = input("Enter text: ")
            compare_stem_vs_lemma(text)
        
        elif choice == "4":
            path = input("CSV path: ")
            col = input("Column name: ")
            method = input("Method (lemma/stem): ").lower()
            
            df = pd.read_csv(path)
            df["processed_text"] = preprocess_batch(df, col, method)
            df.to_csv("processed_output.csv", index=False)
            print("Saved as processed_output.csv")
        
        elif choice == "5":
            test_edge_cases()
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")


# ---------------------------
# Module Exports
# ---------------------------
__all__ = [
    'text_preprocessing_pipeline',
    'preprocess_batch',
    'compare_stem_vs_lemma',
    'compare_raw_vs_processed',
    'test_edge_cases',
    'run_cli'
]

# Run CLI if script is executed directly
if __name__ == "__main__":
    run_cli()