import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-alphabetic
    return text

def get_vectorizer(max_features=5000, ngram_range=(1,2)):
    """Create and return a TF-IDF vectorizer"""
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

def save_vectorizer(vectorizer, path="models/vectorizer.pkl"):
    joblib.dump(vectorizer, path)

def load_vectorizer(path="models/vectorizer.pkl"):
    return joblib.load(path)
