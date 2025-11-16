import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    tokens = word_tokenize(text)
    return tokens
