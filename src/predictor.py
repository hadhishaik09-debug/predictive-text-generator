import json
import os

USER_WORDS_FILE = "data/user_words.json"

def load_user_words():
    if not os.path.exists(USER_WORDS_FILE) or os.path.getsize(USER_WORDS_FILE) == 0:
        return []
    try:
        with open(USER_WORDS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def add_user_word(word):
    words = load_user_words()
    if word not in words:
        words.append(word)
    with open(USER_WORDS_FILE, 'w') as f:
        json.dump(words, f)