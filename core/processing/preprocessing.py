import re
import string
from utils.load_stopwords import load_stopwords
from underthesea import word_tokenize

stopwords, stopwords_dash = load_stopwords()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text) # url
    text = re.sub(r'\b\d{9,11}\b', ' ', text) # phone number
    text = text.translate(str.maketrans('', '', string.punctuation)) # punctuation
    text = re.sub(r'\s+', ' ', text).strip() # extra space
    return text


def preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if t not in stopwords_dash]

    return tokens