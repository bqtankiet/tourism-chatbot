from pathlib import Path

def load_stopwords():
    stopwords_path ="stopwords/vietnamese-stopwords.txt"
    stopwords_dash_path ="stopwords/vietnamese-stopwords-dash.txt"

    with open(stopwords_path, encoding="utf-8") as f:
        stopwords = {w.strip() for w in f}

    with open(stopwords_dash_path, encoding="utf-8") as f:
        stopwords_dash = {w.strip() for w in f}

    return stopwords, stopwords_dash