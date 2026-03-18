def load_stopwords(stopwords_path):
    with open(stopwords_path, encoding="utf-8") as f:
        stopwords = {w.strip() for w in f}

    return stopwords