import pickle
from core.retrieval.bm25 import BM25Retriever

class BM25Loader:

    @staticmethod
    def save(bm25_retriever, path):
        data = {
            "bm25": bm25_retriever.bm25,
            "doc_names": bm25_retriever.doc_names
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path, preprocess_func):
        with open(path, "rb") as f:
            data = pickle.load(f)

        retriever = object.__new__(BM25Retriever)
        retriever.bm25 = data["bm25"]
        retriever.doc_names = data["doc_names"]
        retriever.preprocess = preprocess_func

        return retriever