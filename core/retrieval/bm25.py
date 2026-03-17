from rank_bm25 import BM25Okapi
from .base import BaseRetriever

class BM25Retriever(BaseRetriever):

    def __init__(self, documents, preprocess_func):
        self.doc_names = list(documents.keys())
        self.preprocess = preprocess_func

        tokenized_docs = [
            self.preprocess(documents[d])
            for d in self.doc_names
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, top_k=5):
        tokens = self.preprocess(query)

        scores = self.bm25.get_scores(tokens)

        results = sorted(
            [(self.doc_names[i], scores[i]) for i in range(len(scores))],
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]