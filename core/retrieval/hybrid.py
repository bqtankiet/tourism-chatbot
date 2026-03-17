from .base import BaseRetriever


class HybridRetriever(BaseRetriever):

    def __init__(self, semantic_retriever, bm25_retriever, alpha=0.7):
        """
        alpha: weight cho semantic (embedding)
        bm25 weight = 1 - alpha
        """
        self.semantic = semantic_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha

    def normalize(self, scores: dict):
        if not scores:
            return {}

        max_v = max(scores.values())
        min_v = min(scores.values())

        if max_v == min_v:
            return {k: 0 for k in scores}

        return {
            k: (v - min_v) / (max_v - min_v)
            for k, v in scores.items()
        }

    def search(self, query, top_k=5):
        emb_results = self.semantic.search(query, top_k=None)
        bm25_results = self.bm25.search(query, top_k=None)

        # convert list -> dict
        emb_scores = dict(emb_results)
        bm25_scores = dict(bm25_results)

        # normalize
        emb_scores = self.normalize(emb_scores)
        bm25_scores = self.normalize(bm25_scores)

        # combine
        final_scores = {}
        all_docs = set(emb_scores) | set(bm25_scores)

        for doc in all_docs:
            emb = emb_scores.get(doc, 0)
            bm = bm25_scores.get(doc, 0)

            final_scores[doc] = self.alpha * emb + (1 - self.alpha) * bm

        ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]