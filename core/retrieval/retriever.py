from .base import BaseRetriever
from core.indexing.faiss_topic_indexing import FaissTopicIndexer
from config import path

# Setting
BM25_PATH = path.PROJECT_ROOT / "save/dinh-doc-lap-knowledge-indexes-bm25.pkl"
SEMANTIC_PATH = path.PROJECT_ROOT / "save/dinh-doc-lap-intents-indexes-bge-m3.pkl"
FAISS_INDEX_DIR = path.PROJECT_ROOT / "save/faiss_topic_index"

class Retriever(BaseRetriever):
    def __init__(self):
        self.model = None
        self._faiss_mode = False
        self._faiss_result_map = {}
        if not FAISS_INDEX_DIR.exists():
            raise ValueError(
                f"FAISS index directory not found: {FAISS_INDEX_DIR}. "
                f"Please run core/indexing/faiss_topic_indexing.py first."
            )
        self._faiss_mode = True
    
    def search(self, query, top_k = 5):
        if self._faiss_mode:
            raw_results = FaissTopicIndexer.search(FAISS_INDEX_DIR, query, top_k=top_k)
            self._faiss_result_map = {item["chunk_id"]: item for item in raw_results}
            return [(item["chunk_id"], item["score"]) for item in raw_results]

        return self.model.search(query, top_k)

    def search_threshold(self, query: str, threshold = 0, top_k = 20):
        if self._faiss_mode:
            raw_results = FaissTopicIndexer.search(FAISS_INDEX_DIR, query, top_k=top_k)
            self._faiss_result_map = {item["chunk_id"]: item for item in raw_results}
            results = [(item["chunk_id"], item["score"]) for item in raw_results]

            if not results:
                return []

            # FAISS cosine/IP scores are often < 0.8, so strict filtering can empty results.
            if threshold is None:
                return results

            filtered = [(doc_id, score) for doc_id, score in results if score >= threshold]
            if filtered:
                return filtered

            # Adaptive fallback: keep top positive-score chunks if threshold is too strict.
            positive = [(doc_id, score) for doc_id, score in results if score > 0]
            if positive:
                return positive[: min(5, len(positive))]

            return results[: min(3, len(results))]
        else:
            results = self.model.search(query, top_k=None)

        filtered = [(doc_id, score) for doc_id, score in results if score >= threshold]
        return filtered

    def build_context(self, retrieved_docs):
        if not self._faiss_mode:
            return ""

        blocks = []
        for chunk_id, score in retrieved_docs:
            item = self._faiss_result_map.get(chunk_id)
            if not item:
                continue

            md = item.get("metadata", {})
            source = md.get("source_rel_path") or md.get("source_file", "unknown")
            topic = md.get("topic_key", "unknown")
            text = item.get("text", "")

            blocks.append(
                f"[source={source} | topic={topic} | score={score:.4f}]\n{text}"
            )

        return "\n\n".join(blocks)