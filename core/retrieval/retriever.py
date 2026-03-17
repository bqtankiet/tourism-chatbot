from .base import BaseRetriever
from core.loader.bm25_loader import BM25Loader
from core.processing.preprocessing import preprocess
from core.loader.semanitic_loader import SemanticLoader
from core.retrieval.hybrid import HybridRetriever
from config import path

# Strategies
BM25 = "bm25"
SEMANTIC = "semantic"
HYBRID = "hybrid"

# Setting
BM25_PATH = path.PROJECT_ROOT / "save/dinh-doc-lap-knowledge-indexes-bm25.pkl"
SEMANTIC_PATH = path.PROJECT_ROOT / "save/dinh-doc-lap-intents-indexes-bge-m3.pkl"

class Retriever(BaseRetriever):
    def __init__(self, strategy):
        self.strategy = strategy
        self.model = None
        if strategy == BM25:
            self.model = BM25Loader.load(BM25_PATH, preprocess)
        elif strategy == SEMANTIC:
            self.model = SemanticLoader.load(SEMANTIC_PATH)
        elif strategy == HYBRID:
            bm25 = BM25Loader.load(BM25_PATH, preprocess)
            semantic = SemanticLoader.load(SEMANTIC_PATH)
            self.model = HybridRetriever(semantic, bm25)
        else:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Supported strategies are: {BM25}, {SEMANTIC}, {HYBRID}"
            )
    
    def search(self, query, top_k = 5):
        return self.model.search(query, top_k)

    def search_threshold(self, query: str, threshold = 0.8):
        results = self.model.search(query, top_k=None)
        filtered = [(doc_id, score) for doc_id, score in results if score >= threshold]
        return filtered