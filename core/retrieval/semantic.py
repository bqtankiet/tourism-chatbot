import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseRetriever
from core.loader.model_cache import get_or_create_model

class SemanticRetriever(BaseRetriever):

    def __init__(self, resources, model_name="BAAI/bge-m3"):
        self.model_name = model_name
        self.model = get_or_create_model(
            cache_key=("SentenceTransformer", model_name),
            factory=lambda: SentenceTransformer(model_name),
        )

        self.resource_embeddings = {
            resource: np.array([
                self.model.encode(intent)
                for intent in intents
            ])
            for resource, intents in resources.items()
        }

    def search(self, query, top_k=5):
        q_emb = self.model.encode(query)

        scores = {}

        for resource, embeddings in self.resource_embeddings.items():
            sim = max(np.dot(q_emb, emb) for emb in embeddings)
            scores[resource] = sim

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]