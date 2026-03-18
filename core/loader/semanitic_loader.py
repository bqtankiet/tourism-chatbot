import pickle
from core.retrieval.semantic import SemanticRetriever
from sentence_transformers import SentenceTransformer
from core.loader.model_cache import get_or_create_model

class SemanticLoader:

    @staticmethod
    def save(semantic_retriever, path):
        data = {
            "resource_embeddings": semantic_retriever.resource_embeddings,
            "model_name": semantic_retriever.model_name
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        retriever = object.__new__(SemanticRetriever)

        retriever.resource_embeddings = data["resource_embeddings"]

        # load lại model
        retriever.model_name = data["model_name"]
        retriever.model = get_or_create_model(
            cache_key=("SentenceTransformer", retriever.model_name),
            factory=lambda: SentenceTransformer(retriever.model_name),
        )

        return retriever