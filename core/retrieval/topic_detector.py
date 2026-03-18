from sentence_transformers import SentenceTransformer
import numpy as np
from core.loader.model_cache import get_or_create_model

class TopicDetector:
    def __init__(self, topic_map, model = "BAAI/bge-m3"):
        self.model = get_or_create_model(
            cache_key=("SentenceTransformer", model),
            factory=lambda: SentenceTransformer(model),
        )

        self.topic_map = topic_map

        # encode sẵn
        self.topic_names = list(self.topic_map.keys())
        topic_texts = list(self.topic_map.values())

        self.topic_embeds = self.model.encode(
            topic_texts,
            normalize_embeddings=True
        )
    
    def detect(self, query, top_k=3, threshold=0.3):
        q_embed = self.model.encode(
            query,
            normalize_embeddings=True
        )

        scores = np.dot(self.topic_embeds, q_embed)

        # lấy index top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append({
                    "topic": self.topic_names[idx],
                    "score": score
                })

        if not results:
            return [{"topic": "unknown", "score": float(np.max(scores))}]

        return results
    
    def save(self, save_dir):
        import os, json, numpy as np

        os.makedirs(save_dir, exist_ok=True)

        # chỉ save embeddings + metadata
        np.save(os.path.join(save_dir, "topic_embeds.npy"), self.topic_embeds)

        meta = {
            "topic_names": self.topic_names,
            "topic_map": self.topic_map,
            "model_name": self.model._model_card_vars.get("model_name", None)  # optional
        }

        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, save_dir, model="BAAI/bge-m3"):
        import os, json, numpy as np

        # load metadata
        with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        topic_map = meta["topic_map"]
        topic_names = meta["topic_names"]

        # model trong meta nếu có
        model_name = meta.get("model_name") or model
        model = get_or_create_model(
            cache_key=("SentenceTransformer", model_name),
            factory=lambda: SentenceTransformer(model_name),
        )

        # tạo object
        obj = cls.__new__(cls)
        obj.model = model
        obj.topic_map = topic_map
        obj.topic_names = topic_names

        # load embeddings
        obj.topic_embeds = np.load(os.path.join(save_dir, "topic_embeds.npy"))

        return obj