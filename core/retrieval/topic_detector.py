import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core.loader.model_cache import get_or_create_model


DEFAULT_TOPIC_MODEL = "BAAI/bge-m3"
DEFAULT_TOPIC_MAP_PATH = Path("knowledge/topic_maps.json")
DEFAULT_TOPIC_DETECTOR_DIR = Path("save/topic_detector")


def _load_topic_map(topic_map_path: Path) -> dict:
    if not topic_map_path.exists():
        raise ValueError(f"Topic map not found: {topic_map_path}")

    with open(topic_map_path, "r", encoding="utf-8") as f:
        topic_map = json.load(f)

    if not isinstance(topic_map, dict) or not topic_map:
        raise ValueError("topic_maps.json is empty or invalid")

    return topic_map


def _topic_map_to_texts(topic_map: dict) -> tuple[list[str], list[str]]:
    topic_names = list(topic_map.keys())
    topic_texts: list[str] = []

    for topic_name in topic_names:
        payload = topic_map.get(topic_name, {})
        if isinstance(payload, dict):
            queries = payload.get("queries", [])
            if isinstance(queries, list):
                queries = [str(q).strip() for q in queries if str(q).strip()]
                merged = "\n".join(queries)
                topic_texts.append(f"{topic_name}\n{merged}" if merged else topic_name)
                continue

        topic_texts.append(f"{topic_name}\n{str(payload)}")

    return topic_names, topic_texts

class TopicDetector:
    def __init__(self, topic_map, model=DEFAULT_TOPIC_MODEL):
        self.model = get_or_create_model(
            cache_key=("SentenceTransformer", model),
            factory=lambda: SentenceTransformer(model),
        )
        self.model_name = model

        self.topic_map = topic_map

        self.topic_names, topic_texts = _topic_map_to_texts(self.topic_map)

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
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "topic_names": self.topic_names,
            "topic_map": self.topic_map,
            "topic_embeds": self.topic_embeds,
            "model_name": self.model_name,
        }

        with open(save_path / "detector.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, save_dir, model=DEFAULT_TOPIC_MODEL):
        save_path = Path(save_dir)
        detector_pkl = save_path / "detector.pkl"

        if detector_pkl.exists():
            with open(detector_pkl, "rb") as f:
                payload = pickle.load(f)

            topic_map = payload["topic_map"]
            topic_names = payload["topic_names"]
            topic_embeds = payload["topic_embeds"]
            model_name = payload.get("model_name") or model
        else:
            # Backward compatibility for old files: meta.json + topic_embeds.npy
            meta_path = save_path / "meta.json"
            embeds_path = save_path / "topic_embeds.npy"

            if not meta_path.exists() or not embeds_path.exists():
                raise ValueError(f"Topic detector files not found in: {save_path}")

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            topic_map = meta["topic_map"]
            topic_names = meta["topic_names"]
            topic_embeds = np.load(embeds_path)
            model_name = meta.get("model_name") or model

        loaded_model = get_or_create_model(
            cache_key=("SentenceTransformer", model_name),
            factory=lambda: SentenceTransformer(model_name),
        )

        obj = cls.__new__(cls)
        obj.model = loaded_model
        obj.model_name = model_name
        obj.topic_map = topic_map
        obj.topic_names = topic_names
        obj.topic_embeds = topic_embeds
        return obj


def ensure_topic_detector(
    project_root: str | Path,
    topic_map_path: str | Path = DEFAULT_TOPIC_MAP_PATH,
    save_dir: str | Path = DEFAULT_TOPIC_DETECTOR_DIR,
    model: str = DEFAULT_TOPIC_MODEL,
    force_rebuild: bool = False,
) -> dict | None:
    project_root = Path(project_root)

    resolved_topic_map = Path(topic_map_path)
    if not resolved_topic_map.is_absolute():
        resolved_topic_map = project_root / resolved_topic_map

    resolved_save_dir = Path(save_dir)
    if not resolved_save_dir.is_absolute():
        resolved_save_dir = project_root / resolved_save_dir

    detector_pkl = resolved_save_dir / "detector.pkl"
    legacy_meta = resolved_save_dir / "meta.json"
    legacy_embeds = resolved_save_dir / "topic_embeds.npy"

    if not force_rebuild and (detector_pkl.exists() or (legacy_meta.exists() and legacy_embeds.exists())):
        return None

    topic_map = _load_topic_map(resolved_topic_map)
    detector = TopicDetector(topic_map=topic_map, model=model)
    detector.save(resolved_save_dir)

    return {
        "model_name": model,
        "topic_map_path": str(resolved_topic_map),
        "save_dir": str(resolved_save_dir),
        "detector_path": str(detector_pkl),
    }


if __name__ == "__main__":
    from config import path

    out = ensure_topic_detector(project_root=path.PROJECT_ROOT)
    print("Topic detector ready")
    print(out if out is not None else "Skip build (already exists)")