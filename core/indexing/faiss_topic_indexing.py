import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from core.indexing.knowledge_chunking import KnowledgeTopicChunker
from core.loader.model_cache import get_or_create_model
from core.retrieval.topic_detector import ensure_topic_detector
import faiss


DEFAULT_FAISS_MODEL_NAME = "BAAI/bge-m3"

class FaissTopicIndexer:
    def __init__(self, project_root: str | Path, model_name: str):
        self.project_root = Path(project_root)
        self.model_name = model_name

        self.chunker = KnowledgeTopicChunker(project_root=self.project_root)
        self.model = get_or_create_model(
            cache_key=("SentenceTransformer", model_name),
            factory=lambda: SentenceTransformer(model_name),
        )

    def _build_chunk_text(self, chunk: dict) -> str:
        topic_key = chunk.get("metadata", {}).get("topic_key", "unknown")
        text = chunk.get("text", "")
        return f"Topic: {topic_key}\n\n{text}"

    def build(
        self,
        knowledge_dir: str | Path = "knowledge",
        output_dir: str | Path = "save/faiss_topic_index",
    ) -> dict:
        knowledge_dir = Path(knowledge_dir)
        if not knowledge_dir.is_absolute():
            knowledge_dir = self.project_root / knowledge_dir

        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = self.project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = self.chunker.chunk_knowledge(knowledge_dir)
        if not chunks:
            raise ValueError("No chunks generated from knowledge directory")

        chunk_texts = [self._build_chunk_text(c) for c in chunks]
        embeddings = self.model.encode(chunk_texts, normalize_embeddings=True)
        vectors = np.asarray(embeddings, dtype="float32")

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        index_path = output_dir / "topic_chunks.index"
        vectors_path = output_dir / "topic_vectors.pkl"
        metadata_path = output_dir / "topic_metadata.pkl"
        manifest_path = output_dir / "manifest.pkl"

        faiss.write_index(index, str(index_path))
        with open(vectors_path, "wb") as f:
            pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append(
                {
                    "row_id": i,
                    "chunk_id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                }
            )

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        manifest = {
            "model_name": self.model_name,
            "dimension": int(dim),
            "num_vectors": int(vectors.shape[0]),
            "knowledge_dir": str(knowledge_dir),
            "index_path": str(index_path),
            "vectors_path": str(vectors_path),
            "metadata_path": str(metadata_path),
        }

        with open(manifest_path, "wb") as f:
            pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

        return manifest

    @staticmethod
    def search(index_dir: str | Path, query: str, top_k: int = 5) -> list[dict]:
        index_dir = Path(index_dir)

        manifest_path = index_dir / "manifest.pkl"
        legacy_manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "rb") as f:
                manifest = pickle.load(f)
        elif legacy_manifest_path.exists():
            with open(legacy_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            raise ValueError(f"manifest.pkl not found in: {index_dir}")

        model_name = manifest["model_name"]
        model = get_or_create_model(
            cache_key=("SentenceTransformer", model_name),
            factory=lambda: SentenceTransformer(model_name),
        )
        index = faiss.read_index(str(index_dir / "topic_chunks.index"))

        metadata_path = index_dir / "topic_metadata.pkl"
        legacy_metadata_path = index_dir / "topic_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        elif legacy_metadata_path.exists():
            with open(legacy_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            raise ValueError(f"topic_metadata.pkl not found in: {index_dir}")

        q_vec = model.encode([query], normalize_embeddings=True)
        q_vec = np.asarray(q_vec, dtype="float32")

        scores, indices = index.search(q_vec, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(metadata):
                continue

            item = metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "row_id": item["row_id"],
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                }
            )

        return results


def ensure_faiss_topic_index(
    project_root: str | Path,
    model_name: str = DEFAULT_FAISS_MODEL_NAME,
    knowledge_dir: str | Path = "knowledge",
    output_dir: str | Path = "save/faiss_topic_index",
    force_rebuild: bool = False,
) -> dict | None:
    project_root = Path(project_root)
    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = project_root / resolved_output_dir

    manifest_pkl = resolved_output_dir / "manifest.pkl"
    manifest_json = resolved_output_dir / "manifest.json"

    if not force_rebuild and (manifest_pkl.exists() or manifest_json.exists()):
        return None

    ensure_topic_detector(project_root=project_root)

    indexer = FaissTopicIndexer(project_root=project_root, model_name=model_name)
    return indexer.build(knowledge_dir=knowledge_dir, output_dir=resolved_output_dir)


if __name__ == "__main__":
    from config import path
    root = path.PROJECT_ROOT
    indexer = FaissTopicIndexer(project_root=root, model_name=DEFAULT_FAISS_MODEL_NAME)

    out = indexer.build(
        knowledge_dir="knowledge",
        output_dir="save/faiss_topic_index",
    )

    print("Build done")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    sample_results = indexer.search(
        index_dir=root / "save" / "faiss_topic_index",
        query="gia ve tham quan dinh doc lap",
        top_k=3,
    )

    print("Sample search:")
    for r in sample_results:
        print(f"- rank={r['rank']} score={r['score']:.4f} chunk={r['chunk_id']}")
