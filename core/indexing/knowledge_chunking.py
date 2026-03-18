import json
from pathlib import Path
from typing import Any
from core.retrieval.topic_detector import TopicDetector, ensure_topic_detector


try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:
    raise ImportError(
        "Missing LangChain splitters. Install package: langchain-text-splitters"
    ) from exc


class KnowledgeTopicChunker:
    """Chunk markdown knowledge files by topic using LangChain splitters.

    Topic source: knowledge/topic_maps.json
    """

    DEFAULT_TOPIC_MAP_PATH = Path("knowledge/topic_maps.json")
    DEFAULT_TOPIC_DETECTOR_DIR = Path("save/topic_detector")
    DEFAULT_TOPIC_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        project_root: str | Path,
        topic_map_path: str | Path | None = None,
        topic_detector_dir: str | Path | None = None,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ):
        self.project_root = Path(project_root)
        self.topic_map_path = (
            Path(topic_map_path)
            if topic_map_path is not None
            else self.project_root / self.DEFAULT_TOPIC_MAP_PATH
        )
        self.topic_detector_dir = (
            Path(topic_detector_dir)
            if topic_detector_dir is not None
            else self.project_root / self.DEFAULT_TOPIC_DETECTOR_DIR
        )

        ensure_chunking_prerequisites(
            project_root=self.project_root,
            topic_map_path=self.topic_map_path,
            topic_detector_dir=self.topic_detector_dir,
            topic_model=self.DEFAULT_TOPIC_MODEL,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.topic_map = self._load_topic_map(self.topic_map_path)

        if not self.topic_detector_dir.exists():
            raise ValueError(
                f"Topic detector directory not found: {self.topic_detector_dir}. "
                f"Please build/load save/topic_detector first."
            )

        self.topic_detector = TopicDetector.load(str(self.topic_detector_dir))

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    @staticmethod
    def _load_topic_map(topic_map_path: Path) -> dict[str, Any]:
        if not topic_map_path.exists():
            raise ValueError(f"Topic map not found: {topic_map_path}")

        with open(topic_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or not data:
            raise ValueError("topic_maps.json is empty or invalid")

        return data

    def _detect_topic_key(self, text: str) -> str:
        if not text.strip():
            return "unknown"

        detected = self.topic_detector.detect(text, top_k=1, threshold=0.0)
        if not detected:
            return "unknown"

        return detected[0].get("topic", "unknown")

    def _topic_query_count(self, topic_key: str) -> int:
        topic_data = self.topic_map.get(topic_key, {})
        if isinstance(topic_data, dict):
            queries = topic_data.get("queries", [])
            if isinstance(queries, list):
                return len(queries)
        return 0

    @staticmethod
    def _strip_context_prefix(raw_text: str) -> str:
        marker = "[CONTENT]"
        if marker in raw_text:
            return raw_text.split(marker, 1)[1].strip()
        return raw_text.strip()

    def _extract_path_metadata(self, file_path: Path, knowledge_root: Path) -> dict[str, str]:
        rel = file_path.relative_to(knowledge_root)
        parts = list(rel.parts)

        city = parts[0] if len(parts) > 0 else ""
        place = parts[1] if len(parts) > 1 else ""

        return {
            "city": city,
            "place": place,
            "source_file": file_path.name,
            "source_rel_path": str(rel).replace("\\", "/"),
        }

    def chunk_file(self, file_path: Path, knowledge_root: Path) -> list[dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        text = self._strip_context_prefix(raw_text)
        file_topic_key = self._detect_topic_key(text)

        base_meta = self._extract_path_metadata(file_path, knowledge_root)
        base_meta.update(
            {
                "topic_key": file_topic_key,
                "topic_queries_count": self._topic_query_count(file_topic_key),
            }
        )

        header_docs = self.header_splitter.split_text(text)
        chunks: list[dict[str, Any]] = []
        chunk_idx = 0

        for doc in header_docs:
            section_text = doc.page_content
            section_meta = dict(base_meta)
            section_meta.update(doc.metadata)

            split_docs = self.text_splitter.create_documents(
                texts=[section_text],
                metadatas=[section_meta],
            )

            for split_doc in split_docs:
                md = dict(split_doc.metadata)
                detected_topic = self._detect_topic_key(split_doc.page_content)
                md["topic_key"] = detected_topic
                md["topic_queries_count"] = self._topic_query_count(detected_topic)
                md["chunk_index"] = chunk_idx
                md["chunk_id"] = (
                    f"{md.get('source_rel_path', file_path.name)}::"
                    f"{md.get('topic_key', 'unknown')}::{chunk_idx}"
                )

                chunks.append(
                    {
                        "id": md["chunk_id"],
                        "text": split_doc.page_content,
                        "metadata": md,
                    }
                )
                chunk_idx += 1

        if not chunks:
            fallback_meta = dict(base_meta)
            fallback_topic = self._detect_topic_key(text)
            fallback_meta["topic_key"] = fallback_topic
            fallback_meta["topic_queries_count"] = self._topic_query_count(fallback_topic)
            fallback_meta["chunk_index"] = 0
            fallback_meta["chunk_id"] = (
                f"{fallback_meta.get('source_rel_path', file_path.name)}::"
                f"{fallback_topic}::0"
            )
            chunks.append(
                {
                    "id": fallback_meta["chunk_id"],
                    "text": text,
                    "metadata": fallback_meta,
                }
            )

        return chunks

    def chunk_knowledge(self, knowledge_dir: str | Path) -> list[dict[str, Any]]:
        knowledge_root = Path(knowledge_dir)
        if not knowledge_root.is_absolute():
            knowledge_root = self.project_root / knowledge_root

        if not knowledge_root.exists():
            raise ValueError(f"Knowledge directory not found: {knowledge_root}")

        all_chunks: list[dict[str, Any]] = []
        for md_file in sorted(knowledge_root.rglob("*.md")):
            all_chunks.extend(self.chunk_file(md_file, knowledge_root))

        return all_chunks

    @staticmethod
    def save_chunks_json(output_path: str | Path, chunks: list[dict[str, Any]]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


def ensure_chunking_prerequisites(
    project_root: str | Path,
    topic_map_path: str | Path = KnowledgeTopicChunker.DEFAULT_TOPIC_MAP_PATH,
    topic_detector_dir: str | Path = KnowledgeTopicChunker.DEFAULT_TOPIC_DETECTOR_DIR,
    topic_model: str = KnowledgeTopicChunker.DEFAULT_TOPIC_MODEL,
    force_rebuild_detector: bool = False,
) -> dict:
    project_root = Path(project_root)

    resolved_topic_map_path = Path(topic_map_path)
    if not resolved_topic_map_path.is_absolute():
        resolved_topic_map_path = project_root / resolved_topic_map_path

    resolved_detector_dir = Path(topic_detector_dir)
    if not resolved_detector_dir.is_absolute():
        resolved_detector_dir = project_root / resolved_detector_dir

    resolved_topic_map_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_detector_dir.mkdir(parents=True, exist_ok=True)

    created_topic_map = False
    if not resolved_topic_map_path.exists():
        # Minimal bootstrap topic map so chunking pipeline can initialize.
        bootstrap_topic_map = {
            "unknown": {
                "queries": [
                    "thong tin du lich",
                    "gioi thieu dia diem",
                    "chi tiet dia diem",
                ]
            }
        }
        with open(resolved_topic_map_path, "w", encoding="utf-8") as f:
            json.dump(bootstrap_topic_map, f, ensure_ascii=False, indent=2)
        created_topic_map = True

    detector_result = ensure_topic_detector(
        project_root=project_root,
        topic_map_path=resolved_topic_map_path,
        save_dir=resolved_detector_dir,
        model=topic_model,
        force_rebuild=force_rebuild_detector,
    )

    return {
        "topic_map_path": str(resolved_topic_map_path),
        "topic_detector_dir": str(resolved_detector_dir),
        "created_topic_map": created_topic_map,
        "detector_built": detector_result is not None,
    }


if __name__ == "__main__":
    from config import path
    root = path.PROJECT_ROOT
    prep = ensure_chunking_prerequisites(project_root=root)
    chunker = KnowledgeTopicChunker(project_root=root)

    chunks = chunker.chunk_knowledge("knowledge")
    output_file = root / "save" / "knowledge_topic_chunks.json"
    chunker.save_chunks_json(output_file, chunks)

    print(f"Prepare: {prep}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {output_file}")
