from pathlib import Path

from core.indexing.faiss_topic_indexing import ensure_faiss_topic_index
from core.indexing.knowledge_chunking import ensure_chunking_prerequisites
from core.retrieval.topic_detector import ensure_topic_detector


DEFAULT_KNOWLEDGE_DIR = Path("knowledge")
DEFAULT_TOPIC_MAP_PATH = Path("knowledge/topic_maps.json")
DEFAULT_TOPIC_DETECTOR_DIR = Path("save/topic_detector")
DEFAULT_FAISS_INDEX_DIR = Path("save/faiss_topic_index")


def _count_markdown_files(knowledge_dir: Path) -> int:
    if not knowledge_dir.exists():
        return 0
    return sum(1 for _ in knowledge_dir.rglob("*.md"))


def _detector_artifacts_exist(detector_dir: Path) -> bool:
    detector_pkl = detector_dir / "detector.pkl"
    legacy_meta = detector_dir / "meta.json"
    legacy_embeds = detector_dir / "topic_embeds.npy"
    return detector_pkl.exists() or (legacy_meta.exists() and legacy_embeds.exists())


def _faiss_artifacts_exist(faiss_dir: Path) -> bool:
    index_file = faiss_dir / "topic_chunks.index"
    manifest_pkl = faiss_dir / "manifest.pkl"
    manifest_json = faiss_dir / "manifest.json"
    metadata_pkl = faiss_dir / "topic_metadata.pkl"
    metadata_json = faiss_dir / "topic_metadata.json"

    has_manifest = manifest_pkl.exists() or manifest_json.exists()
    has_metadata = metadata_pkl.exists() or metadata_json.exists()
    return index_file.exists() and has_manifest and has_metadata


def prepare_chat_engine_data(
    project_root: str | Path,
    knowledge_dir: str | Path = DEFAULT_KNOWLEDGE_DIR,
    topic_map_path: str | Path = DEFAULT_TOPIC_MAP_PATH,
    topic_detector_dir: str | Path = DEFAULT_TOPIC_DETECTOR_DIR,
    faiss_index_dir: str | Path = DEFAULT_FAISS_INDEX_DIR,
    force_rebuild: bool = False,
) -> dict:
    project_root = Path(project_root)

    resolved_knowledge_dir = Path(knowledge_dir)
    if not resolved_knowledge_dir.is_absolute():
        resolved_knowledge_dir = project_root / resolved_knowledge_dir

    resolved_topic_map_path = Path(topic_map_path)
    if not resolved_topic_map_path.is_absolute():
        resolved_topic_map_path = project_root / resolved_topic_map_path

    resolved_topic_detector_dir = Path(topic_detector_dir)
    if not resolved_topic_detector_dir.is_absolute():
        resolved_topic_detector_dir = project_root / resolved_topic_detector_dir

    resolved_faiss_index_dir = Path(faiss_index_dir)
    if not resolved_faiss_index_dir.is_absolute():
        resolved_faiss_index_dir = project_root / resolved_faiss_index_dir

    markdown_count = _count_markdown_files(resolved_knowledge_dir)
    if markdown_count == 0:
        raise ValueError(
            f"No markdown knowledge files found in: {resolved_knowledge_dir}. "
            "Please add .md files before starting chat engine."
        )

    chunking_prepare = ensure_chunking_prerequisites(
        project_root=project_root,
        topic_map_path=resolved_topic_map_path,
        topic_detector_dir=resolved_topic_detector_dir,
        force_rebuild_detector=force_rebuild,
    )

    detector_existed = _detector_artifacts_exist(resolved_topic_detector_dir)
    detector_build = ensure_topic_detector(
        project_root=project_root,
        topic_map_path=resolved_topic_map_path,
        save_dir=resolved_topic_detector_dir,
        force_rebuild=force_rebuild,
    )

    faiss_existed = _faiss_artifacts_exist(resolved_faiss_index_dir)
    faiss_build = ensure_faiss_topic_index(
        project_root=project_root,
        knowledge_dir=resolved_knowledge_dir,
        output_dir=resolved_faiss_index_dir,
        force_rebuild=force_rebuild,
    )

    return {
        "knowledge_dir": str(resolved_knowledge_dir),
        "knowledge_markdown_files": markdown_count,
        "topic_map_path": str(resolved_topic_map_path),
        "topic_detector_dir": str(resolved_topic_detector_dir),
        "faiss_index_dir": str(resolved_faiss_index_dir),
        "chunking_prepare": chunking_prepare,
        "detector_existed": detector_existed,
        "detector_built": detector_build is not None,
        "faiss_existed": faiss_existed,
        "faiss_built": faiss_build is not None,
    }


if __name__ == "__main__":
    from config import path

    report = prepare_chat_engine_data(project_root=path.PROJECT_ROOT)
    print("Startup data pipeline ready")
    print(report)
