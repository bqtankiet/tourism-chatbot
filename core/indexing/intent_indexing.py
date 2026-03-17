import os
import json
from core.retrieval.semantic import SemanticRetriever
from core.loader.semanitic_loader import SemanticLoader


def load_resources(path):
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_semantic(resources, model_name="BAAI/bge-m3"):
    print("Building Semantic embeddings...")
    return SemanticRetriever(resources, model_name=model_name)


def main():
    DATA_PATH = "knowledge/ho-chi-minh/dinh-doc-lap/intents.json"
    SAVE_PATH = "save/dinh-doc-lap-intents-indexes-bge-m3.pkl"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # nếu đã tồn tại thì skip
    if os.path.exists(SAVE_PATH):
        print("Semantic index already exists. Skip building.")
        return

    # load intents/resources
    resources = load_resources(DATA_PATH)

    if not resources:
        print("No resources found!")
        return

    print(f"Total resources: {len(resources)}")

    # build semantic
    semantic = build_semantic(resources)

    # save
    SemanticLoader.save(semantic, SAVE_PATH)

    print(f"Semantic embeddings saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()