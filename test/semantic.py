from core.loader.semanitic_loader import SemanticLoader

SEMANTIC_PATH = "save/dinh-doc-lap-intents-indexes-bge-m3.pkl"


def main():
    print("🚀 Loading Semantic Retriever...")

    semantic = SemanticLoader.load(SEMANTIC_PATH)

    print("✅ Semantic loaded")

    while True:
        query = input("\n💬 Nhập câu hỏi (exit để thoát): ")

        if query.lower() == "exit":
            break

        results = semantic.search(query, top_k=5)

        print("\n🧠 Top intents/resources:")
        for r, score in results:
            print(f"- {r}: {score:.4f}")


if __name__ == "__main__":
    main()