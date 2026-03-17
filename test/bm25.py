from core.loader.bm25_loader import BM25Loader
from core.processing.preprocessing import preprocess

BM25_PATH = "save/dinh-doc-lap-knowledge-indexes-bm25.pkl"


def main():
    print("🚀 Loading BM25 Retriever...")

    bm25 = BM25Loader.load(BM25_PATH, preprocess)

    print("✅ BM25 loaded")

    while True:
        query = input("\n💬 Nhập câu hỏi (exit để thoát): ")

        if query.lower() == "exit":
            break

        results = bm25.search(query, top_k=5)

        print("\n📄 Top documents:")
        for doc, score in results:
            print(f"- {doc}: {score:.4f}")


if __name__ == "__main__":
    main()