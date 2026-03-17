from core.loader.semanitic_loader import SemanticLoader
from core.loader.bm25_loader import BM25Loader
from core.processing.preprocessing import preprocess
from core.retrieval.hybrid import HybridRetriever

BM25_PATH = "save/dinh-doc-lap-knowledge-indexes-bm25.pkl"
SEMANTIC_PATH = "save/dinh-doc-lap-intents-indexes-bge-m3.pkl"

def main():
    print("Loading BM25 Retriever...")
    bm25 = BM25Loader.load(BM25_PATH, preprocess)
    print("Loading Semantic Retriever...")
    semantic = SemanticLoader.load(SEMANTIC_PATH)
    
    hybrid = HybridRetriever(semantic, bm25)
    print("Hybrid Retriever created")

    while True:
        query = input("\n💬 Nhập câu hỏi (exit để thoát): ")

        if query.lower() == "exit":
            break

        results = hybrid.search(query, top_k=5)

        print("\n📄 Top documents:")
        for doc, score in results:
            print(f"- {doc}: {score:.4f}")


if __name__ == "__main__":
    main()