from core.retrieval.bm25 import BM25Retriever
from core.processing.preprocessing import preprocess
from utils.load_document import load_documents
from core.loader.bm25_loader import BM25Loader
import os

def build_bm25(documents):
    print("Building BM25 index...")
    bm25 = BM25Retriever(documents, preprocess)
    return bm25

def main():
    DATA_PATH = "knowledge/ho-chi-minh/dinh-doc-lap"
    SAVE_PATH = "save/dinh-doc-lap-knowledge-indexes-bm25.pkl"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # load data
    documents = load_documents(DATA_PATH)

    if not documents:
        print("No documents found!")
        return

    print(f"Total documents: {len(documents)}")

    # build BM25
    bm25 = build_bm25(documents)

    # save
    BM25Loader.save(bm25, SAVE_PATH)

    print(f"BM25 saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()