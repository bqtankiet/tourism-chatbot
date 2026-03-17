from utils.load_document import load_documents
from core.loader.bm25_loader import BM25Loader
from core.loader.semanitic_loader import SemanticLoader
from core.retrieval.hybrid import HybridRetriever
from core.processing.preprocessing import preprocess
from core.retrieval.context_builder import ContextBuilder

PATH = 'knowledge/ho-chi-minh/dinh-doc-lap'
documents = load_documents(PATH)

BM25_PATH = "save/dinh-doc-lap-knowledge-indexes-bm25.pkl"
SEMANTIC_PATH = "save/dinh-doc-lap-intents-indexes-bge-m3.pkl"
print("Loading BM25 Retriever...")
bm25 = BM25Loader.load(BM25_PATH, preprocess)
print("Loading Semantic Retriever...")
semantic = SemanticLoader.load(SEMANTIC_PATH)
retriever = HybridRetriever(semantic, bm25)

context_builder = ContextBuilder(documents)

while True:
    query = input("\n💬 Nhập câu hỏi (exit để thoát): ")
    if query.lower() == "exit":
            break
    results = retriever.search(query, top_k=5)
    text = context_builder.build_text(results)
    print(results)
    print(text)