from chat.llm import LLM

from utils.load_document import load_documents
from core.retrieval.context_builder import ContextBuilder
from core.retrieval.retriever import Retriever
from core.retrieval.skip_retrieval import should_skip_retrieval

PATH = 'knowledge/ho-chi-minh/dinh-doc-lap'
documents = load_documents(PATH)

retriever = Retriever(strategy='hybrid')

context_builder = ContextBuilder(documents)

def main():
    print("Loading LLM...")

    llm = LLM(
        model="ollama/qwen3:0.6b"
    )

    print("LLM ready!")

    context = ""

    while True:
        query = input("\n💬 Nhập câu hỏi (exit để thoát): ")

        if query.lower() == "exit":
            break
        
        results = []
        context = ""
        if not should_skip_retrieval(query):
            results = retriever.search_threshold(query, threshold = 0.8)
            context = context_builder.build_text(results) if results else ""

        print("\n⚙️ DEBUG: \n")
        print(f"should_skip_retrieval: {should_skip_retrieval(query)}")
        print(f"retriever_result: {results}")
        print(f"context_length: {len(context)}")
        print("---"*10)

        print("\n🤖 Trả lời:\n")
        for token in llm.generate(query, context, stream=True):
            print(token, end="", flush=True)
        print("\n")
        print("---"*10)


if __name__ == "__main__":
    main()