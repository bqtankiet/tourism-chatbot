
from chat.llm import LLM
from utils.load_document import load_documents
from core.retrieval.context_builder import ContextBuilder
from core.retrieval.retriever import Retriever
from core.retrieval.skip_retrieval import should_skip_retrieval
from chat.chat_response import ChatResponse

class ChatEngine:
    def __init__(self):
        PATH = 'knowledge/ho-chi-minh/dinh-doc-lap'

        self.documents = load_documents(PATH)
        self.retriever = Retriever(strategy='hybrid')
        self.context_builder = ContextBuilder(self.documents)

        self.llm = LLM(
            model="ollama/qwen3:0.6b"
        )

    def chat(self, query: str, threshold: float = 0.8) -> ChatResponse:
        query_clean = query.strip()

        skip = should_skip_retrieval(query_clean)

        results = []
        context = ""

        if not skip:
            results = self.retriever.search_threshold(query_clean, threshold=threshold)
            context = self.context_builder.build_text(results) if results else ""

        answer_tokens = []
        token_count = 0

        for token in self.llm.generate(query_clean, context, stream=True):
            answer_tokens.append(token)
            token_count += 1

        answer = "".join(answer_tokens)

        debug = {
            "should_skip_retrieval": skip,
            "num_results": len(results),
            "context_length": len(context),
            "threshold": threshold,
        }

        token_usage = {
            "output_tokens": token_count,
        }

        return ChatResponse(
            answer=answer,
            query=query_clean,
            context=context,
            results=results,
            token_usage=token_usage,
            debug=debug
        )

def chat(query: str) -> ChatResponse:
    global _engine
    if "_engine" not in globals():
        _engine = ChatEngine()

    return _engine.chat(query)