
from chat.llm import LLM
from config import path
from core.indexing.faiss_topic_indexing import ensure_faiss_topic_index
from core.retrieval.retriever import Retriever
from core.retrieval.skip_retrieval import should_skip_retrieval
from chat.chat_response import ChatResponse
from core.loader.model_cache import get_or_create_model

class ChatEngine:
    def __init__(self):
        ensure_faiss_topic_index(project_root=path.PROJECT_ROOT)
        self.retriever = Retriever()

        self.llm = get_or_create_model(
            cache_key=("LLM", "ollama/qwen3:0.6b", 0.3, 1024),
            factory=lambda: LLM(model="ollama/qwen3:0.6b"),
        )

    def chat(self, query: str, threshold=0.5, top_k=20) -> ChatResponse:
        query_clean = query.strip()

        skip = should_skip_retrieval(query_clean)

        results = []
        context = ""

        if not skip:
            results = self.retriever.search_threshold(query_clean, threshold=threshold, top_k=top_k)
            context = self.retriever.build_context(results) if results else ""

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
            "retrieval_engine": "faiss_topic_index",
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
    
    from typing import Generator
    def chat_stream(
        self, query: str, threshold=0.5, top_k=20
        ) -> Generator[str, None, None]:

        query_clean = query.strip()
        skip = should_skip_retrieval(query_clean)

        results = []
        context = ""

        if not skip:
            results = self.retriever.search_threshold(
                query_clean, threshold=threshold, top_k=top_k
            )
            context = self.retriever.build_context(results) if results else ""

        # stream token
        for token in self.llm.generate(query_clean, context, stream=True):
            yield token

        yield "[DONE]"