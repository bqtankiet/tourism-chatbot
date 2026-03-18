from contextlib import asynccontextmanager
from typing import Generator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.chat_engine import ChatEngine


chat_engine: ChatEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_engine
    chat_engine = ChatEngine()
    yield
    chat_engine = None


app = FastAPI(
    title="Travel Chatbot API",
    description="Chat bot tư vấn du lịch Việt Nam",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str
    threshold: float = 0.8


class ChatResponseBody(BaseModel):
    answer: str
    query: str
    debug: dict
    token_usage: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "startup": getattr(chat_engine, "startup_report", None),
    }


@app.post("/chat", response_model=ChatResponseBody)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query không được để trống")

    res = chat_engine.chat(req.query, threshold=req.threshold)

    return ChatResponseBody(
        answer=res.answer,
        query=res.query,
        debug=res.debug,
        token_usage=res.token_usage,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query không được để trống")

    def event_generator():
        for token in chat_engine.chat_stream(
            req.query,
            threshold=req.threshold
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
