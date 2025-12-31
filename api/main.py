from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import time

from api.deps import get_assistant
from backend.assistant import MimirAssistant


# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Mimir API",
    version="1.1.0",
)

# ✅ CORS FIX (Vercel + Localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# =========================
# SCHEMAS
# =========================

class Message(BaseModel):
    role: str        # "user" | "mimir"
    content: str


class QueryRequest(BaseModel):
    query: Optional[str] = None
    messages: Optional[List[Message]] = None
    persona: str = "default"
    mode: str = "factual"


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float = 0.9
    metadata: dict = {}


# =========================
# QUERY (NON-STREAM)
# =========================

@app.post("/query", response_model=QueryResponse)
def query_mimir(
    payload: QueryRequest,
    assistant: MimirAssistant = Depends(get_assistant),
):
    """
    Memory-aware query endpoint.

    Priority:
    1) messages (conversation memory)
    2) single query fallback
    """

    # 🔁 MEMORY PATH
    if payload.messages and len(payload.messages) > 0:
        return assistant.query_with_memory(
            messages=[m.dict() for m in payload.messages],
            persona=payload.persona,
            mode=payload.mode,
        )

    # 🔁 LEGACY PATH
    if payload.query and payload.query.strip():
        return assistant.query(
            payload.query,
            payload.persona,
            payload.mode,
        )

    # ❌ EMPTY INPUT
    return QueryResponse(
        answer="No input provided.",
        confidence=0.2,
        metadata={"error": "empty_request"},
    )


# =========================
# QUERY (STREAMING)
# =========================

@app.post("/query/stream")
def query_stream(
    payload: QueryRequest,
    assistant: MimirAssistant = Depends(get_assistant),
):
    def stream():
        # MEMORY PATH
        if payload.messages and len(payload.messages) > 0:
            result = assistant.query_with_memory(
                messages=[m.dict() for m in payload.messages],
                persona=payload.persona,
                mode=payload.mode,
            )

        # LEGACY PATH
        elif payload.query and payload.query.strip():
            result = assistant.query(
                payload.query,
                payload.persona,
                payload.mode,
            )

        else:
            result = {"answer": "No input provided."}

        # Oracle-style token streaming
        for token in result["answer"].split(" "):
            yield token + " "
            time.sleep(0.03)

    return StreamingResponse(stream(), media_type="text/plain")
