# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str   # "user" | "mimir"
    content: str

class QueryRequest(BaseModel):
    messages: Optional[List[Message]] = None
    query: Optional[str] = None
    persona: str = "default"
    mode: str = "factual"

class QueryResponse(BaseModel):
    answer: str
    confidence: float = 0.9
    metadata: dict = {}
