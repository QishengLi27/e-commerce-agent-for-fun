from pydantic import BaseModel
from typing import List, Optional, Dict


class Source(BaseModel):
    type: str
    id: str
    snippet: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    response: str
    sources: List[Source] = []
    cached: bool = False
    latency_ms: Optional[int] = None
    validation_flag: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    checks: Optional[Dict[str, str]] = None
