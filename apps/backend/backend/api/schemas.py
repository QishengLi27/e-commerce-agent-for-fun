
from pydantic import BaseModel


class Source(BaseModel):
    type: str
    id: str
    snippet: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str | None = None
    response: str
    sources: list[Source] = []
    cached: bool = False
    latency_ms: int | None = None
    validation_flag: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    checks: dict[str, str] | None = None
