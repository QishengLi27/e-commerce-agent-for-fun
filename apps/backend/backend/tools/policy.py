from langchain.tools import tool
from backend.config import settings
from backend.knowledge.retrievers import create_policy_retriever


@tool
def policy_retriever_tool(query: str) -> str:
    """Retrieve store policies related to the query.
    Uses the configured retrieval mode (vector | graph | hybrid)."""
    retriever = create_policy_retriever(settings.retrieval_mode)
    result = retriever.retrieve(query)
    if not result:
        return "No relevant policy information found."
    return result
