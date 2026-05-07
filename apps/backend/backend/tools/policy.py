from langchain.tools import tool
from backend.retrieval import get_policy_retriever


@tool
def policy_retriever_tool(query: str) -> str:
    """Retrieve store policies related to the query using hybrid search."""
    retriever = get_policy_retriever()
    docs_and_scores = retriever.retrieve(query, k=3, rerank=True)
    # Only include highly relevant chunks to avoid flooding the LLM with noise
    filtered = [(doc, score) for doc, score in docs_and_scores if score >= 7]
    if not filtered:
        filtered = docs_and_scores[:1]  # fallback to top-1 if nothing scores high
    return "\n\n".join([doc.page_content for doc, _ in filtered])
