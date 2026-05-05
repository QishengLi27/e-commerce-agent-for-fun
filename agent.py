from langchain_openai import ChatOpenAI
from langchain_community.cache import InMemoryCache
# from langchain_community.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
# from langchain.chains import RetrievalQA
from setup_db import get_order_status
from retrieval import get_policy_retriever
from resilience import (
    CircuitBreaker,
    Fallbacks,
    make_retry_decorator,
    is_transient_error,
)
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up semantic cache for responses
cache_embeddings = OpenAIEmbeddings(
    model="embedding-2",  # or "embedding-3"
    openai_api_key = "51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU", # os.getenv("ZHIPUAI_API_KEY"), 
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/" # GLM's OpenAI-compatible endpoint
)
cache_vectorstore = Chroma(collection_name="semantic_cache", embedding_function=cache_embeddings, persist_directory="./cache_db")

def get_cached_response(query: str):
    docs_and_scores = cache_vectorstore.similarity_search_with_score(query, k=1)
    if docs_and_scores and docs_and_scores[0][1] < 0.3:  # Lower score means more similar in Chroma (cosine distance)
        return docs_and_scores[0][0].metadata.get('response')
    return None

def cache_response(query: str, response: str):
    cache_vectorstore.add_texts([query], metadatas=[{"response": response}])

# Load hybrid retriever (Chroma + BM25 + RRF + optional LLM re-rank)
_policy_retriever = get_policy_retriever()


@make_retry_decorator(max_attempts=2)
def _clean_query_api_call(prompt: str):
    """Inner function for the actual API call, wrapped with retry."""
    return llm.invoke(prompt)


def clean_query(query: str) -> str:
    """Fix typos and normalize the user query before retrieval."""
    prompt = (
        "Fix any spelling mistakes in the user query. "
        "Output ONLY the corrected query with no explanation, no quotes, and no extra text.\n\n"
        f"User query: {query}\n"
        "Corrected query:"
    )
    try:
        response = llm_circuit.call(
            _clean_query_api_call,
            Fallbacks.clean_query_failed,
            prompt,
        )
        cleaned = response.content.strip().strip('"').strip("'")
        return cleaned if cleaned else query
    except Exception as e:
        logger.warning(f"[clean_query] Failed after retries: {e}")
        return query


@tool
def order_status_tool(order_id: str) -> str:
    """Get the status of an order by order ID."""
    return get_order_status(order_id)


@tool
def policy_retriever_tool(query: str) -> str:
    """Retrieve store policies related to the query using hybrid search."""
    docs_and_scores = _policy_retriever.retrieve(query, k=3, rerank=True)
    # Only include highly relevant chunks to avoid flooding the LLM with noise
    filtered = [(doc, score) for doc, score in docs_and_scores if score >= 7]
    if not filtered:
        filtered = docs_and_scores[:1]  # fallback to top-1 if nothing scores high
    return "\n\n".join([doc.page_content for doc, _ in filtered])

# Initialize LLM with built-in retry (2 retries) and timeout
# Note: ChatOpenAI doesn't support timeout directly, but we add circuit breaker below
llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key = "51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/",
    max_retries=2,  # LangChain built-in retry for transient errors
    timeout=30,     # seconds per API call
)

# Circuit breaker for LLM API calls
llm_circuit = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)

# Track total agent invocations to detect runaway loops
_agent_call_count = 0
_agent_call_reset_time = time.time()

# Define tools
tools = [order_status_tool, policy_retriever_tool]

# System prompt
system_prompt = """
You are a helpful e-commerce support agent. Use the order_status_tool for order statuses and the policy_retriever_tool for store policy questions.
"""

# Create agent
agent = create_agent(llm, tools, system_prompt=system_prompt)

def extract_agent_response(result: object) -> str:
    """Extract the final AI message content from agent result."""
    if isinstance(result, dict):
        if "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and messages:
                last = messages[-1]
                # LangChain message objects have .content attribute
                if hasattr(last, "content"):
                    return str(last.content)
                # Fallback for dict-style messages
                if isinstance(last, dict) and "content" in last:
                    return str(last["content"])
        if "output" in result:
            return str(result["output"])
    return str(result)


def run_agent_with_cache(user_input: str, max_agent_calls: int = 5):
    """
    Run the agent with resilience patterns:
    - Typo correction with retry + circuit breaker
    - Agent invocation with retry + circuit breaker
    - Max call limit to prevent infinite loops
    """
    global _agent_call_count, _agent_call_reset_time

    # Reset counter every 60 seconds
    if time.time() - _agent_call_reset_time > 60:
        _agent_call_count = 0
        _agent_call_reset_time = time.time()

    _agent_call_count += 1
    if _agent_call_count > max_agent_calls:
        logger.error(f"[agent] Exceeded max calls ({max_agent_calls}) — possible loop")
        return (
            "I'm having trouble processing your request right now. "
            "Please try rephrasing your question or contact support."
        )

    # Step 1: Fix typos (with retry + circuit breaker)
    cleaned_input = clean_query(user_input)

    # Step 2: Check semantic cache
    cached = get_cached_response(cleaned_input)
    if cached:
        print("Cache hit!")
        return cached

    # Step 3: Run agent with circuit breaker protection
    def _invoke_agent():
        return agent.invoke({"messages": [{"role": "user", "content": cleaned_input}]})

    try:
        result = llm_circuit.call(_invoke_agent, Fallbacks.llm_unavailable)
        response = extract_agent_response(result)

        # Don't cache error/fallback messages
        if response == Fallbacks.llm_unavailable():
            return response

        # Cache the response
        cache_response(cleaned_input, response)
        return response

    except Exception as e:
        logger.error(f"[agent] Invocation failed: {e}")
        if is_transient_error(e):
            return (
                "I'm experiencing a temporary issue. "
                "Please try again in a moment."
            )
        return (
            "I couldn't process your request. "
            "Please check your question and try again."
        )

if __name__ == "__main__":
    # Example usage
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == 'quit':
            break
        response = run_agent_with_cache(user_input)
        print(response)
