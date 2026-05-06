from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from setup_db import get_order_status
from retrieval import get_policy_retriever
from memory import MemoryStore
from resilience import (
    CircuitBreaker,
    Fallbacks,
    make_retry_decorator,
)
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Shared Config ────────────────────────────────────────────────────────────

PG_CONNECTION = "postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce"

# ─── Embeddings (shared) ──────────────────────────────────────────────────────

cache_embeddings = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

# ─── Semantic Cache (pgvector) ────────────────────────────────────────────────

cache_vectorstore = PGVector(
    connection_string=PG_CONNECTION,
    embedding_function=cache_embeddings,
    collection_name="semantic_cache",
    distance_strategy="cosine",
)


def get_cached_response(query: str):
    docs_and_scores = cache_vectorstore.similarity_search_with_score(query, k=1)
    # pgvector COSINE distance: lower = more similar
    if docs_and_scores and docs_and_scores[0][1] < 0.3:
        return docs_and_scores[0][0].metadata.get("response")
    return None


def cache_response(query: str, response: str):
    cache_vectorstore.add_texts([query], metadatas=[{"response": response}])


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

_policy_retriever = get_policy_retriever()
memory_store = MemoryStore(filepath="memory_store.json", max_history=8)


# ─── Typo Correction ──────────────────────────────────────────────────────────

@make_retry_decorator(max_attempts=2)
def _clean_query_api_call(prompt: str):
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


# ─── Tools ────────────────────────────────────────────────────────────────────

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


# ─── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    max_retries=2,
    timeout=30,
)

llm_circuit = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)


# ─── Agent ────────────────────────────────────────────────────────────────────

tools = [order_status_tool, policy_retriever_tool]

system_prompt_template = """
You are a helpful e-commerce support agent for an online store.
You have access to the following tools:

{tools}

When you are not sure, answer honestly and do not hallucinate.
Use the exact available tools for order or policy lookup when needed.

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation history:
{conversation_history}

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(
    system_prompt_template,
    input_variables=["input", "conversation_history"],
)

agent = create_react_agent(llm, tools, prompt)

_agent_call_count = 0
_agent_call_reset_time = time.time()


def extract_agent_response(result: object) -> str:
    """Extract the final AI message content from agent result."""
    if isinstance(result, dict):
        if "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return str(last.content)
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

    cleaned_input = clean_query(user_input)
    memory_store.add_user(cleaned_input)

    cached = get_cached_response(cleaned_input)
    if cached:
        print("Cache hit!")
        memory_store.add_agent(cached)
        return cached

    memory_messages = memory_store.get_recent_messages()
    conversation_history = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in memory_messages]
    ) if memory_messages else "No prior conversation."

    def _invoke_agent():
        return agent.invoke(
            {
                "input": cleaned_input,
                "conversation_history": conversation_history,
            }
        )

    try:
        result = llm_circuit.call(_invoke_agent, Fallbacks.llm_unavailable)
        response = extract_agent_response(result)

        if response == Fallbacks.llm_unavailable():
            return response

        cache_response(cleaned_input, response)
        memory_store.add_agent(response)
        return response

    except Exception as e:
        logger.error(f"[agent] Invocation failed: {e}")
        from resilience import is_transient_error
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
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == "quit":
            break
        response = run_agent_with_cache(user_input)
        print(response)
