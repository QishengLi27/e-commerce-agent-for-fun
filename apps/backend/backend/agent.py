import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- Shared Config -------------------------------------------------------------

PG_CONNECTION = "postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce"

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from backend.memory import MemoryStore
from backend.resilience import (
    CircuitBreaker,
    Fallbacks,
    make_retry_decorator,
)
from backend.tools import order_status_tool, list_orders_tool, policy_retriever_tool, get_current_weather

# -- Lazy-loaded resources -----------------------------------------------------
# We defer DB-dependent initialization so the API server can start without
# a live Postgres connection.

_cache_vectorstore = None
_policy_retriever = None
memory_store = MemoryStore(filepath="data/memory_store.json", max_history=8)


def _get_cache_vectorstore():
    global _cache_vectorstore
    if _cache_vectorstore is None:
        _cache_vectorstore = PGVector(
            connection_string=PG_CONNECTION,
            embedding_function=OpenAIEmbeddings(
                model="embedding-2",
                openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
                openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
            ),
            collection_name="semantic_cache",
            distance_strategy="cosine",
        )
    return _cache_vectorstore


# -- Semantic Cache (pgvector) -------------------------------------------------


def get_cached_response(query: str):
    docs_and_scores = _get_cache_vectorstore().similarity_search_with_score(query, k=1)
    # pgvector COSINE distance: lower = more similar
    if docs_and_scores and docs_and_scores[0][1] < 0.3:
        return docs_and_scores[0][0].metadata.get("response")
    return None


def cache_response(query: str, response: str):
    _get_cache_vectorstore().add_texts([query], metadatas=[{"response": response}])


# -- LLM -----------------------------------------------------------------------

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    max_retries=2,
    timeout=30,
)

llm_circuit = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)


# -- Agent ----------------------------------------------------------------------

tools = [order_status_tool, list_orders_tool, policy_retriever_tool, get_current_weather]

# Create agent without system prompt - GLM-4 doesn't support system messages well
agent = create_agent(llm, tools)

_agent_call_count = 0
_agent_call_reset_time = time.time()


def extract_agent_response(result: object) -> str:
    """Extract the final AI message content from agent executor result."""
    raw = ""
    if isinstance(result, dict):
        if "output" in result:
            raw = str(result["output"])
        elif "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    raw = str(last.content)
                elif isinstance(last, dict) and "content" in last:
                    raw = str(last["content"])
    else:
        raw = str(result)

    # ReAct agents sometimes return the full reasoning chain.
    # Extract only the Final Answer if present.
    if "Final Answer:" in raw:
        return raw.split("Final Answer:")[-1].strip()

    return raw


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


@make_retry_decorator(max_attempts=2)
def _clean_query_api_call(prompt: str):
    return llm.invoke(prompt)


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
        logger.error(f"[agent] Exceeded max calls ({max_agent_calls}) -- possible loop")
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
        # GLM-4 works with messages format, put instructions in the human message
        full_content = f"""You are a helpful e-commerce support agent for an online store.

You have access to the following tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in tools])}

When you are not sure, answer honestly and do not hallucinate.
Use the exact available tools for order, policy, or weather lookup when needed.
- For questions about a specific order, use order_status_tool.
- For questions about all orders, use list_orders_tool.
- For questions about store policies, use policy_retriever_tool.
- For questions about weather in a specific city, use get_current_weather.

Use the following format for tool use:
Thought: you should always think about what to do
Action: the action to take, should be one of {', '.join([tool.name for tool in tools])}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation history:
{conversation_history}

Question: {cleaned_input}"""

        return agent.invoke(
            {
                "messages": [HumanMessage(content=full_content)]
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
        from backend.resilience import is_transient_error
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
