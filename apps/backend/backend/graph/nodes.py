"""
LangGraph nodes for the e-commerce support agent.

Each node is a pure function: AgentState -> AgentState.
"""

import re
import logging
from typing import Literal, TypedDict, Annotated
from operator import add

from langchain_core.messages import HumanMessage

from backend.agent import (
    clean_query,
    get_cached_response,
    cache_response,
    memory_store,
    llm,
)
from backend.tools import (
    order_status_tool,
    list_orders_tool,
    policy_retriever_tool,
    get_current_weather,
)

logger = logging.getLogger(__name__)


# ─── State Type ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """LangGraph state schema."""
    messages: Annotated[list, add]
    user_input: str
    intent: str
    order_id: str
    tool_result: str
    final_answer: str
    cached: bool
    validation_flag: str
    validation_notes: str


# ─── Node: sanitize_input ────────────────────────────────────────────────────

# Weather keywords to skip semantic cache (real-time data should not be cached)
_WEATHER_KEYWORDS = {"weather", "temperature", "rain", "sunny", "cloudy", "forecast", "天气"}


def _is_weather_query(text: str) -> bool:
    return any(w in text.lower() for w in _WEATHER_KEYWORDS)


def sanitize_input(state: AgentState) -> AgentState:
    """Clean user input and check semantic cache (skips cache for weather queries)."""
    raw = state.get("user_input", "")
    cleaned = clean_query(raw)
    state["user_input"] = cleaned

    # Skip semantic cache for weather — embeddings of "weather in X" and "weather in Y"
    # are too similar, causing false cache hits across different cities.
    if _is_weather_query(cleaned):
        state["cached"] = False
        logger.info("[graph] Weather query detected, skipping semantic cache")
        return state

    cached = get_cached_response(cleaned)
    if cached:
        state["final_answer"] = cached
        state["cached"] = True
        logger.info("[graph] Cache hit for: %s", cleaned[:50])
    else:
        state["cached"] = False

    return state


# ─── Node: classify_intent ───────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """
    Fast keyword-based intent classification.
    Returns: order | list_orders | policy | weather | unknown
    """
    text = state.get("user_input", "").lower()

    # 1. Weather (most specific)
    if any(w in text for w in ["weather", "天气", "temperature", "rain", "sunny"]):
        state["intent"] = "weather"
        logger.info("[graph] Intent: weather")
        return state

    # 2. List all orders
    if any(w in text for w in ["all orders", "订单列表", "show me orders", "list orders"]):
        state["intent"] = "list_orders"
        logger.info("[graph] Intent: list_orders")
        return state

    # 3. Single order status
    if any(w in text for w in ["order", "订单", "status of", "track"]):
        match = re.search(r"\b(10\d{2,})\b", text)
        if match:
            state["order_id"] = match.group(1)
            state["intent"] = "order"
            logger.info("[graph] Intent: order, id=%s", match.group(1))
            return state

    # 4. Policy / returns
    if any(w in text for w in ["policy", "return", "refund", "shipping", "warranty", "退货", "退款", "政策", "运费"]):
        state["intent"] = "policy"
        logger.info("[graph] Intent: policy")
        return state

    # 5. Fallback
    state["intent"] = "unknown"
    logger.info("[graph] Intent: unknown")
    return state


# ─── Router ──────────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> Literal[
    "order", "list_orders", "policy", "weather", "generate_reply"
]:
    """Conditional edge: decide next node based on intent."""
    if state.get("cached"):
        logger.info("[graph] Route: cached -> generate_reply")
        return "generate_reply"

    intent = state.get("intent", "unknown")
    if intent in ("order", "list_orders", "policy", "weather"):
        logger.info("[graph] Route: %s", intent)
        return intent

    logger.info("[graph] Route: unknown -> generate_reply")
    return "generate_reply"


# ─── Nodes: Tool Execution ───────────────────────────────────────────────────

def order_node(state: AgentState) -> AgentState:
    """Query a single order by ID."""
    order_id = state.get("order_id")
    if not order_id:
        state["tool_result"] = "No order ID provided."
        return state
    result = order_status_tool.invoke({"order_id": order_id})
    state["tool_result"] = result
    logger.info("[graph] Order result: %s", result[:60])
    return state


def list_orders_node(state: AgentState) -> AgentState:
    """List all orders."""
    result = list_orders_tool.invoke({})
    state["tool_result"] = result
    logger.info("[graph] List orders result: %s", result[:60])
    return state


def policy_node(state: AgentState) -> AgentState:
    """Retrieve store policies."""
    query = state.get("user_input", "")
    result = policy_retriever_tool.invoke({"query": query})
    state["tool_result"] = result
    logger.info("[graph] Policy result: %s", result[:60])
    return state


def _extract_city_with_llm(query: str) -> str:
    """Use the LLM to extract a clean city name from a weather query."""
    prompt = (
        "Extract ONLY the city name from the following weather query. "
        "Return just the city name with no explanation, no quotes, and no extra text.\n\n"
        f"Query: {query}\n"
        "City:"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        city = response.content.strip().strip('"').strip("'")
        return city if city else query
    except Exception:
        # Fallback: return the original query and let the tool handle extraction
        return query


def weather_node(state: AgentState) -> AgentState:
    """Get weather for a city."""
    text = state.get("user_input", "")
    # Use LLM for robust city extraction (handles typos, unconventional phrasing, etc.)
    city = _extract_city_with_llm(text)
    result = get_current_weather.invoke({"city": city})
    state["tool_result"] = result
    logger.info("[graph] Weather result: %s", result[:60])
    return state


# ─── Node: generate_reply ────────────────────────────────────────────────────

_REPLY_PROMPT = """You are a helpful e-commerce support agent.
Respond to the user's question based on the information below.
Be concise, friendly, and honest. If the information is insufficient, say so.

User question: {question}
Relevant information: {result}

Your reply:"""


def generate_reply(state: AgentState) -> AgentState:
    """Generate final answer (from cache or LLM)."""
    if state.get("final_answer"):
        logger.info("[graph] Using cached answer")
        return state

    question = state.get("user_input", "")
    result = state.get("tool_result", "No additional information available.")

    prompt = _REPLY_PROMPT.format(question=question, result=result)
    response = llm.invoke([HumanMessage(content=prompt)])
    state["final_answer"] = response.content.strip()
    logger.info("[graph] Generated answer: %s", state["final_answer"][:60])
    return state


# ─── Validation Prompt ─────────────────────────────────────────────────────────

_VALIDATION_PROMPT = """You are an accuracy auditor for an e-commerce support agent.

Your job is to check whether the agent's answer is fully grounded in the provided tool results.
Do NOT answer the user's question. Only assess accuracy.

User question: {question}
Tool results: {tool_result}
Agent answer: {answer}

Check for:
1. Fabricated data — order IDs, dates, amounts, or statuses not in the tool results
2. Contradictions — claims that conflict with the tool results
3. Unsupported claims — factual assertions with no basis in the tool results

Return ONLY one of these exact labels followed by an optional one-line note:

- valid — answer is fully grounded in the tool results
- unverified_claims — answer contains claims not supported by the tool results
- not_applicable — no tool results to validate against

Format: LABEL | note

Example: "valid | All order details match the tool output"
Example: "unverified_claims | Answer mentions order #1005 but tool result only shows #1003"
"""


def _parse_validation(raw: str) -> tuple[str, str]:
    """Parse LLM validation output into (flag, note). Defaults to unverified_claims on malformed input."""
    if not raw:
        return ("unverified_claims", "empty validation response")
    try:
        parts = raw.split("|", 1)
        flag = parts[0].strip()
        note = parts[1].strip() if len(parts) > 1 else ""
        if flag not in ("valid", "unverified_claims", "not_applicable"):
            return ("unverified_claims", f"unrecognized flag: {flag}")
        return (flag, note)
    except Exception:
        return ("unverified_claims", f"parse error: {raw[:100]}")


def validate_reply(state: AgentState) -> AgentState:
    """Validate that the generated answer is grounded in tool results."""
    # Skip if answer came from cache (already validated before caching)
    if state.get("cached"):
        state["validation_flag"] = "valid"
        state["validation_notes"] = "cache hit — previously validated"
        logger.info("[graph] Validation skipped: cache hit")
        return state

    question = state.get("user_input", "")
    tool_result = state.get("tool_result", "")
    answer = state.get("final_answer", "")

    # No tool result to validate against (unknown intent path)
    if not tool_result:
        state["validation_flag"] = "not_applicable"
        state["validation_notes"] = "no tool result to validate against"
        logger.info("[graph] Validation: not_applicable (no tool result)")
        return state

    try:
        prompt = _VALIDATION_PROMPT.format(
            question=question,
            tool_result=tool_result,
            answer=answer,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        flag, note = _parse_validation(raw)
        state["validation_flag"] = flag
        state["validation_notes"] = note
        logger.info("[graph] Validation: %s — %s", flag, note[:80])
    except Exception as e:
        state["validation_flag"] = "unverified_claims"
        state["validation_notes"] = f"validation call failed: {e}"
        logger.warning("[graph] Validation error: %s", e)

    return state


# ─── Node: update_memory ─────────────────────────────────────────────────────

def update_memory(state: AgentState) -> AgentState:
    """Persist conversation to memory store and semantic cache."""
    user_input = state.get("user_input", "")
    answer = state.get("final_answer", "")
    if not user_input or not answer:
        return state

    memory_store.add_user(user_input)
    memory_store.add_agent(answer)

    # Do NOT cache weather responses in semantic cache — real-time data should
    # always be fetched fresh to avoid stale or cross-city cache pollution.
    if _is_weather_query(user_input):
        logger.info("[graph] Memory updated (weather response not cached)")
    else:
        cache_response(user_input, answer)
        logger.info("[graph] Memory updated")

    return state
