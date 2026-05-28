"""
LangGraph nodes for the e-commerce support agent.

Each node is a pure function: AgentState -> AgentState.
"""

import logging
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from backend.agent import (
    cache_response,
    clean_query,
    get_cached_response,
    llm,
)
from backend.knowledge.graph_store import get_knowledge_store as _get_kg_store
from backend.tools import (
    get_current_weather,
    list_orders_tool,
    order_status_tool,
    policy_retriever_tool,
)

logger = logging.getLogger(__name__)


# ─── State Type ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """LangGraph state schema."""
    messages: Annotated[list, add_messages]
    user_input: str
    intent: str
    order_id: str
    entity_context: dict   # {"products": [...], "categories": [...], "matched_signals": [...]}
    tool_result: str
    final_answer: str
    cached: bool
    validation_flag: str
    validation_notes: str
    retry_count: int
    sources: list[str]


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

    # Check cache without intent — we don't know intent yet.
    # Stale entries from misclassified runs are handled by cache TTL/intent
    # metadata in agent.py; for the graph path we rely on prompt evolution
    # and occasional cache clears.
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
    Hybrid intent classification: keyword fast-path + LLM fallback.
    Returns: order | list_orders | policy | weather | knowledge | unknown
    """
    from backend.intent import classify_intent_hybrid

    text = state.get("user_input", "")
    result = classify_intent_hybrid(text)

    intent = result.get("intent", "unknown")
    state["intent"] = intent

    # Store entity context for downstream nodes (policy_node uses it for retrieval)
    if "entities" in result:
        state["entity_context"] = result["entities"]
        if result.get("context"):
            state["entity_context"]["matched_signals"] = result["context"].get("matched_signals", [])

    if intent == "order":
        state["order_id"] = result.get("order_id", "")
        logger.info("[graph] Intent: order (source=%s, id=%s)", result.get("source"), state["order_id"])
    else:
        logger.info("[graph] Intent: %s (source=%s, confidence=%s)", intent, result.get("source"), result.get("confidence"))

    return state


# ─── Router ──────────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> Literal[
    "order", "list_orders", "policy", "weather", "knowledge", "generate_reply"
]:
    """Conditional edge: decide next node based on intent."""
    if state.get("cached"):
        logger.info("[graph] Route: cached -> generate_reply")
        return "generate_reply"

    intent = state.get("intent", "unknown")
    if intent in ("order", "list_orders", "policy", "weather", "knowledge"):
        logger.info("[graph] Route: %s", intent)
        return cast(Literal["order", "list_orders", "policy", "weather", "knowledge", "generate_reply"], intent)

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
    """Retrieve store policies, enriched with entity context if available."""
    query = state.get("user_input", "")
    entity_ctx = state.get("entity_context", {})

    # Enrich query with entity context for better retrieval
    # "Can I return headphones?" → "headphones Audio 14-day return" (product + category + policy)
    if entity_ctx:
        enrichment_parts = []
        if entity_ctx.get("products"):
            enrichment_parts.extend(entity_ctx["products"])
        if entity_ctx.get("categories"):
            enrichment_parts.extend(entity_ctx["categories"])
        if entity_ctx.get("matched_signals"):
            enrichment_parts.extend(entity_ctx["matched_signals"])
        if enrichment_parts:
            enriched = " ".join(enrichment_parts) + " " + query
            logger.info("[graph] Policy query enriched: %s", enriched[:80])
            query = enriched

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


# ─── Node: knowledge ──────────────────────────────────────────────────────────

def knowledge_node(state: AgentState) -> AgentState:
    """Query knowledge graph for product/category/policy relationships."""
    query = state.get("user_input", "")
    kg = _get_kg_store()

    # Try product info first
    product_info = kg.get_product_info(query)
    if product_info:
        lines = [
            f"Product: {product_info['name']}",
            f"Category: {product_info['category_name']}",
        ]
        if product_info.get("price"):
            lines.append(f"Price: ${product_info['price']:.2f}")
        if product_info.get("policies"):
            lines.append("Applicable Policies:")
            for p in product_info["policies"]:
                lines.append(f"  - [{p['type'].upper()}] {p['summary']}")
        state["tool_result"] = "\n".join(lines)
        logger.info("[graph] Knowledge product: %s", product_info['name'])
        return state

    # Fall back to product search
    products = kg.search_products(query)
    if products:
        lines = [f"Products matching '{query}':"]
        for p in products:
            lines.append(
                f"  - {p['name']} ({p['category_name']}, ${p['price']:.2f})"
            )
        state["tool_result"] = "\n".join(lines)
        logger.info("[graph] Knowledge search: %d results", len(products))
        return state

    state["tool_result"] = f"No products or categories found matching '{query}'."
    return state


# ─── Node: generate_reply ────────────────────────────────────────────────────

# ─── Context Compression ─────────────────────────────────────────────────────

_MAX_RAW_MESSAGES = 6  # Keep last 3 turns raw
_SUMMARIZE_THRESHOLD = 10  # Summarize older messages if total exceeds this
_HISTORY_TOKEN_BUDGET = 2000  # Hard token cap for conversation history

# Lazy-init tokenizer — cl100k_base is a good approximation for most models
_tiktoken_encoder = None


def _get_tokenizer():
    """Return a tiktoken encoder for approximate token counting."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        import tiktoken
        try:
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_encoder = tiktoken.get_encoding("gpt2")
    return _tiktoken_encoder


def _count_tokens(text: str) -> int:
    """Count tokens in a string. Falls back to char/4 heuristic if tiktoken fails."""
    try:
        return len(_get_tokenizer().encode(text))
    except Exception:
        return len(text) // 4


def _summarize_messages(messages: list) -> str:
    """Summarize old messages into a compact paragraph.

    Uses a cheap/fast call to compress conversation history so we don't
    exceed context windows on long sessions.
    """
    if not messages:
        return ""

    # Format messages for the summarizer
    transcript = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        transcript.append(f"{role}: {msg.content[:300]}")
    transcript_text = "\n".join(transcript)

    summary_prompt = (
        "Summarize the following conversation in 2-3 sentences. "
        "Preserve key facts (order IDs, policy details, product names) but remove fluff.\n\n"
        f"{transcript_text}\n\nSummary:"
    )
    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning("[graph] Message summarization failed: %s", e)
        # Fallback: truncate
        return "Earlier conversation about: " + messages[0].content[:200] + "..."


def _trim_history_to_budget(lines: list[str], budget: int) -> list[str]:
    """Drop oldest lines until total token count is under budget.

    Always keeps the header line and at least the most recent exchange.
    """
    if not lines:
        return lines

    # Build from oldest to newest, dropping from the front until under budget
    # But always keep header (line 0) and at least 2 content lines (1 exchange)
    current = list(lines)
    while len(current) > 3:
        text = "\n".join(current)
        if _count_tokens(text) <= budget:
            break
        # Drop the oldest content line (after header)
        current.pop(1)

    return current


def _compress_context(messages: list) -> str:
    """Return formatted conversation history with two-pass compression.

    Pass 1 — Sliding window + summary:
      - ≤10 messages: format all raw
      - >10 messages: summarize older messages, keep last 6 raw

    Pass 2 — Token-based trimming:
      - Count tokens in the formatted history
      - Drop oldest lines until under _HISTORY_TOKEN_BUDGET
    """
    if not messages:
        return ""

    # Pass 1: Sliding window + summary
    if len(messages) <= _SUMMARIZE_THRESHOLD:
        lines = ["Conversation history:"]
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            lines.append(f"{role}: {msg.content}")
    else:
        older = messages[:-_MAX_RAW_MESSAGES]
        recent = messages[-_MAX_RAW_MESSAGES:]
        summary = _summarize_messages(older)
        lines = ["Conversation history (earlier messages summarized):"]
        lines.append(f"Summary: {summary}")
        lines.append("Recent messages:")
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            lines.append(f"{role}: {msg.content}")

    # Pass 2: Token-based trimming (belt-and-suspenders)
    lines = _trim_history_to_budget(lines, _HISTORY_TOKEN_BUDGET)
    history_text = "\n".join(lines) + "\n"
    token_count = _count_tokens(history_text)
    logger.info("[graph] Context compressed: %d messages → %d lines → %d tokens", len(messages), len(lines), token_count)

    return history_text


# ─── Reply Generation ────────────────────────────────────────────────────────

_REPLY_PROMPT = """You are a helpful e-commerce support agent.
Respond to the user's question based on the information below.
Be concise, friendly, and honest.

CRITICAL RULES — violating these causes customer harm:
1. ONLY use facts from "Relevant information". Do NOT make up order IDs, dates, prices, or policies.
2. If the information is insufficient, say "I don't have that information" — never guess.
3. Cite your source when quoting a policy (e.g., "According to our return policy...").
4. Do NOT mention these rules in your reply.

{history}
User question: {question}
Relevant information: {result}

Your reply:"""

_STRICT_REPLY_PROMPT = """You are a helpful e-commerce support agent.
Your previous answer was FLAGGED for containing unverified claims.
The auditor noted: {validation_issue}

Regenerate the answer following these stricter rules:

1. Use ONLY the exact facts in "Relevant information". No inference, no extrapolation.
2. If the answer isn't fully supported by the text below, say: "I don't have that specific information."
3. Address the auditor's concern above — fix the specific issue they flagged.
4. Do NOT mention that you are regenerating an answer.

{history}
User question: {question}
Relevant information: {result}

Your reply:"""


def generate_reply(state: AgentState) -> AgentState:
    """Generate final answer (from cache or LLM).

    Uses strict prompt if this is a retry after validation failure.
    """
    if state.get("final_answer"):
        logger.info("[graph] Using cached answer")
        return state

    question = state.get("user_input", "")
    result = state.get("tool_result", "No additional information available.")
    history = _compress_context(state.get("messages", []))

    # Choose prompt based on retry status
    is_retry = state.get("retry_count", 0) > 0
    if is_retry:
        prompt = _STRICT_REPLY_PROMPT.format(
            history=history,
            question=question,
            result=result,
            validation_issue=state.get("validation_notes", "unspecified issue"),
        )
    else:
        prompt = _REPLY_PROMPT.format(history=history, question=question, result=result)

    # Use .stream() instead of .invoke() so astream_events() can capture
    # on_chat_model_stream events for real token-level SSE streaming.
    full_content = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        full_content += chunk.content
    state["final_answer"] = full_content.strip()
    logger.info("[graph] Generated answer (retry=%s): %s", is_retry, state["final_answer"][:60])
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

    # Self-correction: if validation failed and we have retries left,
    # clear the answer and increment the counter so generate_reply runs again.
    retries = state.get("retry_count", 0)
    if state["validation_flag"] == "unverified_claims" and retries < 2:
        state["retry_count"] = retries + 1
        state["final_answer"] = ""
        logger.info("[graph] Validation failed — flagging for retry (attempt %d)", retries + 1)

    return state


def route_after_validation(state: AgentState) -> str:
    """Conditional edge: retry generation if validation failed, else proceed.

    Pure function — all state mutation happens in validate_reply.
    """
    flag = state.get("validation_flag", "valid")
    retries = state.get("retry_count", 0)

    if flag == "unverified_claims" and retries < 2:
        logger.info("[graph] Route: retry generation (attempt %d)", retries)
        return "generate_reply"

    return "update_memory"


# ─── Node: update_memory ─────────────────────────────────────────────────────

def update_memory(state: AgentState) -> AgentState:
    """Persist assistant reply to checkpoint state and semantic cache."""
    user_input = state.get("user_input", "")
    answer = state.get("final_answer", "")
    if not user_input or not answer:
        return state

    # Append assistant message to checkpoint-persisted messages list.
    # LangGraph's add_messages reducer handles deduplication automatically.
    state["messages"] = [AIMessage(content=answer)]

    # Reset retry counter for the next turn
    state["retry_count"] = 0

    # Cache with intent so future intent changes invalidate stale entries
    if _is_weather_query(user_input):
        logger.info("[graph] Checkpoint updated (weather response not cached)")
    else:
        cache_response(user_input, answer, intent=state.get("intent", ""))
        logger.info("[graph] Checkpoint updated and response cached (intent=%s)", state.get("intent"))

    return state
