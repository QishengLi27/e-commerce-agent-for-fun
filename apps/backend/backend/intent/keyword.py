"""
Keyword-based intent classifier with entity-aware matching + LLM fallback.

Best for: small catalogs (<100 products) where substring matching is reliable.
Architecture:
    1. extract_entities(query)    — KG lookup + regex (from base.py)
    2. entity+keyword rules       — deterministic co-signal detection
    3. LLM fallback               — for ambiguous queries, with entity context
"""

import json
import logging

from backend.agent import llm
from backend.intent.base import (
    LIST_ORDERS_PHRASES,
    POLICY_SIGNALS,
    WEATHER_SIGNALS,
    BaseIntentClassifier,
    extract_entities,
)
from backend.resilience import make_retry_decorator

logger = logging.getLogger(__name__)


# ── Classifier-specific keyword sets ───────────────────────────────────────────

_ORDER_SIGNALS = {"order", "订单", "status of", "track"}
_KNOWLEDGE_SIGNALS = {
    "product", "category", "categories", "item info", "price", "tell me about",
    "what is", "which product",
}
_ORDER_ACTION_SIGNALS = {"cancel", "return", "refund", "money back", "exchange"}
_STRONG_POLICY_SIGNALS = {"return", "refund", "exchange", "warranty", "cancel", "money back"}


# ── Keyword Rule Engine ───────────────────────────────────────────────────────

def _classify_with_entities(query: str, entities: dict) -> dict | None:
    """Classify using entity + keyword co-signals.

    Returns a result dict if confident, None if ambiguous (needs LLM fallback).
    """
    lowered = query.lower()
    has_product = bool(entities["products"])
    has_category = bool(entities["categories"])
    has_order = bool(entities["order_ids"])
    policy_hits = [w for w in POLICY_SIGNALS if w in lowered]
    weather_hits = [w for w in WEATHER_SIGNALS if w in lowered]

    # Weather (most specific)
    if weather_hits:
        return {"intent": "weather", "confidence": "high", "source": "keyword"}

    # List all orders
    if any(phrase in lowered for phrase in LIST_ORDERS_PHRASES):
        return {"intent": "list_orders", "confidence": "high", "source": "keyword"}

    # Product + policy signal → policy (before order check!)
    if has_product and policy_hits:
        return {
            "intent": "policy", "confidence": "high", "source": "entity+keyword",
            "context": {"products": entities["products"], "categories": entities["categories"],
                        "order_ids": entities["order_ids"], "matched_signals": policy_hits},
        }

    # Order ID + action signal → policy (user wants to act ON the order)
    if has_order and any(w in lowered for w in _ORDER_ACTION_SIGNALS):
        return {
            "intent": "policy", "confidence": "high", "source": "entity+keyword",
            "context": {"order_ids": entities["order_ids"],
                        "matched_signals": [w for w in _ORDER_ACTION_SIGNALS if w in lowered]},
        }

    # Single order with ID (no action/product conflict)
    if has_order:
        return {"intent": "order", "confidence": "high", "source": "entity",
                "order_id": entities["order_ids"][0]}

    # Category + policy signal → policy
    if has_category and policy_hits:
        return {
            "intent": "policy", "confidence": "high", "source": "entity+keyword",
            "context": {"products": entities["products"], "categories": entities["categories"],
                        "matched_signals": policy_hits},
        }

    # Product, no conflicting signals → knowledge
    if has_product and not any(w in lowered for w in _ORDER_SIGNALS):
        return {"intent": "knowledge", "confidence": "high", "source": "entity",
                "context": {"products": entities["products"]}}

    # Category + knowledge signals → knowledge
    if has_category and any(w in lowered for w in _KNOWLEDGE_SIGNALS):
        return {"intent": "knowledge", "confidence": "high", "source": "entity+keyword",
                "context": {"categories": entities["categories"]}}

    # Strong policy signal without entity match
    strong_hits = [w for w in _STRONG_POLICY_SIGNALS if w in lowered]
    if len(policy_hits) >= 2 or (strong_hits and len(lowered.split()) <= 5):
        return {"intent": "policy", "confidence": "high", "source": "keyword"}

    # Ambiguous — needs LLM
    return None


# ── LLM Fallback ──────────────────────────────────────────────────────────────

_INTENT_PROMPT = """You are an intent classifier for an e-commerce support chatbot.
Classify the user's query into exactly one of these categories:

- order: user asks about a specific order by ID (e.g., "where is order 1001?")
- list_orders: user wants to see all their orders (e.g., "show my orders")
- policy: user asks about store policies — returns, refunds, shipping, warranty
- weather: user asks about weather in a city
- knowledge: user asks about a specific product or category
- unknown: greeting, small talk, or anything else

{entity_context}
Respond with ONLY a JSON object (no markdown, no explanation):
{{"intent": "<category>", "confidence": "high|medium|low", "reason": "one-line reason"}}

User: {query}
"""


@make_retry_decorator(max_attempts=2)
def _llm_classify_raw(query: str, entity_context: str) -> dict:
    prompt = _INTENT_PROMPT.format(query=query, entity_context=entity_context)
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict, got {type(parsed)}")
    return {
        "intent": parsed.get("intent", "unknown"),
        "confidence": parsed.get("confidence", "low"),
        "reason": parsed.get("reason", ""),
        "source": "llm",
    }


_llm_cache: dict[str, dict] = {}
_LLM_CACHE_MAX = 256


def _llm_classify_cached(query: str, entities: dict) -> dict:
    lowered = query.lower().strip()
    if lowered in _llm_cache:
        return _llm_cache[lowered]

    parts = []
    if entities["products"]:
        parts.append(f"The query mentions these KNOWN products: {entities['products']}")
    if entities["categories"]:
        parts.append(f"The query mentions these KNOWN categories: {entities['categories']}")
    if entities["order_ids"]:
        parts.append(f"The query contains these order IDs: {entities['order_ids']}")
    entity_context = ("KNOWN ENTITIES IN QUERY:\n" + "\n".join(parts) + "\n") if parts else ""

    try:
        result = _llm_classify_raw(query, entity_context)
        if result.get("confidence") in ("high", "medium"):
            if len(_llm_cache) >= _LLM_CACHE_MAX:
                _llm_cache.pop(next(iter(_llm_cache)))
            _llm_cache[lowered] = result
        return result
    except Exception as e:
        logger.warning("[intent:keyword] LLM failed: %s", e)
        return {"intent": "unknown", "confidence": "low", "source": "fallback"}


# ── Classifier ────────────────────────────────────────────────────────────────

class KeywordIntentClassifier(BaseIntentClassifier):
    """Entity-aware keyword classifier. Best for catalogs under ~100 products.

    Pipeline: extract entities → keyword rules → LLM fallback"""

    def classify(self, query: str) -> dict:
        entities = extract_entities(query)
        result = _classify_with_entities(query, entities)
        if result:
            if any(entities.values()):
                result["entities"] = {k: v for k, v in entities.items() if v}
            return result

        llm_result = _llm_classify_cached(query, entities)
        if any(entities.values()):
            llm_result["entities"] = {k: v for k, v in entities.items() if v}
        return llm_result


def clear_cache():
    _llm_cache.clear()
