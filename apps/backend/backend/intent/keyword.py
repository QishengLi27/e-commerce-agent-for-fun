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
import re
import string

from backend.agent import llm
from backend.intent.base import (
    LIST_ORDERS_PHRASES,
    POLICY_SIGNALS,
    WEATHER_SIGNALS,
    BaseIntentClassifier,
    extract_entities,
)
from backend.prompts import get_prompt as _get_prompt
from backend.resilience import make_retry_decorator

logger = logging.getLogger(__name__)


# ── Classifier-specific keyword sets ───────────────────────────────────────────

_ORDER_SIGNALS = {"order", "订单", "status of", "track"}
_KNOWLEDGE_SIGNALS = {
    "product",
    "category",
    "categories",
    "item info",
    "price",
    "tell me about",
    "what is",
    "which product",
}
_ORDER_ACTION_SIGNALS = {"cancel", "return", "refund", "money back", "exchange"}
_STRONG_POLICY_SIGNALS = {"return", "refund", "exchange", "warranty", "cancel", "money back"}
_PRODUCT_QA_SIGNALS = {
    # Feature / spec signals
    "does the",
    "do the",
    "can the",
    "how much",
    "how many",
    "feature",
    "spec",
    "specification",
    "battery",
    "camera",
    "weight",
    "size",
    "screen",
    "storage",
    "color",
    "magsafe",
    # Comparison / recommendation signals
    "compare",
    "vs",
    "versus",
    "better",
    "best",
    "cheapest",
    "under",
    "within budget",
    "recommend",
    "suggest",
    "suggestion",
    # Quality / suitability signals (e.g. "good laptop for work")
    "good",
    "great",
    "perfect",
    "ideal",
    "suitable",
    "top",
    "popular",
    "difference between",
}


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

    # Product QA signals + product or generic product term
    # Placed BEFORE policy signals so "does iPhone have MagSafe?" → product_qa, not policy
    # Use word-boundary matching for single-word signals to avoid false hits
    # such as "top" matching inside "laptop".
    qa_hits = []
    for w in _PRODUCT_QA_SIGNALS:
        if " " in w:
            if w in lowered:
                qa_hits.append(w)
        else:
            if re.search(r"\b" + re.escape(w) + r"\b", lowered):
                qa_hits.append(w)

    generic_terms = {
        "phone",
        "laptop",
        "tablet",
        "earbud",
        "headphone",
        "speaker",
        "sneaker",
        "shirt",
        "watch",
    }
    # Strip trailing punctuation so "laptops?" matches "laptop".
    cleaned_words = {w.strip(string.punctuation) for w in lowered.split()}
    has_generic_product = bool(generic_terms & cleaned_words)
    if qa_hits and (has_product or has_generic_product):
        return {
            "intent": "product_qa",
            "confidence": "high",
            "source": "entity+keyword",
            "entities": {"products": entities["products"], "categories": entities["categories"]},
        }

    # Product + policy signal → policy (before order check!)
    if has_product and policy_hits:
        return {
            "intent": "policy",
            "confidence": "high",
            "source": "entity+keyword",
            "context": {
                "products": entities["products"],
                "categories": entities["categories"],
                "order_ids": entities["order_ids"],
                "matched_signals": policy_hits,
            },
        }

    # Order ID + action signal → policy (user wants to act ON the order)
    if has_order and any(w in lowered for w in _ORDER_ACTION_SIGNALS):
        return {
            "intent": "policy",
            "confidence": "high",
            "source": "entity+keyword",
            "context": {
                "order_ids": entities["order_ids"],
                "matched_signals": [w for w in _ORDER_ACTION_SIGNALS if w in lowered],
            },
        }

    # Single order with ID (no action/product conflict)
    if has_order:
        return {
            "intent": "order",
            "confidence": "high",
            "source": "entity",
            "order_id": entities["order_ids"][0],
        }

    # Category + policy signal → policy
    if has_category and policy_hits:
        return {
            "intent": "policy",
            "confidence": "high",
            "source": "entity+keyword",
            "context": {
                "products": entities["products"],
                "categories": entities["categories"],
                "matched_signals": policy_hits,
            },
        }

    # Product, no conflicting signals → knowledge
    if has_product and not any(w in lowered for w in _ORDER_SIGNALS):
        return {
            "intent": "knowledge",
            "confidence": "high",
            "source": "entity",
            "context": {"products": entities["products"]},
        }

    # Category + knowledge signals → knowledge
    if has_category and any(w in lowered for w in _KNOWLEDGE_SIGNALS):
        return {
            "intent": "knowledge",
            "confidence": "high",
            "source": "entity+keyword",
            "context": {"categories": entities["categories"]},
        }

    # Strong policy signal without entity match
    strong_hits = [w for w in _STRONG_POLICY_SIGNALS if w in lowered]
    if len(policy_hits) >= 2 or (strong_hits and len(lowered.split()) <= 5):
        return {"intent": "policy", "confidence": "high", "source": "keyword"}

    # Ambiguous — needs LLM
    return None


# ── LLM Fallback ──────────────────────────────────────────────────────────────


@make_retry_decorator(max_attempts=2)
def _llm_classify_raw(query: str, entity_context: str) -> dict:
    output = _get_prompt("intent").render(query=query, entity_context=entity_context)
    prompt = output.text
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
