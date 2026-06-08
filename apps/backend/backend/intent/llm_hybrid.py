"""
LLM-hybrid intent classifier for medium catalogs (100–10K products).

Architecture:
    1. Fast pre-filter: order IDs, weather, list_orders (regex/keyword, zero LLM cost)
    2. LLM extracts entities + intent in a single structured call
    3. Knowledge Graph validates: do the LLM's entities actually exist?
       - Exact match → confirmed, attach category + policy context
       - Partial match → fuzzy search, return candidates
       - No match → LLM hallucinated, flag as unverified

The KG is the ground-truth anchor that prevents the LLM from hallucinating products.
"""

import json
import logging
from typing import Any

from backend.agent import llm
from backend.intent.base import (
    _ORDER_ID_RE,
    LIST_ORDERS_PHRASES,
    POLICY_SIGNALS,
    WEATHER_SIGNALS,
    BaseIntentClassifier,
)
from backend.resilience import make_retry_decorator

logger = logging.getLogger(__name__)


# ── LLM Prompt for Entity + Intent Extraction ─────────────────────────────────

_LLM_EXTRACT_PROMPT = """You are an intent and entity extractor for an e-commerce support chatbot.

Step 1 — Classify the user's intent into exactly one category:
- order: tracking or checking a specific order by ID
- list_orders: seeing all orders
- policy: returns, refunds, shipping, warranty, cancellation
- weather: weather in a city
- knowledge: product information, categories, pricing
- product_qa: user asks about a product's features, specs, comparisons, or category
- unknown: greeting, small talk, or anything else

Step 2 — Extract mentioned entities:
- Products: any product names the user mentions
- Categories: any product categories the user mentions
- Order IDs: any order ID numbers (format: 10XX)

Respond with ONLY a JSON object (no markdown, no explanation):
{{"intent": "<category>", "confidence": "high|medium|low",
  "entities": {{"products": [...], "categories": [...], "order_ids": [...]}},
  "reason": "one-line reason"}}

User: {query}
"""


# ── KG Validation ─────────────────────────────────────────────────────────────


def _validate_entities(llm_entities: dict) -> dict:
    """Validate LLM-extracted entities against the knowledge graph.

    Returns:
        "confirmed": {products: [...], categories: [...]} — exact matches
        "candidates": {products: [{name, score}, ...]}  — fuzzy matches
        "rejected": [...] — no match found (likely hallucination)
    """
    from backend.knowledge.graph_store import get_knowledge_store

    kg = get_knowledge_store()
    kg._load_names()
    all_products = kg._product_names or []
    all_categories = kg._category_names or []

    confirmed: dict[str, list[str]] = {"products": [], "categories": []}
    candidates: dict[str, list[dict[str, Any]]] = {"products": []}
    rejected = []

    for product in llm_entities.get("products", []):
        lowered = product.lower()
        if lowered in all_products:
            confirmed["products"].append(lowered)
        else:
            # Fuzzy: find closest match
            best: str | None = None
            best_score = 0.0
            for known in all_products:
                if lowered in known or known in lowered:
                    best, best_score = known, 0.9
                    break
                overlap = len(set(lowered.split()) & set(known.split())) / max(
                    len(lowered.split()), len(known.split()), 1
                )
                if overlap > best_score:
                    best, best_score = known, overlap
            if best and best_score > 0.5:
                candidates["products"].append({"name": best, "score": round(best_score, 2)})
            else:
                rejected.append(product)

    for category in llm_entities.get("categories", []):
        lowered = category.lower()
        if lowered in all_categories:
            confirmed["categories"].append(lowered)
        else:
            candidates["products"] = candidates.get("products", [])  # keep as-is
            rejected.append(category)

    return {"confirmed": confirmed, "candidates": candidates, "rejected": rejected}


# ── LLM Classifier ────────────────────────────────────────────────────────────


@make_retry_decorator(max_attempts=2)
def _llm_extract_raw(query: str) -> dict:
    prompt = _LLM_EXTRACT_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


_llm_cache: dict[str, dict] = {}
_LLM_CACHE_MAX = 512


def _llm_extract_cached(query: str) -> dict:
    lowered = query.lower().strip()
    if lowered in _llm_cache:
        return _llm_cache[lowered]
    try:
        result = _llm_extract_raw(query)
        if len(_llm_cache) >= _LLM_CACHE_MAX:
            _llm_cache.pop(next(iter(_llm_cache)))
        _llm_cache[lowered] = result
        return result
    except Exception as e:
        logger.warning("[intent:llm_hybrid] LLM failed: %s", e)
        return {
            "intent": "unknown",
            "confidence": "low",
            "entities": {"products": [], "categories": [], "order_ids": []},
        }


# ── Classifier ────────────────────────────────────────────────────────────────


class LlmHybridIntentClassifier(BaseIntentClassifier):
    """LLM extracts entities + intent, KG validates.

    Best for catalogs of 100–10,000 products where substring matching breaks down.
    The KG acts as the ground-truth anchor: if the LLM hallucinates a product,
    the validation layer catches it and returns candidates instead."""

    def classify(self, query: str) -> dict:
        lowered = query.lower()

        # Fast pre-filters (deterministic, zero LLM cost)
        if any(w in lowered for w in WEATHER_SIGNALS):
            return {"intent": "weather", "confidence": "high", "source": "keyword"}

        if any(phrase in lowered for phrase in LIST_ORDERS_PHRASES):
            return {"intent": "list_orders", "confidence": "high", "source": "keyword"}

        # These order ID + action checks still work at scale
        order_ids = _ORDER_ID_RE.findall(lowered)
        policy_hits = [w for w in POLICY_SIGNALS if w in lowered]
        if order_ids and not policy_hits:
            return {
                "intent": "order",
                "confidence": "high",
                "source": "entity",
                "order_id": order_ids[0],
            }

        # Main path: LLM extracts entities + intent
        llm_result = _llm_extract_cached(query)

        # Validate LLM entities against KG
        validation = _validate_entities(llm_result.get("entities", {}))

        result = {
            "intent": llm_result.get("intent", "unknown"),
            "confidence": llm_result.get("confidence", "low"),
            "source": "llm+kg",
            "entities": validation["confirmed"],
        }

        if validation["candidates"].get("products"):
            result["entity_candidates"] = validation["candidates"]["products"]
            # If nothing confirmed but we have candidates, lower confidence
            if (
                not validation["confirmed"]["products"]
                and not validation["confirmed"]["categories"]
            ):
                result["confidence"] = "low"

        if validation["rejected"]:
            result["rejected_entities"] = validation["rejected"]

        # Enrich with KG context for downstream use
        if validation["confirmed"]["products"]:
            result["context"] = {
                "products": validation["confirmed"]["products"],
                "categories": validation["confirmed"]["categories"],
            }

        return result


def clear_cache():
    _llm_cache.clear()
