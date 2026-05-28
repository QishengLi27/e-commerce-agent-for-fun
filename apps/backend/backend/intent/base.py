"""
Shared interface and constants for intent classifiers.

All classifiers return a uniform dict:
    {
        "intent": str,        # order | list_orders | policy | weather | knowledge | unknown
        "confidence": str,    # high | medium | low
        "source": str,        # entity | entity+keyword | keyword | llm | llm+kg | semantic | fallback
        "order_id": str,      # for order intent
        "entities": dict,     # {"products": [...], "categories": [...], "order_ids": [...]}
        "context": dict,      # classifier-specific enrichment for downstream nodes
    }
"""

import re
from abc import ABC, abstractmethod
from typing import Literal

IntentType = Literal[
    "order", "list_orders", "policy", "weather", "knowledge", "unknown"
]

_ORDER_ID_RE = re.compile(r"\b(10\d{2,})\b")

# ── Shared keyword sets used across all classifiers as pre-filters ────────────

WEATHER_SIGNALS = {
    "weather", "temperature", "rain", "sunny", "cloudy", "forecast", "天气",
}

LIST_ORDERS_PHRASES = {
    "all orders", "订单列表", "show me orders", "list orders", "my orders",
}

POLICY_SIGNALS = {
    "return", "refund", "exchange", "shipping", "delivery", "warranty",
    "policy", "policies", "退货", "退款", "政策", "运费",
    "ship", "money back", "cancel", "package", "send back", "arrive",
}


# ── Entity Extraction (shared, no LLM) ────────────────────────────────────────

def extract_entities(query: str) -> dict:
    """Extract known entities from the query using the knowledge graph + regex."""
    from backend.knowledge.graph_store import get_knowledge_store

    kg = get_knowledge_store()
    lowered = query.lower()

    kg._load_names()
    products = []
    for name in sorted(kg._product_names or [], key=len, reverse=True):
        if name in lowered:
            products.append(name)

    categories = []
    for name in sorted(kg._category_names or [], key=len, reverse=True):
        if name in lowered:
            categories.append(name)

    order_ids = _ORDER_ID_RE.findall(lowered)

    return {"products": products, "categories": categories, "order_ids": order_ids}


# ── Classifier Interface ──────────────────────────────────────────────────────

class BaseIntentClassifier(ABC):
    """Interface for all intent classifiers."""

    @abstractmethod
    def classify(self, query: str) -> dict:
        """Classify a user query. Returns the standard result dict."""
        ...
