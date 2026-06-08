"""
Intent classification package with switchable modes.

Usage:
    from backend.intent import classify_intent_hybrid, get_classifier

    # Auto-picks classifier based on config CLASSIFICATION_MODE
    result = classify_intent_hybrid("Can I return my laptop?")

    # Or get a specific classifier
    classifier = get_classifier("llm_hybrid")
    result = classifier.classify("Where is order 1002?")

Modes:
    keyword    — <100 products, entity+keyword matching with LLM fallback
    llm_hybrid — 100-10K products, LLM extracts entities+intent, KG validates
    semantic   — 10K+ products, vector search top-K + LLM + KG validation
"""

import logging

from backend.config import settings

logger = logging.getLogger(__name__)


def _get_mode() -> str:
    """Read classification mode from config."""
    try:
        return settings.classification_mode
    except AttributeError:
        return "keyword"


def get_classifier(mode: str | None = None):
    """Factory: returns the classifier for the given mode.

    Args:
        mode: "keyword" | "llm_hybrid" | "semantic" (defaults to config)

    Returns:
        BaseIntentClassifier instance
    """
    mode = mode or _get_mode()

    if mode == "keyword":
        from backend.intent.keyword import KeywordIntentClassifier

        return KeywordIntentClassifier()

    elif mode == "llm_hybrid":
        from backend.intent.llm_hybrid import LlmHybridIntentClassifier

        return LlmHybridIntentClassifier()

    elif mode == "semantic":
        from backend.intent.semantic import SemanticIntentClassifier

        return SemanticIntentClassifier()

    else:
        raise ValueError(
            f"Unknown classification mode: {mode}. Expected one of: keyword, llm_hybrid, semantic"
        )


def classify_intent_hybrid(query: str) -> dict:
    """Classify a user query using the configured classifier mode.

    Returns dict with keys:
        intent, confidence, source, order_id (optional), entities (optional), context (optional)
    """
    classifier = get_classifier()
    result = classifier.classify(query)
    logger.info(
        "[intent] %s → %s (source=%s, confidence=%s, mode=%s)",
        query[:60],
        result["intent"],
        result["source"],
        result.get("confidence", "N/A"),
        _get_mode(),
    )
    return result


def clear_cache():
    """Clear all classifier caches."""
    from backend.intent.keyword import clear_cache as clear_keyword
    from backend.intent.llm_hybrid import clear_cache as clear_llm
    from backend.intent.semantic import clear_cache as clear_semantic

    clear_keyword()
    clear_llm()
    clear_semantic()
