"""
Product QA tool — orchestrates Neo4j graph traversal + LlamaIndex RAG.

Handles three query patterns:
  1. Single-product: "Does iPhone 15 Pro have MagSafe?"
     → Neo4j resolve → LlamaIndex RAG (filtered to product) → synthesize
  2. Cross-product: "Which phone under $800 has the best camera?"
     → Neo4j candidate search → LlamaIndex RAG (per candidate) → compare
  3. Category/definition: "What category is iPhone 15 Pro in?"
     → Neo4j category path → synthesize with attributes

The tool is the single integration point between Neo4j, LlamaIndex, and the LLM.
"""

import logging

from langchain.tools import tool

from backend.knowledge.models import ProductRef
from backend.knowledge.neo4j_store import get_neo4j_store
from backend.rag.query_engine import retrieve_product_chunks

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORY_PATH_SIGNALS = {
    "category",
    "categories",
    "what kind",
    "what type",
    "classified",
    "belong to",
    "product line",
}

CROSS_PRODUCT_SIGNALS = {
    "which",
    "compare",
    "vs",
    "versus",
    "better",
    "best",
    "cheapest",
    "under",
    "within budget",
    "difference between",
}

# Pure recommendation requests (always route to recommendation handler)
_RECOMMENDATION_SIGNALS = {"recommend", "suggestion", "suggest"}

# Quality / suitability signals (e.g. "good laptop for work")
_QUALITY_SIGNALS = {"good", "great", "perfect", "ideal", "suitable", "top", "popular"}

# Feature words that indicate a spec/detail question even when quality words are present
# (e.g. "Does this laptop have a good battery?" → spec, not recommendation)
_FEATURE_WORDS = {
    "battery",
    "camera",
    "screen",
    "display",
    "storage",
    "memory",
    "ram",
    "processor",
    "cpu",
    "gpu",
    "graphics",
    "weight",
    "size",
    "dimension",
    "color",
    "port",
    "ports",
    "connectivity",
    "wireless",
    "bluetooth",
    "wifi",
    "magsafe",
    "waterproof",
    "durability",
    "warranty",
}


@tool
def product_qa_tool(query: str) -> str:
    """Answer product questions using the knowledge graph and product descriptions.

    Use this tool when the user asks about:
    - Product specifications, features, or capabilities
      ("Does iPhone 15 Pro have MagSafe?", "How much does MacBook weigh?")
    - Product comparisons
      ("Which phone has the best camera under $800?", "Compare MacBook vs Dell XPS")
    - Category questions
      ("What category is iPhone in?", "Why is this product in Flagship Phones?")
    - Recommendations
      ("What's a good Android phone?", "What accessories for iPhone 15 Pro?")

    Args:
        query: The user's question about a product.

    Returns:
        Formatted answer synthesized from graph data and product descriptions.
    """
    try:
        store = get_neo4j_store()
        lowered = query.lower()

        # ── Pattern 1: Category/path queries ──────────────────────────────
        if any(signal in lowered for signal in CATEGORY_PATH_SIGNALS):
            return _answer_category_query(query, store)

        # ── Pattern 2: Recommendation queries ─────────────────────────────
        # Pure recommendation signals always route here. Quality signals (good/great/etc)
        # route here only when the user is not asking about a specific feature.
        has_recommendation = any(signal in lowered for signal in _RECOMMENDATION_SIGNALS)
        has_quality = any(signal in lowered for signal in _QUALITY_SIGNALS)
        has_feature = any(word in lowered for word in _FEATURE_WORDS)
        if has_recommendation or (has_quality and not has_feature):
            return _answer_recommendation_query(query, store)

        # ── Pattern 3: Cross-product comparison ───────────────────────────
        if any(signal in lowered for signal in CROSS_PRODUCT_SIGNALS):
            return _answer_comparison_query(query, store)

        # ── Default: Single-product detail query ──────────────────────────
        return _answer_single_product_query(query, store)

    except Exception:
        logger.exception("[product_qa_tool] Unhandled error for query: %s", query)
        return (
            "I'm having trouble accessing product information right now. "
            "Please try again in a moment or rephrase your question."
        )


# ── Query Handlers ────────────────────────────────────────────────────────────


def _answer_category_query(query: str, store) -> str:
    """Answer 'what category is X in?' with reasoning from attributes."""
    # Extract product name: resolve the query against known products
    product = _resolve_product_from_query(query, store)
    if not product:
        return (
            f"I couldn't identify a specific product in your question: '{query}'. "
            "Could you specify which product you're asking about?"
        )

    info = store.get_product_info(product.name)
    if not info or not info.category_path:
        return f"I found {product.name}, but couldn't determine its category path."

    # Format category path with explanations derived from attributes
    path_str = " → ".join(info.category_path)
    attr_lines = ""
    if info.attributes:
        attr_lines = "\n\nProduct attributes:\n"
        for attr in info.attributes:
            if attr.name in ("brand", "price", "storage", "screen_size", "color"):
                attr_lines += f"  - {attr.display_name}: {attr.value}"
                if attr.unit:
                    attr_lines += f" {attr.unit}"
                attr_lines += "\n"

    prompt = (
        f"The product '{product.name}' belongs to this category path: {path_str}."
        f"{attr_lines}\n\n"
        f"User question: {query}\n\n"
        f"Explain which category this product belongs to and why, "
        f"based on its position in the category hierarchy and its attributes. "
        f"Be concise and helpful."
    )
    from backend.agent import llm  # lazy import to avoid circular dependency

    response = llm.invoke(prompt)
    return response.content.strip()


def _answer_single_product_query(query: str, store) -> str:
    """Answer a question about a specific product using RAG over its description."""
    product = _resolve_product_from_query(query, store)
    if not product:
        return _fallback_search(query, store)

    # Get graph context (attributes, category, policies)
    info = store.get_product_info(product.name)

    # Get RAG context from LlamaIndex (filtered to this product)
    rag_context = _retrieve_product_chunks(query, product_names=[product.name])

    # Synthesize
    graph_context = ""
    if info:
        graph_context = f"Product: {info.name}\n"
        if info.category_path:
            graph_context += f"Category: {' → '.join(info.category_path)}\n"
        if info.attributes:
            graph_context += "Attributes:\n"
            for attr in info.attributes:
                graph_context += f"  - {attr.display_name}: {attr.value}"
                if attr.unit:
                    graph_context += f" {attr.unit}"
                graph_context += "\n"

    prompt = (
        f"You are a product expert. Answer the user's question about "
        f"'{product.name}' using the information below.\n\n"
        f"Structured product data:\n{graph_context}\n"
        f"Product description excerpts:\n{rag_context}\n\n"
        f"User question: {query}\n\n"
        f"Answer the question accurately. If the information is insufficient, "
        f"say so — don't guess. If relevant, mention where the product sits "
        f"in the category hierarchy."
    )
    from backend.agent import llm  # lazy import to avoid circular dependency

    response = llm.invoke(prompt)
    return response.content.strip()


def _answer_comparison_query(query: str, store, candidates: list | None = None) -> str:
    """Handle cross-product comparison queries.

    Strategy:
      1. Extract constraints from the query (category, brand, max_price)
      2. Neo4j search → candidate products
      3. LlamaIndex RAG on each candidate for the relevant feature
      4. LLM compares and recommends
    """
    # Use provided candidates or search for them
    if candidates is None:
        # Extract category hint from query
        category_hint = _extract_category_hint(query)
        candidates = store.search_products(
            category=category_hint,
            limit=5,
        )

    if not candidates:
        return f"I couldn't find any products matching your criteria in '{query}'."

    if len(candidates) == 1:
        return _answer_single_product_query(f"Tell me about {candidates[0].name}", store)

    # Get RAG context for each candidate
    context_parts = []
    for candidate in candidates:
        price_str = f"${candidate.price:.2f}" if candidate.price else "N/A"
        context_parts.append(
            f"\n--- {candidate.name} ({price_str}) ---\n"
            f"Category: {candidate.category_name or 'N/A'}\n"
        )
        # Get relevant description chunks
        rag_chunks = _retrieve_product_chunks(query, product_names=[candidate.name])
        context_parts.append(rag_chunks or "No detailed description available.")

    combined_context = "\n".join(context_parts)

    prompt = (
        f"You are a product comparison expert. Compare these products "
        f"based on the user's question and recommend the best option.\n\n"
        f"Products:\n{combined_context}\n\n"
        f"User question: {query}\n\n"
        f"Compare the relevant features, explain trade-offs, and give a clear "
        f"recommendation with reasoning. Be fair and honest about each product's "
        f"strengths and weaknesses."
    )
    from backend.agent import llm  # lazy import to avoid circular dependency

    response = llm.invoke(prompt)
    return response.content.strip()


def _answer_recommendation_query(query: str, store) -> str:
    """Handle 'recommend me a phone' type queries.

    Same flow as comparison but with recommendation framing.
    """
    category_hint = _extract_category_hint(query)
    candidates = store.search_products(
        category=category_hint,
        limit=5,
    )
    if not candidates:
        return f"I couldn't find any products matching your criteria in '{query}'."

    # Get RAG context for each candidate
    context_parts = []
    for candidate in candidates:
        price_str = f"${candidate.price:.2f}" if candidate.price else "N/A"
        context_parts.append(
            f"\n--- {candidate.name} ({price_str}) ---\n"
            f"Category: {candidate.category_name or 'N/A'}\n"
        )
        rag_chunks = _retrieve_product_chunks(query, product_names=[candidate.name])
        context_parts.append(rag_chunks or "No detailed description available.")

    combined_context = "\n".join(context_parts)

    prompt = (
        f"You are a product recommendation expert. Based on the user's request, "
        f"recommend the best product from the options below.\n\n"
        f"Products:\n{combined_context}\n\n"
        f"User request: {query}\n\n"
        f"Recommend the most suitable product with clear reasoning. Be fair and "
        f"honest about each option's strengths and weaknesses."
    )
    from backend.agent import llm  # lazy import to avoid circular dependency

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_product_from_query(query: str, store) -> ProductRef | None:
    """Try to identify which product the user is asking about.

    Strategy: try each word/phrase as a product name lookup.
    Uses the Neo4j fulltext index which includes search_terms (synonyms).
    """
    # Try the full query first (for queries that are just a product name)
    result = store.resolve_product(query)
    if result:
        return result

    # Try progressively shorter substrings (skip full query — already tried)
    words = query.split()
    for n in range(len(words) - 1, 0, -1):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i : i + n])
            # Skip very short candidates (likely noise)
            if len(candidate) < 3:
                continue
            result = store.resolve_product(candidate)
            if result:
                return result

    return None


def _extract_category_hint(query: str) -> str | None:
    """Extract a likely category from the query for filtering.

    Simple keyword mapping — the LLM-based route classifier
    should handle this, but this serves as a fast deterministic fallback.
    """
    category_keywords = {
        "phone": "Smartphones",
        "smartphone": "Smartphones",
        "laptop": "Laptops",
        "notebook": "Laptops",
        "tablet": "Tablets",
        "headphone": "Headphones",
        "earbud": "Wireless Earbuds",
        "speaker": "Speakers",
        "keyboard": "Peripherals",
        "mouse": "Peripherals",
        "monitor": "Peripherals",
        "camera": "Smartphones",  # default to phone cameras
        "sneaker": "Sneakers",
        "shoe": "Sneakers",
        "coffee": "Coffee Makers",
        "blender": "Blenders",
    }
    lowered = query.lower()
    for keyword, category in category_keywords.items():
        if keyword in lowered:
            return category
    return None


def _retrieve_product_chunks(query: str, product_names: list[str] | None = None) -> str:
    """Retrieve relevant chunks from product descriptions via LlamaIndex.

    Args:
        query: The user's question.
        product_names: If provided, filter to only these products' chunks.

    Returns:
        Formatted string of retrieved chunks, or empty string on failure.
    """
    try:
        return retrieve_product_chunks(query, product_names=product_names)
    except Exception as e:
        logger.warning("[product_qa] RAG retrieval failed: %s", e)
        return ""


def _fallback_search(query: str, store) -> str:
    """Fallback: do a general product search when no specific product matched."""
    category_hint = _extract_category_hint(query)
    candidates = store.search_products(category=category_hint, limit=5)
    if not candidates:
        return (
            f"I couldn't find any products matching '{query}'. "
            "Could you be more specific about which product you're asking about?"
        )

    return _answer_comparison_query(query, store, candidates=candidates)
