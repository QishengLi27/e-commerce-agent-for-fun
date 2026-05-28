"""
Knowledge graph tools for product and category lookups.

These tools query the PostgreSQL-backed knowledge graph for structured
product/category/policy relationships — complementary to the vector
retrieval used for free-text policy search.
"""

from langchain.tools import tool

from backend.knowledge.graph_store import get_knowledge_store


@tool
def product_info_tool(query: str) -> str:
    """Look up product information including category, price, and applicable policies.
    Use this when the user asks about a specific product, such as:
    - 'what category is <product>?'
    - 'what policies apply to <product>?'
    - 'tell me about <product>'
    - 'can I return <product>?'
    - 'is <product> covered by warranty?'
    """
    store = get_knowledge_store()

    info = store.get_product_info(query)
    if not info:
        return f"No product found matching '{query}'."

    lines = [
        f"Product: {info['name']}",
        f"Category: {info['category_name']}",
    ]
    if info.get("price"):
        lines.append(f"Price: ${info['price']:.2f}")
    if info.get("sku"):
        lines.append(f"SKU: {info['sku']}")

    if info.get("policies"):
        lines.append("\nApplicable Policies:")
        for p in info["policies"]:
            lines.append(f"  - [{p['type'].upper()}] {p['summary']}")

    return "\n".join(lines)


@tool
def category_info_tool(query: str) -> str:
    """List categories or find which policies apply to a category.
    Use this when the user asks:
    - 'what categories do you have?'
    - 'what products are in <category>?'
    - 'what policies apply to <category>?'
    """
    store = get_knowledge_store()

    # If query asks to list categories
    if any(w in query.lower() for w in ["list", "all", "categories", "what categories"]):
        cats = store.get_all_categories()
        if not cats:
            return "No categories found."
        lines = ["Product Categories:"]
        for c in cats:
            lines.append(f"  - {c['name']}: {c.get('description', '')}")
        return "\n".join(lines)

    # Search for products in a category
    products = store.search_products(query)
    if products:
        lines = [f"Products matching '{query}':"]
        for p in products:
            lines.append(
                f"  - {p['name']} (Category: {p['category_name']}, "
                f"Price: ${p['price']:.2f})"
            )
        return "\n".join(lines)

    # Try category-level policy lookup
    policies = store.query_category_policies(query)
    if policies:
        category_name = policies[0].get("category_name", query)
        lines = [f"Policies for category '{category_name}':"]
        for p in policies:
            lines.append(f"  - [{p['policy_type'].upper()}] {p['summary']}")
        return "\n".join(lines)

    return f"No products or categories found matching '{query}'."
