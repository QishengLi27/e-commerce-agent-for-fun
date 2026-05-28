"""
LangGraph StateGraph definition for the e-commerce support agent.

Usage:
    # During app startup (async):
    await init_agent_graph()

    # During request handling (sync or async):
    from backend.graph.agent_graph import get_agent_graph
    graph = get_agent_graph()

The graph is a lazy singleton — call init_agent_graph() once during FastAPI
startup, then use get_agent_graph() everywhere else.
"""

from langgraph.graph import END, StateGraph

from backend.graph.nodes import (
    AgentState,
    classify_intent,
    generate_reply,
    knowledge_node,
    list_orders_node,
    order_node,
    policy_node,
    route_after_validation,
    route_by_intent,
    sanitize_input,
    update_memory,
    validate_reply,
    weather_node,
)

# ─── Lazy singleton ────────────────────────────────────────────────────────────

_agent_graph = None


async def init_agent_graph():
    """Initialize the agent graph asynchronously. Call once during app startup."""
    global _agent_graph
    if _agent_graph is not None:
        return _agent_graph

    from backend.checkpoint import aget_checkpointer
    checkpointer = await aget_checkpointer()
    _agent_graph = _build_graph(checkpointer)
    return _agent_graph


def get_agent_graph():
    """Return the compiled agent graph (must call init_agent_graph first)."""
    global _agent_graph
    if _agent_graph is None:
        from backend.checkpoint import get_checkpointer
        _agent_graph = _build_graph(get_checkpointer())
    return _agent_graph


def _build_graph(checkpointer):
    """Build and compile the StateGraph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("sanitize_input", sanitize_input)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("order_node", order_node)
    builder.add_node("list_orders_node", list_orders_node)
    builder.add_node("policy_node", policy_node)
    builder.add_node("weather_node", weather_node)
    builder.add_node("knowledge_node", knowledge_node)
    builder.add_node("generate_reply", generate_reply)
    builder.add_node("validate_reply", validate_reply)
    builder.add_node("update_memory", update_memory)

    # Entry point
    builder.set_entry_point("sanitize_input")

    # Edges
    builder.add_edge("sanitize_input", "classify_intent")

    # Conditional routing based on intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "order": "order_node",
            "list_orders": "list_orders_node",
            "policy": "policy_node",
            "weather": "weather_node",
            "knowledge": "knowledge_node",
            "generate_reply": "generate_reply",
        },
    )

    # All tool nodes converge to generate_reply
    builder.add_edge("order_node", "generate_reply")
    builder.add_edge("list_orders_node", "generate_reply")
    builder.add_edge("policy_node", "generate_reply")
    builder.add_edge("weather_node", "generate_reply")
    builder.add_edge("knowledge_node", "generate_reply")

    # Final steps with self-correction loop
    builder.add_edge("generate_reply", "validate_reply")
    builder.add_conditional_edges(
        "validate_reply",
        route_after_validation,
        {
            "generate_reply": "generate_reply",
            "update_memory": "update_memory",
        },
    )
    builder.add_edge("update_memory", END)

    return builder.compile(checkpointer=checkpointer)
