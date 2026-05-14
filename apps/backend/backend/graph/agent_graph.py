"""
LangGraph StateGraph definition for the e-commerce support agent.
"""

from langgraph.graph import StateGraph, END

from backend.graph.nodes import (
    AgentState,
    sanitize_input,
    classify_intent,
    route_by_intent,
    order_node,
    list_orders_node,
    policy_node,
    weather_node,
    generate_reply,
    validate_reply,
    update_memory,
)

# ─── Build Graph ─────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

# Register nodes
builder.add_node("sanitize_input", sanitize_input)
builder.add_node("classify_intent", classify_intent)
builder.add_node("order_node", order_node)
builder.add_node("list_orders_node", list_orders_node)
builder.add_node("policy_node", policy_node)
builder.add_node("weather_node", weather_node)
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
        "generate_reply": "generate_reply",
    },
)

# All tool nodes converge to generate_reply
builder.add_edge("order_node", "generate_reply")
builder.add_edge("list_orders_node", "generate_reply")
builder.add_edge("policy_node", "generate_reply")
builder.add_edge("weather_node", "generate_reply")

# Final steps
# Validate before persisting
builder.add_edge("generate_reply", "validate_reply")
builder.add_edge("validate_reply", "update_memory")
builder.add_edge("update_memory", END)

# Compile
agent_graph = builder.compile()
