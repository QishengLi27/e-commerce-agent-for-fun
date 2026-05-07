from langchain.tools import tool
from backend.db.setup import get_order_status, get_all_orders


@tool
def order_status_tool(order_id: str) -> str:
    """Get the status of an order by order ID."""
    return get_order_status(order_id)


@tool
def list_orders_tool() -> str:
    """List all orders in the system."""
    return get_all_orders()
