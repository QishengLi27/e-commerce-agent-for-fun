"""
Database setup for PostgreSQL (replaces SQLite).

Run this to create/reset the orders table:
    python -m backend.db.setup
"""

import psycopg2
from backend.config import settings

CONNECTION_STRING = settings.pg_connection_raw


def get_pg_connection():
    return psycopg2.connect(CONNECTION_STRING)


def setup_database():
    conn = get_pg_connection()
    cursor = conn.cursor()

    # Create orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT,
            status TEXT,
            estimated_delivery TEXT
        )
    """)

    # Insert mock orders
    mock_orders = [
        ("1001", "John Doe", "Delivered", "2023-10-15"),
        ("1002", "Jane Smith", "Shipped", "2023-10-20"),
        ("1003", "Bob Johnson", "Processing", "2023-10-25"),
        ("1004", "Alice Brown", "Delivered", "2023-10-10"),
    ]

    cursor.executemany(
        "INSERT INTO orders (order_id, customer_name, status, estimated_delivery) VALUES (%s, %s, %s, %s) ON CONFLICT (order_id) DO UPDATE SET customer_name=EXCLUDED.customer_name, status=EXCLUDED.status, estimated_delivery=EXCLUDED.estimated_delivery",
        mock_orders,
    )

    conn.commit()
    conn.close()
    print("Database setup complete.")


def get_order_status(order_id: str) -> str:
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status, estimated_delivery FROM orders WHERE order_id = %s",
        (order_id,),
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        status, delivery = result
        return f"Order {order_id} status: {status}. Estimated delivery: {delivery}."
    else:
        return f"Order {order_id} not found."


def get_all_orders() -> str:
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, customer_name, status, estimated_delivery FROM orders ORDER BY order_id"
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "No orders found."
    lines = ["Here are all orders:"]
    for order_id, customer_name, status, estimated_delivery in rows:
        lines.append(f"- Order {order_id} ({customer_name}): {status}, estimated delivery {estimated_delivery}")
    return "\n".join(lines)


if __name__ == "__main__":
    setup_database()
