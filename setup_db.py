import sqlite3

def setup_database():
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()

    # Create orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_name TEXT,
            status TEXT,
            estimated_delivery TEXT
        )
    ''')

    # Insert mock orders
    mock_orders = [
        ('1001', 'John Doe', 'Delivered', '2023-10-15'),
        ('1002', 'Jane Smith', 'Shipped', '2023-10-20'),
        ('1003', 'Bob Johnson', 'Processing', '2023-10-25'),
        ('1004', 'Alice Brown', 'Delivered', '2023-10-10')
    ]

    cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?)', mock_orders)

    # Commit and close
    conn.commit()
    conn.close()
    print("Database setup complete.")

def get_order_status(order_id: str):
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()
    cursor.execute('SELECT status, estimated_delivery FROM orders WHERE order_id = ?', (order_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        status, delivery = result
        return f"Order {order_id} status: {status}. Estimated delivery: {delivery}."
    else:
        return f"Order {order_id} not found."

if __name__ == "__main__":
    setup_database()