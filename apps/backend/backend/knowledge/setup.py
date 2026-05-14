"""
Entry point for setting up the knowledge graph database.

Usage:
    python -m backend.knowledge.setup
"""

from backend.knowledge.schema import setup_knowledge_db

if __name__ == "__main__":
    setup_knowledge_db()
