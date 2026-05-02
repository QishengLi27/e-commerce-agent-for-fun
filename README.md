# Smart E-Commerce Support Agent

A lightweight AI customer service agent for an online store using LangChain, ChromaDB, SQLite, and semantic caching.

## Setup

1. Create a virtual environment and install required packages:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you still encounter dependency issues, reinstall with the pinned versions:
   ```
   pip install --force-reinstall -r requirements.txt
   ```

2. Set up the database:
   ```
   python setup_db.py
   ```

3. Set up the vector database:
   ```
   python setup_vector_db.py
   ```

4. Run the agent:
   ```
   python agent.py
   ```

## Test Scenarios

1. **Test RAG/Vector DB**: "What is your policy on returning electronics?"
2. **Test Relational DB**: "What is the status of order 1001?"
3. **Test Cache**: Ask "What is the return policy?" then "How do I return items?"
4. **Test Multi-Step**: "Can I still return the item in order 1001?" (assuming order 1001 was delivered 40 days ago)

## Notes

- Uses OpenAI API for LLM and embeddings. Set your API key in environment variables.
- For local LLM, replace with Ollama integration.
- Database and vector store are persisted locally.