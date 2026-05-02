from langchain_openai import ChatOpenAI
from langchain_community.cache import InMemoryCache
# from langchain_community.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
# from langchain.chains import RetrievalQA
from setup_db import get_order_status
from retrieval import get_policy_retriever
import os

# Set up semantic cache for responses
cache_embeddings = OpenAIEmbeddings(
    model="embedding-2",  # or "embedding-3"
    openai_api_key = "51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU", # os.getenv("ZHIPUAI_API_KEY"), 
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/" # GLM's OpenAI-compatible endpoint
)
cache_vectorstore = Chroma(collection_name="semantic_cache", embedding_function=cache_embeddings, persist_directory="./cache_db")

def get_cached_response(query: str):
    docs_and_scores = cache_vectorstore.similarity_search_with_score(query, k=1)
    if docs_and_scores and docs_and_scores[0][1] < 0.3:  # Lower score means more similar in Chroma (cosine distance)
        return docs_and_scores[0][0].metadata.get('response')
    return None

def cache_response(query: str, response: str):
    cache_vectorstore.add_texts([query], metadatas=[{"response": response}])

# Load hybrid retriever (Chroma + BM25 + RRF + optional LLM re-rank)
_policy_retriever = get_policy_retriever()


def clean_query(query: str) -> str:
    """Fix typos and normalize the user query before retrieval."""
    prompt = (
        "Fix any spelling mistakes in the user query. "
        "Output ONLY the corrected query with no explanation, no quotes, and no extra text.\n\n"
        f"User query: {query}\n"
        "Corrected query:"
    )
    try:
        response = llm.invoke(prompt)
        cleaned = response.content.strip().strip('"').strip("'")
        return cleaned if cleaned else query
    except Exception:
        return query


@tool
def order_status_tool(order_id: str) -> str:
    """Get the status of an order by order ID."""
    return get_order_status(order_id)


@tool
def policy_retriever_tool(query: str) -> str:
    """Retrieve store policies related to the query using hybrid search."""
    docs_and_scores = _policy_retriever.retrieve(query, k=3, rerank=True)
    # Only include highly relevant chunks to avoid flooding the LLM with noise
    filtered = [(doc, score) for doc, score in docs_and_scores if score >= 7]
    if not filtered:
        filtered = docs_and_scores[:1]  # fallback to top-1 if nothing scores high
    return "\n\n".join([doc.page_content for doc, _ in filtered])

# Initialize LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo")
# Initialize LLM with GLM API
llm = ChatOpenAI(
    model="glm-4-flash",  # "glm-4" or "glm-4-flash" supports tool calling
    openai_api_key = "51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU", #os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/"
)

# Define tools
tools = [order_status_tool, policy_retriever_tool]

# System prompt
system_prompt = """
You are a helpful e-commerce support agent. Use the order_status_tool for order statuses and the policy_retriever_tool for store policy questions.
"""

# Create agent
agent = create_agent(llm, tools, system_prompt=system_prompt)

def extract_agent_response(result: object) -> str:
    """Extract the final AI message content from agent result."""
    if isinstance(result, dict):
        if "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and messages:
                last = messages[-1]
                # LangChain message objects have .content attribute
                if hasattr(last, "content"):
                    return str(last.content)
                # Fallback for dict-style messages
                if isinstance(last, dict) and "content" in last:
                    return str(last["content"])
        if "output" in result:
            return str(result["output"])
    return str(result)


def run_agent_with_cache(user_input: str):
    # Fix typos before the agent sees the query
    cleaned_input = clean_query(user_input)
    
    # Check semantic cache against the cleaned query
    cached = get_cached_response(cleaned_input)
    if cached:
        print("Cache hit!")
        return cached
    
    # Run agent with the cleaned query
    result = agent.invoke({"messages": [{"role": "user", "content": cleaned_input}]})
    response = extract_agent_response(result)
    
    # Cache the response against the cleaned query
    cache_response(cleaned_input, response)
    
    return response

if __name__ == "__main__":
    # Example usage
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == 'quit':
            break
        response = run_agent_with_cache(user_input)
        print(response)
