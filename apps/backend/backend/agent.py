import os
import time
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from backend.memory import MemoryStore
from backend.resilience import (
    CircuitBreaker,
    Fallbacks,
    make_retry_decorator,
)
from backend.tools import order_status_tool, list_orders_tool, policy_retriever_tool, get_current_weather
from backend.config import settings


class AgentManager:
    """Manages the e-commerce support agent with caching, resilience, and streaming."""

    def __init__(self):
        self.pg_connection = settings.database_url
        self.memory_store = MemoryStore(filepath=settings.memory_filepath, max_history=8)
        self.tools = [order_status_tool, list_orders_tool, policy_retriever_tool, get_current_weather]
        self._cache_vectorstore = None
        self._llm = None
        self._llm_circuit = None
        self._agent = None
        self._agent_call_count = 0
        self._agent_call_reset_time = time.time()

    @property
    def cache_vectorstore(self):
        if self._cache_vectorstore is None:
            from langchain_community.vectorstores import PGVector
            from langchain_openai import OpenAIEmbeddings
            self._cache_vectorstore = PGVector(
                connection_string=self.pg_connection,
                embedding_function=OpenAIEmbeddings(
                    model=settings.embedding_model,
                    openai_api_key=settings.openai_api_key,
                    openai_api_base=settings.openai_api_base,
                ),
                collection_name="semantic_cache",
                distance_strategy="cosine",
            )
        return self._cache_vectorstore

    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.openai_model,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_api_base,
                max_retries=2,
                timeout=30,
                streaming=True,
            )
        return self._llm

    @property
    def llm_circuit(self):
        if self._llm_circuit is None:
            self._llm_circuit = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=60.0)
        return self._llm_circuit

    @property
    def agent(self):
        if self._agent is None:
            self._agent = create_agent(self.llm, self.tools)
        return self._agent

    def get_cached_response(self, query: str, intent: str = ""):
        docs_and_scores = self.cache_vectorstore.similarity_search_with_score(query, k=1)
        if docs_and_scores and docs_and_scores[0][1] < 0.3:
            metadata = docs_and_scores[0][0].metadata
            response = metadata.get("response")
            cached_intent = metadata.get("intent", "")

            # Intent changed since last cache → stale answer. Example:
            # Old run classified as "knowledge" (no tool result) → "I don't know"
            # New run correctly classified as "policy" → need fresh retrieval.
            if intent and cached_intent and intent != cached_intent:
                logger.info("[cache] Intent mismatch (cached=%s, current=%s) — skipping", cached_intent, intent)
                return None

            # Belt-and-suspenders: never return cached weather responses.
            # Weather data is real-time and should always be fetched fresh.
            if response and any(k in response.lower() for k in ("°c", "temperature", "wind speed", "weather in")):
                logger.info("[cache] Rejecting weather-related cached response")
                return None
            return response
        return None

    def cache_response(self, query: str, response: str, intent: str = ""):
        self.cache_vectorstore.add_texts([query], metadatas=[{"response": response, "intent": intent}])

    def extract_agent_response(self, result: object) -> str:
        """Extract the final AI message content from agent executor result."""
        raw = ""
        if isinstance(result, dict):
            if "output" in result:
                raw = str(result["output"])
            elif "messages" in result:
                messages = result["messages"]
                if isinstance(messages, list) and messages:
                    last = messages[-1]
                    if hasattr(last, "content"):
                        raw = str(last.content)
                    elif isinstance(last, dict) and "content" in last:
                        raw = str(last["content"])
        else:
            raw = str(result)

        if "Final Answer:" in raw:
            return raw.split("Final Answer:")[-1].strip()
        return raw

    def clean_query(self, query: str) -> str:
        """Fix typos and normalize the user query before retrieval."""
        prompt = (
            "Fix any spelling mistakes in the user query. "
            "Output ONLY the corrected query with no explanation, no quotes, and no extra text.\n\n"
            f"User query: {query}\n"
            "Corrected query:"
        )
        try:
            response = self.llm_circuit.call(
                self._clean_query_api_call,
                Fallbacks.clean_query_failed,
                prompt,
            )
            cleaned = response.content.strip().strip('"').strip("'")
            return cleaned if cleaned else query
        except Exception as e:
            logger.warning(f"[clean_query] Failed after retries: {e}")
            return query

    @make_retry_decorator(max_attempts=2)
    def _clean_query_api_call(self, prompt: str):
        return self.llm.invoke(prompt)

    def clear_semantic_cache(self):
        """Delete all entries from the semantic cache. Use after prompt/intent changes."""
        try:
            self.cache_vectorstore.delete_collection()
            logger.info("[cache] Semantic cache cleared")
        except Exception as e:
            logger.warning("[cache] Failed to clear cache: %s", e)

    def run_agent_with_cache(self, user_input: str, max_agent_calls: int = 5):
        """
        Run the agent with resilience patterns:
        - Typo correction with retry + circuit breaker
        - Agent invocation with retry + circuit breaker
        - Max call limit to prevent infinite loops
        """
        if time.time() - self._agent_call_reset_time > 60:
            self._agent_call_count = 0
            self._agent_call_reset_time = time.time()

        self._agent_call_count += 1
        if self._agent_call_count > max_agent_calls:
            logger.error(f"[agent] Exceeded max calls ({max_agent_calls}) -- possible loop")
            return (
                "I'm having trouble processing your request right now. "
                "Please try rephrasing your question or contact support."
            )

        cleaned_input = self.clean_query(user_input)
        self.memory_store.add_user(cleaned_input)

        cached = self.get_cached_response(cleaned_input)
        if cached:
            print("Cache hit!")
            self.memory_store.add_agent(cached)
            return cached

        memory_messages = self.memory_store.get_recent_messages()
        conversation_history = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in memory_messages]
        ) if memory_messages else "No prior conversation."

        def _invoke_agent():
            full_content = f"""You are a helpful e-commerce support agent for an online store.

You have access to the following tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in self.tools])}

When you are not sure, answer honestly and do not hallucinate.
Use the exact available tools for order, policy, or weather lookup when needed.
- For questions about a specific order, use order_status_tool.
- For questions about all orders, use list_orders_tool.
- For questions about store policies, use policy_retriever_tool.
- For questions about weather in a specific city, use get_current_weather.

Use the following format for tool use:
Thought: you should always think about what to do
Action: the action to take, should be one of {', '.join([tool.name for tool in self.tools])}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation history:
{conversation_history}

Question: {cleaned_input}"""

            return self.agent.invoke(
                {
                    "messages": [HumanMessage(content=full_content)]
                }
            )

        try:
            result = self.llm_circuit.call(_invoke_agent, Fallbacks.llm_unavailable)
            response = self.extract_agent_response(result)

            if response == Fallbacks.llm_unavailable():
                return response

            self.cache_response(cleaned_input, response)
            self.memory_store.add_agent(response)
            return response

        except Exception as e:
            logger.error(f"[agent] Invocation failed: {e}")
            from backend.resilience import is_transient_error
            if is_transient_error(e):
                return (
                    "I'm experiencing a temporary issue. "
                    "Please try again in a moment."
                )
            return (
                "I couldn't process your request. "
                "Please check your question and try again."
            )

    async def stream_agent_response(self, user_input: str, max_agent_calls: int = 5):
        """
        Async generator that yields the agent's final answer with a smooth
        streaming effect. Tokens are collected and replayed in small chunks
        so the frontend shows a visible typing animation.
        """
        if time.time() - self._agent_call_reset_time > 60:
            self._agent_call_count = 0
            self._agent_call_reset_time = time.time()

        self._agent_call_count += 1
        if self._agent_call_count > max_agent_calls:
            logger.error(f"[agent] Exceeded max calls ({max_agent_calls}) -- possible loop")
            yield "I'm having trouble processing your request right now. Please try rephrasing your question or contact support."
            return

        cleaned_input = self.clean_query(user_input)
        self.memory_store.add_user(cleaned_input)

        cached = self.get_cached_response(cleaned_input)
        if cached:
            print("Cache hit!")
            self.memory_store.add_agent(cached)
            for word in cached.split(" "):
                yield word + (" " if word != cached.split(" ")[-1] else "")
                await asyncio.sleep(0.03)
            return

        memory_messages = self.memory_store.get_recent_messages()
        conversation_history = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in memory_messages]
        ) if memory_messages else "No prior conversation."

        full_content = f"""You are a helpful e-commerce support agent for an online store.
            You have access to the following tools:
            {chr(10).join([f"- {tool.name}: {tool.description}" for tool in self.tools])}

            When you are not sure, answer honestly and do not hallucinate.
            Use the exact available tools for order, policy, or weather lookup when needed.
            - For questions about a specific order, use order_status_tool.
            - For questions about all orders, use list_orders_tool.
            - For questions about store policies, use policy_retriever_tool.
            - For questions about weather in a specific city, use get_current_weather.

            Use the following format for tool use:
            Thought: you should always think about what to do
            Action: the action to take, should be one of {', '.join([tool.name for tool in self.tools])}
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Conversation history:
            {conversation_history}

            Question: {cleaned_input}
"""

        def _chunk_to_text(chunk: object) -> str:
            if chunk is None:
                return ''
            if isinstance(chunk, str):
                return chunk
            if isinstance(chunk, dict):
                return str(chunk.get('content') or chunk.get('text') or '')
            return str(getattr(chunk, 'content', None) or getattr(chunk, 'text', None) or '')

        try:
            full_response = ""
            got_stream = False

            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=full_content)]},
                version="v2",
            ):
                event_name = event.get("event")
                data = event.get("data", {})

                if event_name == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    chunk_text = _chunk_to_text(chunk)
                    if not chunk_text:
                        continue

                    got_stream = True
                    full_response += chunk_text
                    yield chunk_text
                    continue

                if event_name == "on_chat_model_end" and not got_stream:
                    output = _chunk_to_text(data.get("output"))
                    if not output:
                        continue

                    full_response += output
                    yield output

            if full_response:
                self.cache_response(cleaned_input, full_response)
                self.memory_store.add_agent(full_response)

        except Exception as e:
            logger.error(f"[agent] Streaming failed: {e}")
            from backend.resilience import is_transient_error
            if is_transient_error(e):
                yield "I'm experiencing a temporary issue. Please try again in a moment."
            else:
                yield "I couldn't process your request. Please check your question and try again."


# -- Global Instance ------------------------------------------------------------

agent_manager = AgentManager()

# -- Module-level aliases for other modules to import --------------------------

clean_query = agent_manager.clean_query
get_cached_response = agent_manager.get_cached_response
cache_response = agent_manager.cache_response
llm = agent_manager.llm
memory_store = agent_manager.memory_store

# -- Convenience Functions -----------------------------------------------------

def run_agent_with_cache(user_input: str, max_agent_calls: int = 5):
    return agent_manager.run_agent_with_cache(user_input, max_agent_calls)


async def stream_agent_response(user_input: str, max_agent_calls: int = 5):
    async for chunk in agent_manager.stream_agent_response(user_input, max_agent_calls):
        yield chunk


if __name__ == "__main__":
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == "quit":
            break
        response = run_agent_with_cache(user_input)
        print(response)
