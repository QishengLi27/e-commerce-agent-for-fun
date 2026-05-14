# LangChain & LangGraph Streaming Protocol — End-to-End Trace

This traces every streaming mechanism: `stream()`, `astream()`, `astream_events()`, `transform()`, and how LangGraph integrates them.

---

## The 4 Streaming APIs

| API | Output | Use case |
|-----|--------|----------|
| `stream()` / `astream()` | Chunks of the final output | Token-by-token LLM output |
| `astream_events()` | Events (start/stream/end) | UI progress indicators, debugging |
| `transform()` | Iterator → Iterator | Chaining streaming steps in RunnableSequence |
| LangGraph `stream_mode` | State updates per node | Values, updates, checkpoints, debug |

---

# Part 1: `stream()` / `astream()` — Chunking Output

**File:** `langchain_core/runnables/base.py`

### The default implementation (for non-streaming Runnables)

```python
class Runnable:
    async def astream(self, input, config=None, **kwargs):
        # Default: just yield the entire output as one chunk
        yield await self.ainvoke(input, config, **kwargs)
```

For a `RunnableLambda`, this yields the entire result as a single chunk.

### ChatOpenAI.astream() — the real streaming

**File:** `langchain_openai/chat_models/base.py`

```python
# ChatOpenAI overrides _astream() to yield token-by-token
async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
    payload = self._get_request_payload(messages, stop=stop, **kwargs)

    # Call OpenAI API with stream=True
    async for chunk in self.async_client.chat.completions.create(
        **payload, stream=True
    ):
        # Each chunk is a tiny JSON object with partial content
        # chunk.choices[0].delta.content = "Hel"
        # chunk.choices[0].delta.content = "lo"
        message_chunk = _convert_delta_to_message_chunk(chunk)
        yield ChatGenerationChunk(message=message_chunk)
```

The OpenAI streaming API returns partial text as it's generated. LangChain wraps each chunk in a `ChatGenerationChunk` with an `AIMessageChunk`.

### RunnableSequence.astream() — chaining streaming steps

**File:** `langchain_core/runnables/base.py:4123`

```python
async def astream(self, input, config=None, **kwargs):
    async def input_aiter() -> AsyncIterator[Input]:
        yield input              # Wrap single input in an async iterator

    async for chunk in self.atransform(input_aiter(), config):
        yield chunk
```

This delegates to `atransform()`, which chains each step's streaming:

```python
async def _atransform(self, inputs, run_manager, config, **kwargs):
    final_pipeline = inputs       # Start with the input iterator
    for idx, step in enumerate(self.steps):
        # Chain this step's transform onto the pipeline
        final_pipeline = step.atransform(final_pipeline, config)
    async for output in final_pipeline:
        yield output
```

**How it works for `prompt | llm | parser`:**

```
inputs: AsyncIterator[{"topic": "cats"}]              ← yields once

step[0] atransform:  ChatPromptTemplate._atransform()
  → reads {"topic": "cats"}, yields ChatPromptValue once

step[1] atransform:  ChatOpenAI._atransform()
  → reads ChatPromptValue, yields AIMessageChunks:
     "Cats" → " are" → " small" → " mammals" → ...

step[2] atransform:  output_parser._atransform()
  → reads each AIMessageChunk, yields parsed chunks:
     "Cats" → " are" → " small" → " mammals" → ...

FINAL output: "Cats are small mammals..." streamed chunk-by-chunk
```

**Key insight:** Each step in the chain acts as a **transform** — it reads from the previous step's streaming iterator and yields its own streaming output. If a step doesn't support streaming (e.g., `RunnableLambda`), it buffers all input, calls `invoke()`, and yields the result as one chunk. Streaming pauses at that point and resumes for subsequent steps.

---

# Part 2: `astream_events()` — Structured Event Streaming

**File:** `langchain_core/runnables/base.py:1317`

This is the most powerful streaming API. It emits events at every lifecycle point:

```python
async def astream_events(
    self, input, config=None, *,
    version="v2",
    include_names=None, include_types=None, include_tags=None,
    exclude_names=None, exclude_types=None, exclude_tags=None,
    **kwargs,
) -> AsyncIterator[StreamEvent]:
```

### The event types

| Event | Emitted when | `data` field |
|-------|-------------|-------------|
| `on_chain_start` | A chain/runnable starts executing | `{"input": ...}` |
| `on_chain_stream` | A chain/runnable yields a chunk | `"partial_output_chunk"` |
| `on_chain_end` | A chain/runnable finishes | `{"output": ...}` |
| `on_chat_model_start` | An LLM call starts | `{"input": messages}` |
| `on_chat_model_stream` | An LLM yields a token | `AIMessageChunk(content="Hel")` |
| `on_chat_model_end` | An LLM call finishes | `{"output": full_message}` |
| `on_tool_start` | A tool starts executing | `{"input": tool_args}` |
| `on_tool_end` | A tool finishes executing | `{"output": tool_result}` |
| `on_retriever_start` | A retriever starts searching | `{"input": query}` |
| `on_retriever_end` | A retriever finishes | `{"output": documents}` |
| `on_prompt_start` | A prompt template starts formatting | `{"input": template_vars}` |
| `on_prompt_end` | A prompt template finishes | `{"output": ChatPromptValue}` |

### How it works under the hood

`astream_events()` creates a callback handler that intercepts all events and yields them:

```python
async def astream_events(self, input, config=None, *, version="v2", ...):
    # 1. Create a special callback handler
    event_streamer = _EventStreamingCallbackHandler(...)

    # 2. Inject it into config
    config = patch_config(config, callbacks=[event_streamer])

    # 3. Run the chain normally (invoke or stream)
    task = asyncio.create_task(self.ainvoke(input, config, **kwargs))

    # 4. Yield events as they arrive from the callback handler
    async for event in event_streamer.events():
        if _should_include(event, include_names, include_types, ...):
            yield event

    # 5. Wait for the chain to complete
    await task
```

**Key design:** `astream_events()` doesn't change how the chain runs. It intercepts the callback system. Every `invoke()`/`stream()` calls `on_chain_start`/`on_chain_stream`/`on_chain_end`. The event streamer captures these and yields them as structured `StreamEvent` dicts.

### The v2 event format

```python
class StreamEvent(TypedDict):
    event: str          # "on_chat_model_stream"
    name: str           # "ChatOpenAI"
    run_id: str         # "uuid-abc-123"
    tags: list[str]     # ["production"]
    metadata: dict      # {"request_id": "xyz"}
    data: dict          # {"chunk": AIMessageChunk(content="Hel")}
    parent_ids: list[str]  # ["uuid-root"]
```

---

# Part 3: `transform()` — The Iterator Protocol

**File:** `langchain_core/runnables/base.py:1563`

```python
class Runnable:
    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        # Default: buffer all input, call invoke(), yield result
        final = None
        for chunk in input:
            if final is None:
                final = chunk
            else:
                final = final + chunk  # or some merge logic

        if final is None:
            return
        yield self.invoke(final, config, **kwargs)
```

**Override for streaming-capable components:**

```python
# ChatOpenAI overrides transform to yield tokens
class ChatOpenAI:
    def _transform(self, messages_iter, run_manager, config, **kwargs):
        messages = []                     # Buffer all messages
        for chunk in messages_iter:
            messages.append(chunk)        # Must have full prompt to start

        # Now stream the response
        for token in self._stream(messages):  # token-by-token from API
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
```

**The pattern:** A streaming component must:
1. **Buffer** enough input to start generating (for LLMs: the full prompt)
2. **Yield** output chunks as they become available

If a step buffers everything before yielding, streaming stalls at that step. This is why `RunnableLambda` breaks streaming — it has one input and one output, not `Iterator[Input] → Iterator[Output]`.

---

# Part 4: LangGraph Streaming

**File:** `langgraph/pregel/main.py`

LangGraph has its own streaming system, orthogonal to LangChain's:

```python
graph.stream(input, config, stream_mode="values")
graph.stream(input, config, stream_mode="updates")
graph.stream(input, config, stream_mode="messages")     # Token-by-token from LLM
graph.stream(input, config, stream_mode="checkpoints")  # Checkpoint saved events
graph.stream(input, config, stream_mode="debug")        # Detailed debugging
```

### stream_mode="values" — Full state after each step

```python
# Yields the full AgentState after each super-step
yield {"messages": [...], "intent": "policy", "final_answer": None, ...}
yield {"messages": [...], "intent": "policy", "final_answer": "Our policy...", ...}
```

### stream_mode="updates" — Only what changed

```python
# Yields the partial state updates from each node
yield {"sanitize_input": {"cleaned_input": "...", "cached": False}}
yield {"classify_intent": {"intent": "policy"}}
yield {"policy_node": {"tool_result": "Return Policy: ..."}}
```

### stream_mode="messages" — LLM token streaming

```python
# Yields (AIMessageChunk, metadata) tuples from inside the LLM node
yield (AIMessageChunk(content="Our"), {"langgraph_step": 3, ...})
yield (AIMessageChunk(content=" return"), {"langgraph_step": 3, ...})
yield (AIMessageChunk(content=" policy"), {"langgraph_step": 3, ...})
```

**How it works:** LangGraph injects a `StreamMessagesHandler` callback that traps `on_chat_model_stream` events from LLM calls inside nodes and re-emits them as `(token, metadata)` tuples:

```python
# From Pregel.stream()
if "messages" in stream_modes:
    callbacks.append(
        StreamMessagesHandler(stream.put, subgraphs, parent_ns)
    )
```

### The StreamProtocol

The stream is implemented as a queue with a protocol:

```python
class StreamProtocol:
    def __call__(self, value: StreamChunk):
        # value = (namespace, mode, data)
        # namespace: tuple[str, ...] for subgraph nesting
        # mode: "values" | "updates" | "messages" | "checkpoints" | "debug"
        # data: the actual payload
        if mode in self.modes:
            self.queue.put(value)
```

---

# Part 5: Real Streaming vs Fake Streaming in Your Project

### What you have now (fake streaming)

```python
# backend/api/routes.py
async def _stream_response(agent_graph, initial_state):
    result = await loop.run_in_executor(None, agent_graph.invoke, initial_state)
    # ← waits for ENTIRE graph to finish

    answer = result["final_answer"]
    for word in answer.split(" "):
        yield word + " "
        await asyncio.sleep(0.03)  # ← artificial delay
```

### What real streaming looks like

```python
# Real streaming with LangGraph
async def _stream_response_real(agent_graph, initial_state):
    async for event in agent_graph.astream(
        initial_state,
        config,
        stream_mode=["messages", "values"]
    ):
        if event[1] == "messages":  # (namespace, mode, data)
            token, metadata = event[2]
            yield token.content        # ← real token as LLM generates it
        elif event[1] == "values":
            # Can yield state updates when each node completes
            pass
```

---

# Interview Talking Points

## "How does streaming work through a chain?"

> "Each step in a `RunnableSequence` implements `transform()` which takes an `Iterator[Input]` and returns an `Iterator[Output]`. The chain chains these iterators together. If a step buffers all input before yielding (like a `RunnableLambda`), streaming stalls at that boundary. LLMs buffer the full prompt, then yield tokens one at a time."

## "What's the difference between stream() and astream_events()?"

> "`stream()` yields output chunks — useful for token-by-token display. `astream_events()` yields lifecycle events — `on_llm_start`, `on_llm_stream`, `on_llm_end` — with metadata like run IDs and parent IDs. It works by injecting a callback handler into the config so it intercepts all events without changing the chain's execution."

## "How does LangGraph stream compare to LangChain stream?"

> "LangGraph's `stream()` operates at the graph level — it yields state after each node (values mode) or only the changes (updates mode). LangChain's `stream()` operates at the chain level — it yields tokens from the LLM. LangGraph's `messages` mode bridges the gap: it uses a callback handler to intercept LLM token streaming from inside graph nodes and re-emits them as `(token, metadata)` tuples."

## "Why does your project have fake streaming?"

> "The current implementation calls `agent_graph.invoke()` which blocks until the entire graph finishes, then replays tokens with delays. Real streaming requires switching to `agent_graph.astream()` with `stream_mode='messages'`, which yields tokens as the LLM generates them. The graph must be compiled with appropriate callbacks to intercept the internal LLM calls."
