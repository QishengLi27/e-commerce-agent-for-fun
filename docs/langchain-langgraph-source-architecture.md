# LangChain & LangGraph Source Code Architecture

This guide maps the codebase structure and traces a concrete execution path through both frameworks. It is designed to help you read the source code, not just use the APIs.

---

# Part 1: LangChain — The Layered Architecture

LangChain is not a single package. It is a family of packages with strict dependency rules:

```
langchain-core          (base abstractions, no providers)
    ↑
langchain-community     (third-party integrations: pgvector, bm25, etc.)
    ↑
langchain-openai        (OpenAI chat models, embeddings)
    ↑
langchain               (high-level chains, agents, pre-built recipes)
    ↑
langgraph               (state machines, built ON TOP of langchain-core)
```

Each layer can only import from layers below it. `langchain-core` has zero external provider dependencies.

---

## Layer 1: langchain-core — The Abstractions

This is where the contracts are defined. Every other package implements these interfaces.

### Key modules

| Module | Purpose | Your code uses it via |
|--------|---------|----------------------|
| `langchain_core.language_models` | `BaseLLM`, `BaseChatModel`, `BaseLanguageModel` | `ChatOpenAI` inherits from `BaseChatModel` |
| `langchain_core.messages` | `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` | Every node in your graph |
| `langchain_core.tools` | `BaseTool`, `tool` decorator | `@tool` on `order_status_tool` |
| `langchain_core.embeddings` | `Embeddings` base class | `OpenAIEmbeddings` inherits from it |
| `langchain_core.vectorstores` | `VectorStore` base class | `PGVector` inherits from it |
| `langchain_core.runnables` | `Runnable`, `RunnableSequence`, `RunnableLambda` | The composition layer — see below |
| `langchain_core.callbacks` | `BaseCallbackHandler`, callback managers | Logging, tracing, token counting |
| `langchain_core.outputs` | `LLMResult`, `ChatGeneration`, `GenerationChunk` | Internal result types |

### The Runnable Protocol — the center of everything

`Runnable` is an abstract base class that defines a uniform interface:

```python
# From langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):
    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output: ...

    def batch(self, inputs: List[Input], ...) -> List[Output]: ...
    def stream(self, input: Input, ...) -> Iterator[Output]: ...
    def ainvoke(self, input: Input, ...) -> Awaitable[Output]: ...
    # ... plus composition methods
```

**Why this matters:** Every component — LLM, prompt template, tool, vector store — implements `Runnable`. This means they all compose the same way.

```python
# Your code composes runnables implicitly:
chain = prompt_template | llm | output_parser
# The `|` operator is defined in Runnable — it creates a RunnableSequence
```

The `|` operator is syntactic sugar for:

```python
RunnableSequence(first=prompt_template, middle=[llm], last=output_parser)
```

### RunnableSequence — how the pipe works

File: `langchain_core/runnables/base.py` (class `RunnableSequence`)

```python
class RunnableSequence(Runnable[Input, Output]):
    def invoke(self, input, config=None):
        # Steps through each runnable in order, passing output[n] as input[n+1]
        for step in self.steps:
            input = step.invoke(input, config)
        return input
```

This is the engine behind every LangChain chain. Your `generate_reply` prompt → LLM → answer is a 2-step `RunnableSequence`.

### Callbacks — the cross-cutting concern

File: `langchain_core/callbacks/manager.py`

Callbacks are injected into every `invoke()`/`stream()` call. They let you observe without changing behavior:

```python
# Pseudo-flow inside every Runnable.invoke():
def invoke(self, input, config):
    callback_manager = get_callback_manager_for_config(config)
    callback_manager.on_llm_start(...)      # before
    output = self._call(input)              # actual work
    callback_manager.on_llm_end(...)        # after
    return output
```

Your `langchain_openai` ChatOpenAI uses this to emit `httpx` logs.

---

## Layer 2: langchain-openai — Provider Implementation

File: `langchain_openai/chat_models/base.py` (class `ChatOpenAI`)

`ChatOpenAI` inherits from `BaseChatModel` (in `langchain_core`), which inherits from `Runnable`.

```python
class ChatOpenAI(BaseChatModel):
    # Your config values map here:
    model: str = "glm-4-flash"          # passed to the API
    openai_api_key: str = ...           # authentication
    openai_api_base: str = ...          # routed to Zhipu
    temperature: float = 0.7
    max_retries: int = 2
```

### The invoke() flow

```python
# ChatOpenAI.invoke() → BaseChatModel.invoke() → ...
def _generate(self, messages, stop, run_manager, **kwargs):
    # 1. Convert LangChain messages to OpenAI dict format
    message_dicts = [_convert_message_to_dict(m) for m in messages]

    # 2. Call the OpenAI client (or httpx directly)
    response = self.client.create(
        model=self.model,
        messages=message_dicts,
        stream=False,
        **params
    )

    # 3. Convert OpenAI response back to LangChain AIMessage
    return ChatResult(generations=[ChatGeneration(message=AIMessage(...))])
```

The `create()` call goes through `openai-python` (the official OpenAI SDK), which handles retries, streaming, and JSON parsing. The Zhipu API base URL makes it transparent — LangChain doesn't know it's not talking to OpenAI.

---

## Layer 3: langchain-community — Third-Party Integrations

File: `langchain_community/vectorstores/pgvector.py` (class `PGVector`)

This is where integrations live. `PGVector` is not in `langchain-core` because it depends on `psycopg2` and `pgvector`.

```python
class PGVector(VectorStore):
    def __init__(self, connection_string, embedding_function, collection_name, ...):
        # Creates SQLAlchemy engine + session
        self._engine = create_engine(connection_string)
        # Creates tables if missing
        self.create_vector_extension()
        self.create_tables_if_not_exists()
```

### How similarity_search works

```python
def similarity_search_with_score(self, query: str, k: int = 4):
    # 1. Embed the query using the provided embedding function
    embedding = self.embedding_function.embed_query(query)

    # 2. Build the SQL query with pgvector operator
    # Using cosine distance: embedding <=> query_embedding
    results = session.execute(
        select(self.EmbeddingStore)
        .order_by(self.EmbeddingStore.embedding.cosine_distance(embedding))
        .limit(k)
    )

    # 3. Convert DB rows to LangChain Document objects
    return [(Document(page_content=row.document, metadata=row.cmetadata), row.distance)
            for row in results]
```

The `VectorStore` base class in `langchain-core` defines the interface (`similarity_search`, `add_texts`, etc.), and `PGVector` implements it with Postgres-specific SQL.

---

## Layer 4: langchain — High-Level Recipes

This layer provides convenience functions like `create_agent()`.

File: `langchain/agents/__init__.py` → `create_agent()`

```python
def create_agent(llm, tools, **kwargs):
    # 1. Bind tools to the LLM (injects tool schemas into system prompt)
    llm_with_tools = llm.bind_tools(tools)

    # 2. Create a ReAct-style prompt
    prompt = hub.pull("hwchase17/react")

    # 3. Build the agent executor (loop: LLM → parse tool call → execute → repeat)
    agent = create_react_agent(llm_with_tools, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, **kwargs)
```

This is the "black box" your project moved away from. The agent executor is an implicit loop that you can't introspect without reading the source.

---

# Part 2: LangGraph — The State Machine Layer

LangGraph is NOT inside LangChain. It is a separate package (`langgraph`) that builds ON TOP of `langchain-core`. It adds state management, persistence, and graph execution.

File structure:

```
langgraph/
  graph/
    state.py          # StateGraph definition and compilation
    graph.py          # Generic graph base class
  channels/
    base.py           # How state fields are merged (reducers)
    last_value.py     # "replace" semantics
    binop.py          # "add" semantics (for lists)
  checkpoint/
    base.py           # Interface for persisting state
    memory.py         # In-memory checkpoint implementation
    sqlite.py         # SQLite checkpoint persistence
  prebuilt/
    tool_node.py      # Pre-built tool execution node
```

---

## The core abstraction: StateGraph

File: `langgraph/graph/state.py`

```python
class StateGraph:
    def __init__(self, state_schema: Type[Any]):
        self.state_schema = state_schema          # Your AgentState TypedDict
        self.nodes: dict[str, Runnable] = {}      # node_name → function
        self.edges: dict[str, str] = {}           # unconditional edges
        self.conditional_edges: dict[str, Callable] = {}  # routing functions

    def add_node(self, name: str, action: Runnable | Callable):
        self.nodes[name] = coerce_to_runnable(action)

    def add_edge(self, start: str, end: str):
        self.edges[start] = end

    def add_conditional_edges(self, start: str, condition: Callable, mapping: dict):
        self.conditional_edges[start] = (condition, mapping)

    def compile(self) -> CompiledStateGraph:
        # Validates the graph (no dead ends, entry point exists)
        return CompiledStateGraph(self)
```

Your code:

```python
# backend/graph/agent_graph.py
builder = StateGraph(AgentState)
builder.add_node("sanitize_input", sanitize_input)
builder.add_node("classify_intent", classify_intent)
# ...
builder.compile()
```

---

## How compilation works

File: `langgraph/graph/state.py` (class `CompiledStateGraph`)

When you call `.compile()`, LangGraph:

1. **Validates the graph** — ensures every node has an exit path, entry point is set
2. **Wraps each node** in a `RunnableLambda` (so it implements `Runnable.invoke()`)
3. **Builds a Pregel graph** — an internal representation optimized for parallel execution

```python
class CompiledStateGraph:
    def __init__(self, builder: StateGraph):
        self.graph = builder
        # Build channel map from state_schema annotations
        self.channels = self._build_channels(builder.state_schema)
        # Wrap nodes
        self.nodes = {name: coerce_to_runnable(action)
                      for name, action in builder.nodes.items()}

    def invoke(self, input: dict, config=None) -> dict:
        # 1. Initialize state from input
        state = self._initialize_state(input)

        # 2. Execute graph using Pregel engine
        for _super_step in range(self.recursion_limit):
            # Determine which nodes to run (based on previous step's outputs)
            to_run = self._get_next_nodes(state)

            # Run each node in parallel (if no dependencies)
            outputs = {}
            for node_name in to_run:
                node = self.nodes[node_name]
                # Node receives current state, returns partial state updates
                output = node.invoke(state)
                outputs[node_name] = output

            # Merge outputs back into state using channel reducers
            state = self._apply_writes(state, outputs)

            # Check for termination
            if self._is_done(state):
                break

        return state
```

This is the engine. Your `sanitize_input` → `classify_intent` → `route_by_intent` flow is orchestrated by this loop.

---

## Reducers — how state is merged

File: `langgraph/channels/binop.py` (for `Annotated[list, add]`)

Your `AgentState` has:

```python
messages: Annotated[list, add]
```

The `add` operator is a LangGraph reducer. It means: when a node returns `{"messages": [new_message]}`, append to the list instead of replacing it.

```python
# From langgraph/channels/binop.py
class BinaryOperatorAggregate:
    def update(self, values: Sequence):
        # values = list of partial state updates for this field
        result = self.initial_value_factory()
        for value in values:
            result = self.operator(result, value)  # operator = add for lists
        return result
```

Without a reducer (default), a node's return overwrites the field. With `add`, it appends. This is why `messages` accumulates across nodes while `intent` gets replaced.

---

## Conditional edges — how routing works

Your `route_by_intent` function:

```python
def route_by_intent(state: AgentState) -> str:
    if state.get("cached"):
        return "generate_reply"
    intent = state.get("intent", "unknown")
    return intent  # returns a string like "order", "policy", etc.
```

LangGraph uses this return value as a key into the mapping dict:

```python
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {"order": "order_node", "policy": "policy_node", ...}
)
```

Internally, LangGraph compiles this into a Pregel conditional edge. After `classify_intent` runs, the returned string is looked up in the mapping, and only the matching target node is queued for the next step.

---

# Part 3: Tracing a Full Request

Let's trace `POST /api/chat` with query `"What's the return policy?"` through both frameworks.

### Step 1: FastAPI route receives request

```
FastAPI route.py → agent_graph.invoke({"user_input": "What's the return policy?"})
```

### Step 2: LangGraph execution loop

```
CompiledStateGraph.invoke()
  └─→ Step 0: "sanitize_input" node
        └─→ clean_query() → LLM call (via langchain_openai ChatOpenAI)
        └─→ get_cached_response() → PGVector query (via langchain_community)
        Returns: {"cleaned_input": "What is the return policy?", "cached": False}

  └─→ Step 1: "classify_intent" node
        └─→ Keyword matching
        Returns: {"intent": "policy"}

  └─→ Step 2: Conditional edge "classify_intent" → "policy_node"
        (route_by_intent returns "policy", mapped to "policy_node")

  └─→ Step 3: "policy_node" node
        └─→ policy_retriever_tool.invoke()
              └─→ get_policy_retriever().retrieve()
                    └─→ _dense_retrieve() → PGVector.similarity_search_with_score()
                          └─→ OpenAIEmbeddings.embed_query() → Zhipu API
                          └─→ SQL query with cosine distance
                    └─→ _sparse_retrieve() → BM25.get_scores()
                    └─→ _rrf_fuse() → rank-based fusion
                    └─→ _llm_rerank() → ChatOpenAI.invoke() → Zhipu API
                    └─→ Score >= 7 filter
        Returns: {"tool_result": "Return Policy: Our store offers a 30-day..."}

  └─→ Step 4: "generate_reply" node
        └─→ ChatOpenAI.invoke(prompt) → Zhipu API
        Returns: {"final_answer": "Our return policy allows returns within 30 days..."}

  └─→ Step 5: "validate_reply" node
        └─→ ChatOpenAI.invoke(validation_prompt) → Zhipu API
        Returns: {"validation_flag": "valid", "validation_notes": "..."}

  └─→ Step 6: "update_memory" node
        └─→ MemoryStore.add_user() / add_agent() → JSON file write
        └─→ cache_response() → PGVector.add_texts()
```

### Step 3: FastAPI returns response

```
Return: {"answer": "Our return policy...", "cached": false, "latency_ms": 1200}
```

---

# Part 4: Key Design Patterns to Notice

### Pattern 1: Everything is a Runnable

The `Runnable` protocol is the plugin architecture. LLMs, tools, prompts, vector stores, and even your custom node functions are all coerced into `Runnable` before execution. This uniformity is why `prompt | llm | parser` works — each piece speaks the same interface.

### Pattern 2: Callbacks for Observability

Callbacks are injected at every layer (LLM, chain, tool) without the layer knowing. Your `httpx` logs come from a default callback handler in `langchain_openai`. You could add LangSmith, Weights & Biases, or custom logging the same way.

### Pattern 3: State as Immutable-ish Writes

LangGraph nodes don't mutate state in place. They return partial state dicts, and the framework merges them. This makes parallel execution safe — two nodes can run simultaneously without race conditions because they each return their own writes, merged later.

### Pattern 4: Pregel for Parallelism

LangGraph's execution engine is modeled after Google's Pregel graph processing framework. Each "super step" runs all nodes that have no unresolved dependencies. In your graph, everything is sequential, but if you had nodes that branch and then merge (e.g., parallel tool calls), Pregel would run them concurrently.

---

# Part 5: How to Read the Source Code

## Start here

1. **`langchain_core/runnables/base.py`** — Read `Runnable`, `RunnableSequence`, `RunnableLambda`. These are the composition primitives.

2. **`langchain_core/language_models/chat_models.py`** — Read `BaseChatModel`. See how `invoke()` delegates to `_generate()`, and how `_generate()` is what providers override.

3. **`langchain_openai/chat_models/base.py`** — Read `ChatOpenAI`. This is the concrete implementation of `BaseChatModel`.

4. **`langgraph/graph/state.py`** — Read `StateGraph` and `CompiledStateGraph`. This is the orchestration engine.

5. **`langgraph/pregel/__init__.py`** — Read `Pregel.invoke()`. This is the low-level graph execution loop.

## Reading strategy

Don't read top-to-bottom. Pick a specific question and trace it:

- "How does `llm.invoke()` work?" → Start at `BaseChatModel.invoke()` → `ChatOpenAI._generate()` → `openai-python` client
- "How does the pipe operator work?" → `Runnable.__or__()` → `RunnableSequence` constructor → `RunnableSequence.invoke()`
- "How does LangGraph route between nodes?" → `CompiledStateGraph.invoke()` → `Pregel._get_next_nodes()` → conditional edge resolution

---

# Part 6: Interview-Ready Insights

## What to say about LangChain's architecture

> "LangChain is built on a `Runnable` protocol where every component — LLM, tool, prompt — implements the same `invoke()`/`stream()` interface. This lets you compose them with the `|` operator into `RunnableSequence`s. Underneath, callbacks are injected at every layer for observability without modifying behavior."

## What to say about LangGraph's value over LangChain

> "LangChain's `create_agent()` gives you a black-box ReAct loop. LangGraph makes the control flow explicit — you define nodes as pure functions and edges as state transitions. This lets you insert checkpoints, validation, and caching at specific points in the flow, and it makes debugging deterministic because you can inspect state between every node."

## What to say about the execution engine

> "LangGraph compiles the graph into a Pregel-style execution engine. Each super-step runs all nodes whose dependencies are satisfied, merges their outputs back into state using reducers (like `add` for lists), and resolves conditional edges to determine the next step. This makes parallel node execution safe because writes are isolated until the merge phase."
