# How LangChain and LangGraph Manage Context

Context means different things at different layers: conversation history, system instructions, tool results, state accumulated across a multi-step process, and cross-cutting metadata like request IDs. The two frameworks handle these differently.

---

# Part 1: LangChain — Context as Data Flow

LangChain's philosophy: **context is what flows through the pipe.** Each `Runnable` receives input, transforms it, and passes output to the next `Runnable`. There is no global state — only the data currently in flight.

## The three kinds of context in LangChain

### 1. Input/output context (the data being processed)

```python
# A simple chain
chain = prompt_template | llm | output_parser

result = chain.invoke({"product": "laptop"})
```

What flows through:
```
{"product": "laptop"}
    ↓ prompt_template
ChatPromptValue(messages=[SystemMessage("..."), HumanMessage("Tell me about laptop")])
    ↓ llm
AIMessage(content="A laptop is a portable computer...")
    ↓ output_parser
"A laptop is a portable computer..."
```

Each step receives the output of the previous step. No step knows what happened before its input arrived. This is functional composition — pure data flow.

### 2. Configuration context (how to run, not what to run)

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    callbacks=[my_tracer],
    metadata={"request_id": "abc-123"},
    tags=["production"],
)

llm.invoke("hello", config=config)
```

`RunnableConfig` is a **side channel**. It doesn't change the input data, but it affects how the runnable behaves:
- `callbacks`: who gets notified of events
- `metadata`: key-value pairs for observability
- `tags`: filtering/grouping for analytics
- `run_name`: human-readable identifier
- `recursion_limit`: safety limit for nested calls

**How it propagates:**

```python
# Inside RunnableSequence.invoke()
for step in self.steps:
    input = step.invoke(input, config)  # ← same config passed to every step
```

Every runnable in the chain receives the same `config`. This is how LangChain implements cross-cutting concerns without global variables.

### 3. Memory context (conversation history)

LangChain's base chat model is **stateless**. It doesn't remember anything between calls:

```python
llm.invoke("My name is Alice")   # LLM responds
llm.invoke("What's my name?")    # LLM has no idea — no history was passed
```

To add memory, you must explicitly pass history as part of the input:

```python
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage("My name is Alice"),
    AIMessage("Nice to meet you, Alice!"),
    HumanMessage("What's my name?"),
]

llm.invoke(messages)  # Now the LLM sees the full conversation
```

This is the fundamental contract: **the LLM only sees what you put in the messages list.**

## How `RunnableWithMessageHistory` adds memory

File: `langchain_core/runnables/history.py`

This is a wrapper that auto-injects conversation history into the prompt:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,  # function: session_id → MessageHistory
    input_messages_key="question",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"question": "What's my name?"},
    config={"configurable": {"session_id": "user-123"}}
)
```

**What happens inside:**

```python
# Pseudo-code from RunnableWithMessageHistory.invoke()
def invoke(self, input, config):
    session_id = config["configurable"]["session_id"]
    history = self.get_session_history(session_id)  # ← load from DB/memory

    # 1. Add the new user message to history
    history.add_user_message(input["question"])

    # 2. Build the full prompt: system + history + current question
    full_input = {
        "question": input["question"],
        "history": history.messages,
    }

    # 3. Run the chain
    response = self.chain.invoke(full_input)

    # 4. Save the response to history
    history.add_ai_message(response)

    return response
```

**Key design:** History is external to the chain. The chain itself is still stateless — `RunnableWithMessageHistory` is a wrapper that loads, injects, and saves history around each invocation.

## How `ChatPromptTemplate` assembles context

File: `langchain_core/prompts/chat.py`

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Current user: {user_name}"),
    ("placeholder", "{history}"),  # ← messages get inserted here
    ("human", "{question}"),
])
```

When you call `prompt.invoke({"user_name": "Alice", "history": [...], "question": "..."})`:

```python
def invoke(self, input):
    # 1. Format each message template
    messages = []
    for message_template in self.messages:
        if message_template.type == "placeholder":
            messages.extend(input["history"])  # ← inject history messages
        else:
            messages.append(message_template.format(**input))

    # 2. Return as ChatPromptValue
    return ChatPromptValue(messages=messages)
```

The prompt template is a **blueprint** for assembling context. It defines where system instructions go, where history goes, and where the current query goes.

---

# Part 2: LangGraph — Context as Mutable State

LangGraph's philosophy: **context is a shared state object that every node reads and writes.** Unlike LangChain's pipe, where data flows sequentially and each step only sees the previous step's output, LangGraph nodes see the accumulated state of the entire graph execution so far.

## The StateGraph contract

```python
class AgentState(TypedDict):
    messages: Annotated[list, add]
    user_input: str
    intent: str
    tool_result: str
    final_answer: str
    cached: bool
```

Every node function signature:

```python
def my_node(state: AgentState) -> AgentState:
    # Read from state
    current_intent = state.get("intent", "unknown")

    # Compute something
    result = do_work(current_intent)

    # Write back partial update
    return {"tool_result": result, "intent": "processed"}
```

**Key difference from LangChain:** Nodes return **partial state updates**, not the full output. LangGraph merges the returned dict back into the shared state.

## How state merging works

### Default behavior: replace

If a field has no reducer annotation, returning it **overwrites** the previous value:

```python
class AgentState(TypedDict):
    intent: str  # no reducer

def classify_intent(state):
    return {"intent": "policy"}  # ← replaces whatever was there before
```

### With reducers: combine

If a field has a reducer, the returned value is **combined** with the existing value:

```python
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # reducer = add

def add_user_message(state):
    return {"messages": [HumanMessage("hello")]}  # ← APPENDS to existing list
```

**How it works under the hood:**

File: `langgraph/channels/binop.py`

```python
class BinaryOperatorAggregate:
    def update(self, values):
        result = self.initial_value_factory()  # start with empty list
        for value in values:
            result = self.operator(result, value)  # operator = add for lists
        return result
```

When multiple nodes run in parallel and each returns `{"messages": [new_msg]}`, the reducer concatenates all of them into a single list.

## State is scoped per graph invocation

```python
# Invocation 1
result1 = agent_graph.invoke({"user_input": "What's the return policy?"})

# Invocation 2 — starts fresh, no memory of invocation 1
result2 = agent_graph.invoke({"user_input": "How much is shipping?"})
```

Unlike `RunnableWithMessageHistory`, LangGraph doesn't automatically load history. **You must pass history in the initial state:**

```python
agent_graph.invoke({
    "user_input": "How much is shipping?",
    "messages": result1["messages"],  # ← carry forward from previous invocation
})
```

Or use **checkpoints** to persist state across invocations.

## Checkpoints: persistent state between invocations

File: `langgraph/checkpoint/base.py`

LangGraph can save the full state after each super-step to a checkpoint store:

```python
from langgraph.checkpoint.memory import MemorySaver

# Wrap the compiled graph with checkpointing
graph = agent_graph.compile(checkpointer=MemorySaver())

# Invoke with a thread_id
config = {"configurable": {"thread_id": "conversation-123"}}
result = graph.invoke({"user_input": "hello"}, config=config)

# Later, resume from the same thread
result2 = graph.invoke({"user_input": "follow-up"}, config=config)
# ← starts from the checkpointed state, not from empty state
```

**How it works:**

```python
# Pseudo-code from CompiledStateGraph.invoke()
def invoke(self, input, config=None):
    thread_id = config.get("configurable", {}).get("thread_id")

    # 1. Load checkpoint if resuming
    if thread_id and self.checkpointer:
        checkpoint = self.checkpointer.get(thread_id)
        if checkpoint:
            state = checkpoint.state  # ← resume from here
        else:
            state = self._initialize_state(input)
            self.checkpointer.put(thread_id, state)  # ← save initial state
    else:
        state = self._initialize_state(input)

    # 2. Run the graph
    for step in self._get_steps(state):
        state = self._run_step(step, state)
        if self.checkpointer:
            self.checkpointer.put(thread_id, state)  # ← save after each step

    return state
```

This is how LangGraph implements **conversation memory** without you manually passing history around. The checkpoint stores the full state, and you resume by providing the same `thread_id`.

---

# Part 3: Side-by-Side Comparison

| Aspect | LangChain (chains) | LangGraph (graphs) |
|--------|-------------------|-------------------|
| **Mental model** | Data flows through a pipe | Nodes read/write shared state |
| **What a step sees** | Only the previous step's output | The accumulated state of all previous steps |
| **Memory pattern** | External — `RunnableWithMessageHistory` wraps the chain | Built-in — checkpoints persist state per thread |
| **Parallelism** | Sequential by design | Nodes can run in parallel, writes merged by reducers |
| **Debugging** | Hard — intermediate state is invisible | Easy — inspect state after every node |
| **Configuration** | `RunnableConfig` passed through every step | `RunnableConfig` + graph config (thread_id, etc.) |

## Example: same conversation, two approaches

### LangChain approach

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

llm = ChatOpenAI()
chain = prompt | llm

# Add memory wrapper
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: InMemoryChatMessageHistory(),
    input_messages_key="question",
    history_messages_key="history",
)

# Call 1
chain_with_memory.invoke(
    {"question": "My name is Alice"},
    config={"configurable": {"session_id": "abc"}}
)

# Call 2 — memory is loaded automatically by the wrapper
chain_with_memory.invoke(
    {"question": "What's my name?"},
    config={"configurable": {"session_id": "abc"}}
)
```

**How context flows:**
- The wrapper loads history from `InMemoryChatMessageHistory` for session "abc"
- Injects it into the prompt under `{history}`
- Runs the chain
- Saves the new response back to history

### LangGraph approach

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]
    user_input: str

def chat_node(state: State):
    # The LLM sees ALL messages in state["messages"]
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", "__end__")

# Compile with checkpointing
graph = builder.compile(checkpointer=MemorySaver())

# Call 1
graph.invoke(
    {"user_input": "My name is Alice", "messages": [HumanMessage("My name is Alice")]},
    config={"configurable": {"thread_id": "abc"}}
)

# Call 2 — resumes from checkpoint, state["messages"] already contains history
graph.invoke(
    {"user_input": "What's my name?", "messages": [HumanMessage("What's my name?")]},
    config={"configurable": {"thread_id": "abc"}}
)
```

**How context flows:**
- Call 1: initial state has 1 message → node runs → state now has 2 messages → checkpoint saves
- Call 2: loads checkpoint → state already has 2 messages → new message appended → node runs → state now has 4 messages

---

# Part 4: Callbacks — The Hidden Context Layer

Both frameworks share a callback system for cross-cutting concerns. Callbacks carry **execution context** (not business logic context):

```python
from langchain_core.callbacks import BaseCallbackHandler

class RequestIdInjector(BaseCallbackHandler):
    def __init__(self, request_id: str):
        self.request_id = request_id

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Access metadata passed via RunnableConfig
        print(f"[{self.request_id}] LLM call started")

    def on_llm_end(self, response, **kwargs):
        print(f"[{self.request_id}] LLM call finished")
```

The callback handler receives the same `RunnableConfig` that was passed to `invoke()`, so it can access `metadata`, `tags`, etc. This is how you propagate request IDs, user IDs, or trace context through the entire pipeline without modifying business logic.

## Your project's context management

Looking at your actual code:

```python
# backend/graph/nodes.py — LangGraph state
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add]        # ← reducer: accumulates across nodes
    user_input: str                       # ← no reducer: overwritten each step
    intent: str
    tool_result: str
    final_answer: str
    cached: bool
    validation_flag: str
    validation_notes: str
```

**What works well:**
- `messages` uses `add` reducer — accumulates conversation history
- Each node has a single responsibility — `sanitize_input`, `classify_intent`, etc.
- State is typed — catches missing fields at development time

**What's missing:**
- No `session_id` in state — you can't resume conversations
- `memory_store` is global JSON file — not scoped to sessions
- No checkpointing — every invocation starts from scratch

---

# Part 5: Interview Talking Points

## "How does LangChain manage conversation context?"

> "LangChain's core chat models are stateless — they only see the messages you pass in a single call. To add memory, you either manually maintain a messages list and pass it each time, or use `RunnableWithMessageHistory` which wraps your chain and auto-loads/injects/saves history from an external store like Redis or PostgreSQL. The chain itself never holds state — memory is always external."

## "How does LangGraph differ?"

> "LangGraph uses a shared state object that every node reads and writes. Nodes return partial state updates, and the framework merges them using reducers — like `add` for lists that should accumulate. For persistence, you attach a `Checkpointer` which saves the full state after each step. Resuming a conversation is just invoking the graph with the same `thread_id` — the checkpointer loads the previous state automatically."

## "When would you use one vs the other?"

> "For simple Q&A chains, LangChain's pipe model is cleaner — less boilerplate. For multi-step agents with tool use, validation, and conditional routing, LangGraph's explicit state management is essential. You can't insert a validation step in the middle of a LangChain pipe without rewriting the whole chain. In LangGraph, you just add a node and an edge."
