# Pregel Execution Engine — How LangGraph Runs Your Graph

LangGraph compiles your `StateGraph` into a Pregel-style execution engine. This document traces every step from `graph.invoke()` to `END`.

---

## The Big Picture

```
graph.invoke({"user_input": "hello"})
    │
    ▼
Pregel.stream()                     [pregel/main.py]
    │ creates
    ▼
SyncPregelLoop                      [pregel/_loop.py]
    │
    ├── __enter__()                  # Load checkpoint, apply input
    │
    ├── LOOP:
    │   ├── tick()                   # Prepare tasks (which nodes to run)
    │   ├── [execute tasks]          # Run node functions
    │   └── after_tick()             # Apply writes, save checkpoint
    │
    └── __exit__()                   # Final checkpoint, output
```

---

# Part 1: Pregel.stream() — The Entry Point

**File:** `langgraph/pregel/main.py:2491`

```python
class Pregel(Runnable):
    def stream(self, input, config=None, *, stream_mode="values", ...):
        # 1. Prepare config
        config = ensure_config(self.config, config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(...)

        # 2. Create the execution loop
        loop = SyncPregelLoop(
            input,
            stream=StreamProtocol(stream.put, stream_modes),
            config=config,
            checkpointer=self.checkpointer,
            nodes=self.nodes,
            specs=self.channels,
            ...
        )

        # 3. Run the loop
        with loop as loop_:
            # Phase 1: Load/resume
            # Phase 2: Loop
            while True:
                if not loop_.tick():        # Prepare tasks
                    break                    # No more tasks → done
                loop_.execute_tasks()        # Run nodes
                loop_.after_tick()           # Apply writes

        # 4. Return output
        return loop_.output
```

---

# Part 2: `nodes` — How Nodes Are Wrapped

**File:** `langgraph/pregel/_read.py`

When you call `builder.add_node("my_node", my_function)`, LangGraph wraps your function in a `PregelNode`:

```python
class PregelNode:
    bound: Runnable          # Your function wrapped as a Runnable
    triggers: list[str]      # Which channels trigger this node
    input_schema: ...        # What input this node expects
    output_schema: ...       # What output this node produces

    def __init__(self, bound, triggers=None, ...):
        self.bound = coerce_to_runnable(bound)  # ← your function becomes a Runnable
        self.triggers = triggers or []
```

**Your code:**
```python
builder.add_node("sanitize_input", sanitize_input)
```

**Becomes:**
```python
PregelNode(
    bound=RunnableLambda(sanitize_input),
    triggers=["user_input"],  # This node runs when "user_input" channel updates
)
```

---

# Part 3: `channels` — How State Becomes Versioned

**File:** `langgraph/channels/`

Each field in your `AgentState` becomes a `Channel` object:

| AgentState field | Channel type | Behavior |
|-----------------|-------------|----------|
| `user_input: str` | `LastValue` | Replace on write |
| `intent: str` | `LastValue` | Replace on write |
| `messages: Annotated[list, add]` | `BinaryOperatorAggregate` | Append on write |

```python
class LastValue:
    """Channel that keeps only the latest value."""
    def update(self, values):
        return values[-1]  # Last writer wins

class BinaryOperatorAggregate:
    """Channel that combines values with an operator (e.g., list add)."""
    def update(self, values):
        result = []
        for v in values:
            result = self.operator(result, v)  # add for lists
        return result
```

---

# Part 4: The Pregel Loop — Step by Step

## Phase 1: `__enter__` — Initialization

```
1. Load checkpoint (or create empty one)
2. Restore channels from checkpoint values
3. Map input to channel writes → creates "input" checkpoint
4. Set step = checkpoint_metadata["step"] + 1
5. Set stop = step + recursion_limit + 1
```

## Phase 2: `tick()` — Which Nodes to Run

```python
def tick(self):
    # Check iteration limit
    if self.step > self.stop:
        self.status = "out_of_steps"
        return False

    # Prepare next tasks
    self.tasks = prepare_next_tasks(
        self.checkpoint,
        self.checkpoint_pending_writes,
        self.nodes,
        self.channels,
        self.managed,
        ...
    )

    # No tasks → done
    if not self.tasks:
        self.status = "done"
        return False

    # Check interrupts
    if self.interrupt_before and should_interrupt(...):
        raise GraphInterrupt()

    return True
```

### How `prepare_next_tasks` decides which nodes to run

**File:** `langgraph/pregel/_algo.py`

```python
def prepare_next_tasks(checkpoint, pending_writes, nodes, channels, ...):
    tasks = {}

    for node_name, node in nodes.items():
        # 1. Check if any of this node's triggers have been updated
        triggers = node.triggers
        if triggers and not any(t in updated_channels for t in triggers):
            continue  # ← skip: nothing this node cares about has changed

        # 2. Build the task
        task = PregelExecutableTask(
            id=str(uuid6()),
            name=node_name,
            path=(node_name,),
            writes=[],
            input=read_channels(channels, node.input_keys),
            ...
        )
        tasks[task.id] = task

    return tasks
```

**Key insight:** Nodes run when their trigger channels have new versions. Your `sanitize_input` node has `triggers=["user_input"]`. When you pass `{"user_input": "hello"}` as input, the `user_input` channel gets a new version → `sanitize_input` is triggered.

## Phase 3: After `tick()` — Execute Tasks

Tasks are executed by the caller (in `main.py`):

```python
# From Pregel.stream()
while True:
    if not loop_.tick():
        break
    loop_.execute_tasks()     # ← runs each task's node function
    loop_.after_tick()
```

Each task's `bound` (your node function) is called with the current state:

```python
# sanitize_input receives current state
node_output = node.bound.invoke(state)
# Returns {"cleaned_input": "...", "cached": False}
# → This is stored as task.writes = [("cleaned_input", "..."), ("cached", False)]
```

## Phase 4: `after_tick()` — Merge and Save

```python
def after_tick(self):
    # 1. Collect all writes from all tasks
    writes = [w for t in self.tasks.values() for w in t.writes]

    # 2. Apply writes to channels
    self.updated_channels = apply_writes(
        self.checkpoint,
        self.channels,
        self.tasks.values(),
        self.checkpointer_get_next_version,
        ...
    )

    # 3. Save checkpoint
    self._put_checkpoint({"source": "loop"})

    # 4. Clear pending writes
    self.checkpoint_pending_writes.clear()

    # 5. Check interrupt_after
    if self.interrupt_after and should_interrupt(...):
        raise GraphInterrupt()
```

### `apply_writes()` — the channel update mechanism

```python
def apply_writes(checkpoint, channels, tasks, get_next_version, ...):
    updated_channels = set()

    for task in tasks:
        for channel_name, value in task.writes:
            # Get next version
            new_version = get_next_version(
                checkpoint["channel_versions"].get(channel_name),
                channel_name
            )

            # Update the channel
            channel = channels[channel_name]
            channel.update([value])

            # Record the new version
            checkpoint["channel_versions"][channel_name] = new_version
            updated_channels.add(channel_name)

    return updated_channels
```

---

# Part 5: Conditional Edges

Your `route_by_intent` function:

```python
def route_by_intent(state):
    if state.get("cached"):
        return "generate_reply"
    return state.get("intent", "unknown")  # "order", "policy", "weather", etc.
```

After `classify_intent` runs, `prepare_next_tasks` invokes the conditional edge:

```python
# From the conditional edge logic:
next_node_name = route_by_intent(state)
# next_node_name = "order", "policy", "weather", or "generate_reply"

# Only the matched node is queued for the next step
tasks[next_node_name] = PregelExecutableTask(...)
```

---

# Part 6: Your Graph, Step by Step

```
INVOKE: graph.invoke({"user_input": "What's the return policy?"})

STEP -1: __enter__()
  ├── empty_checkpoint() → channel_values={}, versions={}
  ├── _first() applies input:
  │   ├── user_input channel: version 1, value "What's the return policy?"
  │   └── _put_checkpoint(source="input", step=-1)
  └── step = 0, stop = 25

STEP 0: tick()
  ├── prepare_next_tasks()
  │   └── "sanitize_input": triggers=["user_input"], user_input channel updated → RUN
  └── tasks = {"t1": sanitize_input}

EXECUTE: task "t1" → sanitize_input(state)
  └── Returns {"cleaned_input": "What is the return policy?", "cached": False}

after_tick():
  ├── apply_writes()
  │   ├── clean_input channel: version 1, value "What is the return policy?"
  │   └── cached channel: version 1, value False
  ├── _put_checkpoint(source="loop", step=0)
  └── step = 1

STEP 1: tick()
  ├── prepare_next_tasks()
  │   └── "classify_intent": triggers=["cleaned_input"] → RUN
  └── tasks = {"t2": classify_intent}

EXECUTE: task "t2" → classify_intent(state)
  └── Returns {"intent": "policy"}

after_tick():
  ├── apply_writes()
  │   └── intent channel: version 1, value "policy"
  ├── _put_checkpoint(source="loop", step=1)
  └── step = 2

STEP 2: tick()
  ├── prepare_next_tasks()
  │   ├── "policy_node": triggers=["intent"], intent channel updated → RUN
  │   │   (route_by_intent returned "policy")
  │   └── "order_node", "weather_node", "generate_reply": skipped
  └── tasks = {"t3": policy_node}

EXECUTE: task "t3" → policy_node(state)
  └── Calls policy_retriever_tool → returns "Return Policy: Our store offers..."
  └── Returns {"tool_result": "Return Policy: Our store offers a 30-day..."}

after_tick():
  ├── apply_writes() → tool_result channel updated
  ├── _put_checkpoint(source="loop", step=2)
  └── step = 3

... (continues: generate_reply, validate_reply, update_memory) ...

FINAL STEP: tick()
  ├── prepare_next_tasks() → {} (all nodes have run, no triggers updated)
  └── status = "done", return False

__exit__():
  ├── Final checkpoint saved
  └── Output = read_channels(channels, output_keys)
```

---

# Part 7: Key Design Decisions

## 1. Version-based triggering

Nodes aren't queued by name. They're queued when their **trigger channels** have new versions. This is why after `sanitize_input` writes to `cleaned_input`, only `classify_intent` (which triggers on `cleaned_input`) runs — not `generate_reply` (which doesn't trigger on that channel).

## 2. Parallel-safe writes

Each node returns a dict. The framework merges them through channels. If 3 nodes all return `{"messages": [msg]}`, the `BinaryOperatorAggregate` channel concatenates all three lists safely.

## 3. Pregel heritage

The execution model is inspired by Google's Pregel graph processing framework. A "super-step" = tick → execute → after_tick. Each super-step processes all ready nodes in parallel, then merges results.

## 4. Recursion limit

```python
self.stop = self.step + self.config["recursion_limit"] + 1
```

Default recursion_limit is 25. This prevents infinite tool-calling loops. If the graph exceeds 25 steps, it stops with `status="out_of_steps"`.

---

# Interview Talking Points

## "How does LangGraph decide which nodes to run?"

> "Each node has trigger channels — typically the state fields it reads from the previous node's output. In `prepare_next_tasks()`, LangGraph checks which channels have been updated this super-step and only queues nodes whose triggers include those updated channels. This is version-based, not name-based."

## "How does parallel execution work?"

> "If multiple nodes are triggered by the same channel update, they're all queued in the same `tick()`. The Pregel executor runs them in parallel via `ThreadPoolExecutor`. Writes from parallel nodes are merged through channel reducers — `LastValue` takes the last write, `BinaryOperatorAggregate` concatenates."

## "How does the recursion limit work?"

> "After loading a checkpoint, `stop = step + recursion_limit + 1`. Each `tick()` checks `step > stop`. If a tool-calling agent loops too many times, the graph stops with `out_of_steps` instead of infinite-looping."
