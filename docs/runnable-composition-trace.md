# Runnable Composition — `|` Operator Trace

This traces the composition engine that makes `prompt | llm | parser` work.

---

## The `|` Operator in One Line

```python
# langchain_core/runnables/base.py:622
class Runnable:
    def __or__(self, other):
        return RunnableSequence(self, coerce_to_runnable(other))
```

That's it. Everything else is what `RunnableSequence.invoke()` and `coerce_to_runnable()` do.

---

# Part 1: `coerce_to_runnable()` — The Universal Adapter

**File:** `langchain_core/runnables/base.py` (function distributed through the module)

This function converts **anything** into a `Runnable`:

```python
def coerce_to_runnable(thing: RunnableLike) -> Runnable:
    if isinstance(thing, Runnable):
        return thing                    # Already a Runnable — pass through
    if callable(thing):
        return RunnableLambda(thing)    # Function → lambda wrapper
    if isinstance(thing, dict):
        return RunnableParallel(thing)  # Dict → parallel execution
    raise TypeError(...)
```

**The 3 transformations:**

| Input | Transformed to | Example |
|-------|---------------|---------|
| `Runnable` | Passed through | `ChatOpenAI()` stays `ChatOpenAI()` |
| `Callable` (function) | `RunnableLambda(func)` | `lambda x: x + 1` becomes a runnable |
| `dict` | `RunnableParallel(dict)` | `{"a": fn1, "b": fn2}` becomes parallel |

**Why this matters:** It means you can freely mix functions, dicts, and runnables with `|`:

```python
chain = (
    prompt_template          # Runnable
    | llm                    # Runnable
    | (lambda x: x.content)  # Callable → RunnableLambda
    | {"key": parser}        # dict → RunnableParallel
)
```

Every piece gets coerced to `Runnable`, so they all speak the same `invoke`/`stream` interface.

---

# Part 2: `RunnableSequence` — The Sequential Pipe

**File:** `langchain_core/runnables/base.py:2861`

## Constructor: how sequences flatten

```python
class RunnableSequence:
    def __init__(self, *steps):
        steps_flat = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)  # ← FLATTEN nested sequences
            else:
                steps_flat.append(coerce_to_runnable(step))
        self.first = steps_flat[0]
        self.middle = steps_flat[1:-1]
        self.last = steps_flat[-1]
```

**Flattening:** `(A | B) | C` doesn't create `RunnableSequence(RunnableSequence(A,B), C)`. It creates `RunnableSequence(A, B, C)`. This prevents deep nesting.

**The three-part structure (first, middle, last) is for type checking only.** `first.InputType` becomes the sequence's `InputType`, `last.OutputType` becomes the sequence's `OutputType`.

## invoke() — the core loop

```python
def invoke(self, input, config=None, **kwargs):
    # 1. Set up callbacks
    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
    run_manager = callback_manager.on_chain_start(None, input, ...)

    # 2. Pipe input through each step
    input_ = input
    try:
        for i, step in enumerate(self.steps):
            # Create child config for this step
            config = patch_config(config, callbacks=run_manager.get_child(f"seq:step:{i+1}"))

            if i == 0:
                input_ = step.invoke(input_, config, **kwargs)  # first step gets kwargs
            else:
                input_ = step.invoke(input_, config)            # subsequent steps don't
    except BaseException as e:
        run_manager.on_chain_error(e)
        raise
    else:
        run_manager.on_chain_end(input_)
        return input_
```

**Key design decisions:**

1. **Only the first step gets `**kwargs`.** This prevents accidental parameter leakage. If `llm.invoke()` accepts `temperature`, the prompt template step shouldn't receive it.

2. **Each step gets a child callback manager.** This creates a trace tree: `chain → seq:step:1 (prompt) → seq:step:2 (llm)`.

3. **RunnableSequence with nested sequences auto-flattens.** `RunnableSequence.__or__` checks for `RunnableSequence` on the right side and merges:

```python
def __or__(self, other):
    if isinstance(other, RunnableSequence):
        return RunnableSequence(self.first, *self.middle, self.last,
                                other.first, *other.middle, other.last)
    return RunnableSequence(self.first, *self.middle, self.last,
                            coerce_to_runnable(other))
```

---

# Part 3: `RunnableParallel` — Concurrent Execution

**File:** `langchain_core/runnables/base.py:3609`

## invoke() — ThreadPool for concurrent steps

```python
def invoke(self, input, config=None, **kwargs):
    config = ensure_config(config)
    run_manager = callback_manager.on_chain_start(...)

    def _invoke_step(step, input_, config, key):
        child_config = patch_config(config, callbacks=run_manager.get_child(f"map:key:{key}"))
        return step.invoke(input_, child_config)

    # Execute ALL steps concurrently via ThreadPoolExecutor
    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(_invoke_step, step, input, config, key)
            for key, step in self.steps__.items()
        ]
        output = {key: future.result() for key, future in zip(steps, futures)}

    run_manager.on_chain_end(output)
    return output
```

**Key design:**
- Every step gets the **same input** (no piping between steps)
- Steps execute **concurrently** via `ThreadPoolExecutor`
- Results are collected into a **dict keyed by step name**
- Each step gets its own **child callback** manager

## The dict shorthand

```python
chain = prompt | {"joke": joke_chain, "poem": poem_chain}
# The dict is coerced to RunnableParallel via coerce_to_runnable()
```

When `coerce_to_runnable()` sees a dict, it creates `RunnableParallel(steps__=dict)`. Each value in the dict is independently coerced to a `Runnable`.

---

# Part 4: `RunnableLambda` — Wrapping Functions

**File:** `langchain_core/runnables/base.py:4443`

```python
class RunnableLambda(Runnable[Input, Output]):
    def __init__(self, func):
        self.func = func

    def _invoke(self, input, config, **kwargs):
        # Check if func accepts config parameter
        if self._accepts_config:
            return self.func(input, config=config, **kwargs)
        # Check if func accepts kwargs
        if self._accepts_run_manager:
            kwargs["run_manager"] = ...
        return self.func(input, **kwargs)
```

`RunnableLambda` is the bridge between pure functions and the `Runnable` interface. It introspects the function's signature to determine what to inject:
- If the function has a `config` parameter → pass it
- If the function has `**kwargs` → pass all kwargs

---

# Part 5: `RunnableBinding` — Freezing Arguments

**File:** `langchain_core/runnables/base.py:6013`

`RunnableBinding` wraps a `Runnable` with baked-in arguments:

```python
class RunnableBinding(RunnableBindingBase):
    bound: Runnable          # The underlying runnable
    kwargs: dict[str, Any]   # Frozen kwargs to always pass
    config: RunnableConfig   # Merged config

    def _invoke(self, input, config, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}  # baked kwargs + invocation kwargs
        return self.bound.invoke(input, config, **merged_kwargs)
```

This is what `llm.bind_tools([...])` uses. It creates `RunnableBinding(bound=ChatOpenAI(), kwargs={"tools": [...]})`. Every call to this bound version auto-injects the tools.

---

# Part 6: Complete Composition Trace

```
prompt | llm | output_parser

What actually happens:

1. prompt.__or__(llm)
   → RunnableSequence(prompt, llm)     # first=prompt, last=llm

2. sequence.__or__(output_parser)       # sequence = RunnableSequence(prompt, llm)
   → other = coerce_to_runnable(output_parser)
   → other is not RunnableSequence
   → RunnableSequence(prompt, llm, output_parser)  # flattened

3. chain.invoke({"topic": "cats"})
   → callback_manager.on_chain_start(...)
   → step[0]: prompt.invoke({"topic": "cats"}, config)
        → ChatPromptValue(messages=[SystemMessage(...), HumanMessage("tell me about cats")])
   → step[1]: llm.invoke(ChatPromptValue, config)
        → AIMessage(content="Cats are...")
   → step[2]: output_parser.invoke(AIMessage, config)
        → "Cats are..."
   → callback_manager.on_chain_end("Cats are...")
   → return "Cats are..."
```

---

# Part 7: `RunnableConfig` — the Propagation Mechanism

Every `invoke()` accepts a `config` argument. It propagates automatically through `RunnableSequence.invoke()`:

```python
config = RunnableConfig(
    callbacks=[my_handler],
    metadata={"request_id": "abc"},
    tags=["prod"],
    configurable={"thread_id": "t1"},  # LangGraph uses this
)

chain.invoke(input, config=config)
```

Inside the chain, every step receives the same `config` (with child patches for tracing). This is how:
- **Callbacks** work — they're in `config["callbacks"]`
- **Checkpointing** works — `thread_id` is in `config["configurable"]`
- **Streaming** works — `CONFIG_KEY_STREAM` is in the configurable dict
- **Tracing** works — `metadata` and `tags` propagate everywhere

---

# Interview Talking Points

## "What does `prompt | llm` actually do?"

> "The `|` operator calls `self.__or__(other)` which creates a `RunnableSequence`. `coerce_to_runnable()` converts the right side — if it's a function, it becomes `RunnableLambda`; if it's a dict, `RunnableParallel`. Sequences auto-flatten so chaining multiple `|` operators doesn't create nested wrappers."

## "How does RunnableParallel execute steps concurrently?"

> "It wraps each step in a partial function and submits them all to a `ThreadPoolExecutor`. Results are collected into a dict keyed by step name. Every step receives the exact same input — unlike `RunnableSequence` where output pipes to the next step."

## "How does config propagate through a chain?"

> "`RunnableSequence.invoke()` creates a root `run_manager`, then patches the config with child run managers for each step. The `CONF` dict in the config carries LangGraph's checkpoint configuration (thread_id, checkpoint_ns, etc.) through the entire execution graph."
