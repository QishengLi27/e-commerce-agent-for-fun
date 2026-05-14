# Tool Calling — End-to-End Source Code Trace

This traces tool calling from your `@tool` decorator through the LLM, back through the response parser, and into ToolNode execution.

---

## The Full Chain (7 Stages)

```
Stage 1: @tool decorator      → StructuredTool created, args_schema auto-generated
Stage 2: bind_tools()          → Tool converted to OpenAI JSON Schema, stored in kwargs
Stage 3: ChatOpenAI._generate() → kwargs["tools"] injected into HTTP request payload
Stage 4: LLM responds           → finish_reason="tool_calls", response.choices[0].message.tool_calls
Stage 5: _create_chat_result()  → _convert_dict_to_message() parses tool_calls into AIMessage
Stage 6: ToolNode._func()       → Extracts tool_calls from AIMessage, executes each tool
Stage 7: ToolMessage returned   → Tool output wrapped, appended to messages via reducer
```

---

# Stage 1: The `@tool` Decorator

**File:** `langchain_core/tools/convert.py:76`

### Your code

```python
@tool
def order_status_tool(order_id: str) -> str:
    """Look up the status of a specific order by order ID."""
    return get_order_status(order_id)
```

### What happens

The `@tool` decorator does:

```python
def tool(name_or_callable=None, *, description=None, ...):
    # 1. If called with arguments (@tool(...))
    #    → returns _create_tool_factory(tool_name)
    #    which is a SECOND decorator that wraps the function

    # 2. If called without arguments (@tool)
    #    → name_or_callable = the function itself
    #    → extracts function name, docstring, type hints
    #    → calls StructuredTool.from_function(func, ...)
```

For your case `@tool` (no arguments):

```python
# Inside _create_tool_factory → _tool_factory
def _tool_factory(dec_func):
    if not inspect.iscoroutinefunction(dec_func):
        coroutine = None
        func = dec_func         # ← your order_status_tool function
        schema = args_schema    # ← None (auto-inferred)

    if infer_schema:
        return StructuredTool.from_function(
            func,               # the actual function
            coroutine,
            name="order_status_tool",     # from func.__name__
            description=None,             # from func.__doc__
            return_direct=False,
            args_schema=None,             # auto-inferred from type hints
            infer_schema=True,
            ...
        )
```

### How StructuredTool.from_function() auto-generates the schema

**File:** `langchain_core/tools/structured.py`

```python
class StructuredTool(BaseTool):
    @classmethod
    def from_function(cls, func, coroutine=None, name=None, description=None,
                      args_schema=None, infer_schema=True, ...):
        if infer_schema and args_schema is None:
            # Auto-generates a Pydantic model from function signature
            args_schema = create_schema_from_function(name, func)

        return cls(
            name=name,
            description=description or func.__doc__,
            args_schema=args_schema,
            func=func,
            coroutine=coroutine,
            ...
        )
```

`create_schema_from_function()` uses `pydantic.create_model()` to build:

```python
# For order_status_tool(order_id: str) -> str:
# Auto-generated Pydantic model:
class order_status_toolSchema(BaseModel):
    order_id: str = Field(description="order_id")
```

This schema becomes the JSON Schema sent to the LLM:

```json
{
    "name": "order_status_tool",
    "description": "Look up the status of a specific order by order ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "order_id"}
        },
        "required": ["order_id"]
    }
}
```

### BaseTool.invoke() — the entry point for tool calls

**File:** `langchain_core/tools/base.py:635`

```python
def invoke(self, input: str | dict | ToolCall, config=None, **kwargs):
    tool_input, kwargs = _prep_run_args(input, config, **kwargs)
    return self.run(tool_input, **kwargs)
```

`_prep_run_args()` normalizes the input: if it's a `ToolCall` object (with `.args`), it extracts the args dict. Then `self.run()` validates against `args_schema` and calls `self._run()`.

---

# Stage 2: bind_tools() — Wiring Tools to the LLM

**File:** `langchain_openai/chat_models/base.py:2119`

### Your code (in create_agent or bind_tools)

```python
llm_with_tools = llm.bind_tools([order_status_tool, policy_retriever_tool, ...])
```

### What happens

```python
def bind_tools(self, tools, *, tool_choice=None, strict=None, ...):
    # 1. Convert each tool to OpenAI JSON Schema
    formatted_tools = [
        convert_to_openai_tool(tool, strict=strict) for tool in tools
    ]
    # formatted_tools = [
    #     {"type": "function", "function": {"name": "order_status_tool", "description": "...", "parameters": {...}}},
    #     {"type": "function", "function": {"name": "policy_retriever_tool", ...}},
    # ]

    # 2. Handle tool_choice
    if tool_choice:
        # "any" → "required", str → {"type": "function", "function": {"name": "..."}}
        kwargs["tool_choice"] = normalized_tool_choice

    # 3. Return a new ChatOpenAI with tools baked into kwargs
    return self.bind(tools=formatted_tools, **kwargs)
    # bind() creates a RunnableBinding — a frozen copy with default kwargs
```

**Key:** `bind()` creates a `RunnableBinding`. It doesn't modify the original LLM. It returns a new runnable that always passes `tools=formatted_tools` to every invocation.

### convert_to_openai_tool() — LangChain tool → OpenAI function schema

**File:** `langchain_core/utils/function_calling.py:515`

```python
def convert_to_openai_tool(tool, *, strict=None) -> dict:
    if isinstance(tool, BaseTool):
        # Extract name, description, args_schema
        oai_function = convert_to_openai_function(tool, strict=strict)
        return {"type": "function", "function": oai_function}
    elif isinstance(tool, dict):
        return tool  # Already an OpenAI format dict
    elif callable(tool):
        # Auto-convert from function signature
        ...
```

`convert_to_openai_function()` extracts:
- `name` from `tool.name`
- `description` from `tool.description`
- `parameters` from `tool.tool_call_schema.model_json_schema()`

Result:

```json
{
    "type": "function",
    "function": {
        "name": "order_status_tool",
        "description": "Look up the status of a specific order by order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    }
}
```

---

# Stage 3: ChatOpenAI._generate() — Injecting Tools into the Request

**File:** `langchain_openai/chat_models/base.py:1612`

When `invoke()` is called on the bound LLM:

```python
def _generate(self, messages, stop=None, run_manager=None, **kwargs):
    self._ensure_sync_client_available()
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    # kwargs contains: tools=[...], tool_choice="auto"
    # ← These were injected by bind() / bind_tools()
```

### _get_request_payload() — building the HTTP body

```python
def _get_request_payload(self, input_, *, stop=None, **kwargs):
    messages = self._convert_input(input_).to_messages()
    payload = {**self._default_params, **kwargs}

    # Convert messages to OpenAI format
    payload["messages"] = [
        _convert_message_to_dict(m) for m in messages
    ]

    # kwargs already contains "tools" from bind_tools!
    # payload = {
    #     "model": "glm-4-flash",
    #     "messages": [{"role": "user", "content": "What's order 1001?"}],
    #     "tools": [{"type": "function", "function": {...}}, ...],
    #     "tool_choice": "auto",   # optional, depends on bind_tools args
    # }

    return payload
```

Then the HTTP call:

```python
raw_response = self.client.with_raw_response.create(**payload)
# POST https://open.bigmodel.cn/api/paas/v4/chat/completions
# Body: {"model": "glm-4-flash", "messages": [...], "tools": [...], ...}
```

---

# Stage 4: The LLM Responds

The LLM decides to call a tool instead of answering directly. The response:

```json
{
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "order_status_tool",
                    "arguments": "{\"order_id\": \"1001\"}"
                }
            }]
        },
        "finish_reason": "tool_calls"
    }]
}
```

Key signals:
- `content: null` — the assistant text is empty because it wants to call a tool
- `tool_calls` — array of tool calls (can be multiple for parallel calls)
- `finish_reason: "tool_calls"` — the LLM stopped because it wants tool execution
- `arguments` — JSON string, not an object (per OpenAI's API spec)

---

# Stage 5: _create_chat_result() — Parsing the Response

**File:** `langchain_openai/chat_models/base.py:1714`

```python
def _create_chat_result(self, response, generation_info=None):
    response_dict = response.model_dump(...)
    choices = response_dict["choices"]

    for res in choices:
        message = _convert_dict_to_message(res["message"])
        gen = ChatGeneration(message=message, ...)
        generations.append(gen)

    return ChatResult(generations=generations, ...)
```

### _convert_dict_to_message() — OpenAI dict → AIMessage

**File:** `langchain_openai/chat_models/base.py:198`

```python
def _convert_dict_to_message(_dict):
    role = _dict["role"]
    if role == "assistant":
        content = _dict.get("content", "") or ""

        # Parse tool_calls from the response
        tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))

        return AIMessage(
            content=content,          # "" (empty because tool call)
            additional_kwargs={},
            tool_calls=tool_calls,    # [{"name": "order_status_tool", "args": {"order_id": "1001"}, "id": "call_abc123"}]
        )
```

The `parse_tool_call()` function:
```python
def parse_tool_call(raw_tool_call, return_id=False):
    func = raw_tool_call["function"]
    return {
        "name": func["name"],
        "args": json.loads(func["arguments"]),  # ← JSON parse: "{\"order_id\": \"1001\"}" → {"order_id": "1001"}
        "id": raw_tool_call["id"],
    }
```

**Result:** The `AIMessage` returned to your code has:
- `content = ""` (empty)
- `tool_calls = [{"name": "order_status_tool", "args": {"order_id": "1001"}, "id": "call_abc123"}]`

---

# Stage 6: ToolNode — Executing the Tools

**File:** `langgraph/prebuilt/tool_node.py:792`

In LangGraph, when a node produces an `AIMessage` with `tool_calls`, the `tools_condition` router branches to the ToolNode.

### ToolNode._func()

```python
def _func(self, input, config, runtime):
    # 1. Parse tool calls from the input (AIMessage)
    tool_calls, input_type = self._parse_input(input)
    # tool_calls = [{"name": "order_status_tool", "args": {"order_id": "1001"}, "id": "call_abc123"}]

    # 2. Build ToolRuntime for each call
    tool_runtimes = []
    for call, cfg in zip(tool_calls, config_list):
        tool_runtime = ToolRuntime(
            state=self._extract_state(input, cfg),
            tool_call_id=call["id"],
            config=cfg,
            context=runtime.context,
            store=runtime.store,
            stream_writer=runtime.stream_writer,
            ...
        )
        tool_runtimes.append(tool_runtime)

    # 3. Execute tools in parallel (thread pool)
    with get_executor_for_config(config) as executor:
        outputs = list(executor.map(self._run_one, tool_calls, input_types, tool_runtimes))

    # 4. Combine outputs back into state format
    return self._combine_tool_outputs(outputs, input_type)
```

### ToolNode._run_one() — execute a single tool

```python
def _run_one(self, call, input_type, tool_runtime):
    tool_name = call["name"]
    tool = self._tools_by_name[tool_name]

    try:
        # Call the tool with args from the LLM
        result = tool.invoke(call["args"], tool_runtime.config)
        return ToolMessage(
            content=str(result),
            tool_call_id=call["id"],
            name=tool_name,
        )
    except Exception as e:
        if self._handle_tool_errors:
            return ToolMessage(
                content=f"Error: {e}",
                tool_call_id=call["id"],
                name=tool_name,
                status="error",
            )
        raise
```

**The call chain inside `tool.invoke()`:**

```
ToolNode._run_one(call={"name": "order_status_tool", "args": {"order_id": "1001"}, "id": "..."})
  → tool.invoke(input={"order_id": "1001"}, config=...)
    → _prep_run_args(input) → {"order_id": "1001"}
    → self.run({"order_id": "1001"})
      → args_schema.model_validate({"order_id": "1001"}) → Schema(order_id="1001")
      → self._run(order_id="1001")
        → get_order_status("1001")   # ← YOUR ACTUAL FUNCTION
        → "Order #1001: Processing"
      → return "Order #1001: Processing"
    → return "Order #1001: Processing"
  → ToolMessage(content="Order #1001: Processing", tool_call_id="call_abc123", name="order_status_tool")
```

### _combine_tool_outputs()

```python
def _combine_tool_outputs(self, outputs, input_type):
    # tool_messages = [
    #     ToolMessage(content="Order #1001: Processing", tool_call_id="call_abc123", name="order_status_tool"),
    # ]

    if input_type == "dict":
        return {"messages": tool_messages}
    elif input_type == "list":
        return tool_messages
```

Returns `{"messages": [ToolMessage(...)]}`. In LangGraph, this merges into state via the `add` reducer on `messages`.

---

# Stage 7: The Loop Continues

The `ToolMessage` is appended to `state["messages"]` by the reducer. On the next loop iteration:

```
messages = [
    HumanMessage("What's order 1001?"),
    AIMessage(content="", tool_calls=[{"name": "order_status_tool", "args": {"order_id": "1001"}, "id": "call_abc123"}]),
    ToolMessage(content="Order #1001: Processing", tool_call_id="call_abc123", name="order_status_tool"),
]
```

The LLM sees the full conversation with the tool result and now generates the final answer:

```
AIMessage(content="Your order #1001 is currently being processed.")
```

---

# The Complete Round-Trip

```
1. YOUR CODE
   @tool
   def order_status_tool(order_id: str) -> str:
       ...
   → StructuredTool created with auto-generated Pydantic args_schema

2. BINDING
   llm.bind_tools([order_status_tool])
   → convert_to_openai_tool() → {"type": "function", "function": {"name": "...", "parameters": {...}}}
   → ChatOpenAI.bind(tools=[...]) → RunnableBinding with tools in kwargs

3. INVOCATION
   llm_with_tools.invoke("What's order 1001?")
   → ChatOpenAI._generate()
   → _get_request_payload() → payload["tools"] = [openai_tool_schemas]
   → HTTP POST to Zhipu API

4. LLM RESPONSE
   {choices: [{message: {content: null, tool_calls: [{function: {name: "order_status_tool", arguments: '{"order_id": "1001"}'}}]}}]}

5. PARSE RESPONSE
   _create_chat_result() → _convert_dict_to_message()
   → AIMessage(content="", tool_calls=[{"name": "order_status_tool", "args": {"order_id": "1001"}, "id": "call_abc123"}])

6. EXECUTE TOOL
   ToolNode._func()
   → _run_one(call) → tool.invoke({"order_id": "1001"}) → get_order_status("1001") → "Order #1001: Processing"
   → ToolMessage(content="Order #1001: Processing", tool_call_id="call_abc123", name="order_status_tool")
   → Return {"messages": [ToolMessage(...)]}

7. LOOP
   LangGraph appends ToolMessage to state["messages"] via add reducer
   LLM sees: HumanMessage + AIMessage(tool_call) + ToolMessage(result)
   LLM generates final answer: "Your order #1001 is currently being processed."
```

---

# Key Design Patterns

## 1. Schema auto-generation from type hints

```python
def foo(x: int, y: str = "default") -> str:
    ...

# Auto-generated:
class fooSchema(BaseModel):
    x: int
    y: str = "default"
```

No separate schema definition needed. The `@tool` decorator introspects the function signature.

## 2. bind() creates immutable copies

`bind_tools()` doesn't modify the original LLM. It creates a `RunnableBinding` — a frozen snapshot with baked-in kwargs. This means you can have one base LLM and multiple bound versions:

```python
base_llm = ChatOpenAI(model="glm-4-flash")
llm_with_tools = base_llm.bind_tools(tools)      # for the agent
llm_plain = base_llm                              # for generation
llm_validator = base_llm.bind(temperature=0.0)    # for validation
```

## 3. The parallel execution pattern

```python
with get_executor_for_config(config) as executor:
    outputs = list(executor.map(self._run_one, tool_calls, ...))
```

If the LLM calls 3 tools, they execute in parallel via `ThreadPoolExecutor`. This is why LangGraph's `ToolNode` is better than a sequential loop.

## 4. ToolMessage preserves the tool_call_id

```python
ToolMessage(content="result", tool_call_id="call_abc123", name="tool_name")
```

The `tool_call_id` is critical. Without it, the LLM can't match which result belongs to which tool call when multiple tools are called. The OpenAI API requires the `tool_call_id` in the response to correctly pair `tool_calls` with their results.

---

# Interview Talking Points

## "How does @tool work under the hood?"

> "The decorator introspects the function's type hints and docstring, auto-generates a Pydantic model for argument validation, and wraps the function in a `StructuredTool`. The Pydantic model becomes the JSON Schema sent to the LLM. When the tool is called, `BaseTool.invoke()` validates the input against the schema and calls the wrapped function."

## "How does bind_tools() work?"

> "It converts each tool to an OpenAI function schema using `convert_to_openai_tool()`, then calls `bind()` which creates a `RunnableBinding` — a frozen copy of the model with those tools baked into `kwargs`. When `_generate()` runs, `_get_request_payload()` picks up `tools` from kwargs and includes them in the HTTP request body alongside the messages."

## "How does ToolNode execute tools in parallel?"

> "`ToolNode._func()` extracts all `tool_calls` from the incoming `AIMessage`, builds a `ToolRuntime` for each, and uses `concurrent.futures.ThreadPoolExecutor.map()` to execute them concurrently. Each result is wrapped in a `ToolMessage` with the matching `tool_call_id`. The outputs are combined and returned as state updates."

## "What connects tool output back to the conversation?"

> "The `tool_call_id` in `ToolMessage` matches the `id` in the original `ToolCall`. When the LLM sees the full message list — `HumanMessage` → `AIMessage(tool_calls=[...])` → `ToolMessage(tool_call_id=matching_id)` — it pairs each result with its request and generates a grounded final answer. In LangGraph, the `add` reducer on `messages` appends the `ToolMessage` to the state so the next LLM call sees it."
