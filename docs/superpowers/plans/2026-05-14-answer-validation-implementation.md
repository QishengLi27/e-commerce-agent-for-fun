# Answer Validation Step — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-based validation node that reviews generated answers for hallucinations before they reach the user.

**Architecture:** Insert a `validate_reply` node between `generate_reply` and `update_memory` in the LangGraph pipeline. The node calls the LLM with a structured prompt comparing the answer against tool results, sets a `validation_flag` in state, and always passes through. The flag is surfaced via the API response schema.

**Tech Stack:** LangGraph, LangChain, FastAPI, Pydantic

**Files:**
- Modify: `apps/backend/backend/graph/nodes.py`
- Modify: `apps/backend/backend/graph/agent_graph.py`
- Modify: `apps/backend/backend/api/schemas.py`
- Modify: `apps/backend/backend/api/routes.py`

---

### Task 1: Add validation fields to AgentState

**Files:**
- Modify: `apps/backend/backend/graph/nodes.py:33-41`

- [ ] **Step 1: Add `validation_flag` and `validation_notes` to `AgentState`**

Replace the existing `AgentState` class with:

```python
class AgentState(TypedDict, total=False):
    """LangGraph state schema."""
    messages: Annotated[list, add]
    user_input: str
    intent: str
    order_id: str
    tool_result: str
    final_answer: str
    cached: bool
    validation_flag: str
    validation_notes: str
```

---

### Task 2: Add `validate_reply` node + prompt + parsing helper

**Files:**
- Modify: `apps/backend/backend/graph/nodes.py` (append after `generate_reply`, before `update_memory`)

- [ ] **Step 1: Add the validation prompt constant**

Insert after the `_REPLY_PROMPT` block (~line 207):

```python
_VALIDATION_PROMPT = """You are an accuracy auditor for an e-commerce support agent.

Your job is to check whether the agent's answer is fully grounded in the provided tool results.
Do NOT answer the user's question. Only assess accuracy.

User question: {question}
Tool results: {tool_result}
Agent answer: {answer}

Check for:
1. Fabricated data — order IDs, dates, amounts, or statuses not in the tool results
2. Contradictions — claims that conflict with the tool results
3. Unsupported claims — factual assertions with no basis in the tool results

Return ONLY one of these exact labels followed by an optional one-line note:

- valid — answer is fully grounded in the tool results
- unverified_claims — answer contains claims not supported by the tool results
- not_applicable — no tool results to validate against

Format: LABEL | note

Example: "valid | All order details match the tool output"
Example: "unverified_claims | Answer mentions order #1005 but tool result only shows #1003"
"""
```

- [ ] **Step 2: Add the parsing helper**

Insert after the prompt constant:

```python
def _parse_validation(raw: str) -> tuple[str, str]:
    """Parse LLM validation output into (flag, note). Defaults to unverified_claims on malformed input."""
    if not raw:
        return ("unverified_claims", "empty validation response")
    try:
        parts = raw.split("|", 1)
        flag = parts[0].strip()
        note = parts[1].strip() if len(parts) > 1 else ""
        if flag not in ("valid", "unverified_claims", "not_applicable"):
            return ("unverified_claims", f"unrecognized flag: {flag}")
        return (flag, note)
    except Exception:
        return ("unverified_claims", f"parse error: {raw[:100]}")
```

- [ ] **Step 3: Add the `validate_reply` node function**

Insert after the parsing helper:

```python
def validate_reply(state: AgentState) -> AgentState:
    """Validate that the generated answer is grounded in tool results."""
    # Skip if answer came from cache (already validated before caching)
    if state.get("cached"):
        state["validation_flag"] = "valid"
        state["validation_notes"] = "cache hit — previously validated"
        logger.info("[graph] Validation skipped: cache hit")
        return state

    question = state.get("user_input", "")
    tool_result = state.get("tool_result", "")
    answer = state.get("final_answer", "")

    # No tool result to validate against (unknown intent path)
    if not tool_result:
        state["validation_flag"] = "not_applicable"
        state["validation_notes"] = "no tool result to validate against"
        logger.info("[graph] Validation: not_applicable (no tool result)")
        return state

    try:
        prompt = _VALIDATION_PROMPT.format(
            question=question,
            tool_result=tool_result,
            answer=answer,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        flag, note = _parse_validation(raw)
        state["validation_flag"] = flag
        state["validation_notes"] = note
        logger.info("[graph] Validation: %s — %s", flag, note[:80])
    except Exception as e:
        state["validation_flag"] = "unverified_claims"
        state["validation_notes"] = f"validation call failed: {e}"
        logger.warning("[graph] Validation error: %s", e)

    return state
```

---

### Task 3: Register `validate_reply` node and rewire edges

**Files:**
- Modify: `apps/backend/backend/graph/agent_graph.py`

- [ ] **Step 1: Import `validate_reply`**

Change line 16 in the import block — add `validate_reply` to the existing import:

```python
from backend.graph.nodes import (
    AgentState,
    sanitize_input,
    classify_intent,
    route_by_intent,
    order_node,
    list_orders_node,
    policy_node,
    weather_node,
    generate_reply,
    validate_reply,
    update_memory,
)
```

- [ ] **Step 2: Register the node**

After `builder.add_node("generate_reply", generate_reply)` (line 31), add:

```python
builder.add_node("validate_reply", validate_reply)
```

- [ ] **Step 3: Rewire edges**

Replace line 60 (`builder.add_edge("generate_reply", "update_memory")`) with:

```python
builder.add_edge("generate_reply", "validate_reply")
builder.add_edge("validate_reply", "update_memory")
```

The final edge section should read:

```python
# All tool nodes converge to generate_reply
builder.add_edge("order_node", "generate_reply")
builder.add_edge("list_orders_node", "generate_reply")
builder.add_edge("policy_node", "generate_reply")
builder.add_edge("weather_node", "generate_reply")

# Validate before persisting
builder.add_edge("generate_reply", "validate_reply")
builder.add_edge("validate_reply", "update_memory")
builder.add_edge("update_memory", END)
```

---

### Task 4: Add `validation_flag` to ChatResponse schema

**Files:**
- Modify: `apps/backend/backend/api/schemas.py`

- [ ] **Step 1: Add the field to `ChatResponse`**

Replace the `ChatResponse` class with:

```python
class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    response: str
    sources: List[Source] = []
    cached: bool = False
    latency_ms: Optional[int] = None
    validation_flag: Optional[str] = None
```

---

### Task 5: Include `validation_flag` in API responses

**Files:**
- Modify: `apps/backend/backend/api/routes.py`

- [ ] **Step 1: Pass `validation_flag` in `/chat` endpoint**

In the `chat` function (~line 70), add `validation_flag` to the `ChatResponse` construction:

```python
return ChatResponse(
    session_id=request.session_id,
    response=result.get("final_answer", ""),
    cached=result.get("cached", False),
    latency_ms=latency,
    validation_flag=result.get("validation_flag"),
)
```

- [ ] **Step 2: Pass `validation_flag` in `/chat/stream` endpoint**

In `_stream_response` (~line 88), after the answer is obtained, yield the validation flag as a final SSE metadata event before `[DONE]`:

```python
answer = result.get("final_answer", "")
validation_flag = result.get("validation_flag")

# Stream word-by-word for visible typing effect
words = answer.split(" ")
for i, word in enumerate(words):
    chunk = word + (" " if i < len(words) - 1 else "")
    safe = chunk.replace("\n", "\\n").replace("\r", "")
    yield f"data: {safe}\n\n"
    await asyncio.sleep(0.03)

if validation_flag:
    yield f"data: [FLAG:{validation_flag}]\n\n"

yield "data: [DONE]\n\n"
```

---

### Task 6: Manual verification

- [ ] **Step 1: Verify the graph compiles without errors**

Run: `cd apps/backend && python -c "from backend.graph.agent_graph import agent_graph; print('Graph compiled OK')"`
Expected: prints "Graph compiled OK" with no tracebacks.

- [ ] **Step 2: Verify the state schema accepts new fields**

Run: `cd apps/backend && python -c "
from backend.graph.nodes import AgentState
s = AgentState(user_input='test', messages=[], validation_flag='valid', validation_notes='ok')
print('State created OK:', s.get('validation_flag'))
"`
Expected: prints "State created OK: valid"

- [ ] **Step 3: Smoke test with `main.py` (CLI path)**

If `main.py` has a quick chat mode, run it with a simple order query and verify no crashes. The legacy `agent.py` path does NOT go through the graph, so validation only applies to the API and any graph-invoking paths.

- [ ] **Step 4: Verify API returns validation_flag**

Start the API server, send a chat request, and confirm the response JSON includes `validation_flag`.

---

### Validation checklist

| Check | Expectation |
|-------|-------------|
| Graph compiles | No errors |
| State accepts new fields | `validation_flag` and `validation_notes` present |
| `/chat` response includes `validation_flag` | Field present, one of `valid`/`unverified_claims`/`not_applicable` |
| Cache hit sets `validation_flag: "valid"` | Skip logic works |
| LLM call failure doesn't crash | Falls back to `unverified_claims` |
| Unknown intent sets `not_applicable` | No tool result case handled |
| Existing intents (order, weather, policy, list_orders) still work | No regression |
