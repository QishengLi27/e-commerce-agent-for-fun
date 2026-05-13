# Answer Validation Step — Design Spec

## Summary

Add an LLM-based validation node to the LangGraph agent pipeline that reviews the generated answer for accuracy (hallucinations, fabricated data, unsupported claims) before it reaches the user. On failure, flag the answer but always pass it through — never block.

## Motivation

The agent currently generates replies from tool results with no verification. The LLM can hallucinate order IDs, amounts, dates, or make claims unsupported by the actual tool output. This validation step catches those issues and surfaces them for monitoring.

## State additions

Two new fields in `AgentState` (`backend/graph/nodes.py`):

```python
validation_flag: str    # "valid" | "unverified_claims" | "not_applicable"
validation_notes: str   # brief explanation from the LLM validator
```

## Graph flow

```
sanitize_input → classify_intent → [tool_node] → generate_reply
                                                       ↓
                                              validate_reply  ← NEW
                                                       ↓
                                              update_memory → END
```

Edge change: `generate_reply → update_memory` becomes `generate_reply → validate_reply → update_memory`.

## `validate_reply` node

### Skip conditions

- **Cache hit** (`cached=True`): answer was already validated before caching, skip to avoid redundant LLM calls
- **No tool result**: intent was `unknown` and no tool ran — nothing to validate against, flag `not_applicable`

### Core logic

1. Build prompt with user question, tool result, and generated answer
2. Call LLM for accuracy assessment
3. Parse the structured response (LABEL | note)
4. Set `validation_flag` and `validation_notes` in state
5. Always pass through — answer is never blocked or rewritten

### Validation prompt

```
You are an accuracy auditor for an e-commerce support agent.

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
```

### Parsing

Split on `"|"` — first part is the flag, second is the note. If the LLM returns malformed output, default to `unverified_claims` (fail-safe: err on the side of flagging).

### Error handling

If the LLM call fails (timeout, API error), set `validation_flag = "unverified_claims"` with an error note and pass through. The validation must never block the response.

## API changes

Add `validation_flag` to `ChatResponse` in `backend/api/schemas.py`:

```python
class ChatResponse(BaseModel):
    session_id: str
    response: str
    cached: bool
    latency_ms: int
    validation_flag: str | None = None  # None when cached (pre-validated)
```

Both `/chat` and `/chat/stream` endpoints include the flag in their responses.

### Streaming note

For `/chat/stream`, the validation flag is only available after the full graph invocation completes. The streaming endpoint currently runs the full graph before streaming words — so the flag can be yielded as a final SSE event (e.g., `data: {"validation_flag": "valid"}`) after `[DONE]`, or included as an HTTP header. Simplest approach: return it in the post-stream metadata if needed; otherwise, the non-streaming `/chat` endpoint provides it directly.

## Files touched

| File | Change |
|------|--------|
| `backend/graph/nodes.py` | Add `validate_reply` node, prompt, parsing helper; add `validation_flag` and `validation_notes` to `AgentState` |
| `backend/graph/agent_graph.py` | Register `validate_reply` node, rewire `generate_reply → validate_reply → update_memory` |
| `backend/api/schemas.py` | Add `validation_flag` to `ChatResponse` |
| `backend/api/routes.py` | Include `validation_flag` in response dicts |

No new files, no new dependencies.

## Edge case summary

| Case | Behavior |
|------|----------|
| Cache hit | Skip validation |
| Unknown intent (no tool_result) | Flag `not_applicable` |
| LLM call fails | Flag `unverified_claims`, note the error, pass through |
| Malformed validation output | Default to `unverified_claims` |

## Future extensions

- **Retry loop**: if `unverified_claims`, loop back to `generate_reply` with a correction hint (requires conditional edge from `validate_reply`)
- **Metrics dashboard**: expose validation pass/fail rates per intent type
- **Threshold-based blocking**: if validation fails > N times in a session, switch to templated fallback responses
