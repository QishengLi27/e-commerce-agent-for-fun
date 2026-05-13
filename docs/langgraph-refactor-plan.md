# LangGraph 重构计划

## 目标

把当前的"黑盒 Agent"（`create_agent()`）改造成一个**显式状态机**——每个决策步骤都可见、可控、可调试。

## 为什么用 LangGraph？

| 维度 | 当前 `create_agent()` | LangGraph |
|------|----------------------|-----------|
| 可控性 | 黑盒，不知道 LLM 什么时候调工具 | 每个节点、每条边都明确定义 |
| 调试 | 出问题时只能看日志猜 | 可以导出 Mermaid 图，可视化执行路径 |
| 中断/审核 | 不支持 | 支持 human-in-the-loop，某步卡住等人确认 |
| 循环 | 隐式循环（LLM 自己决定） | 显式循环，可设最大轮数、超时 |
| 状态持久化 | 无 | 可保存/恢复状态，断点续跑 |

---

## 最终架构图

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  sanitize_input │  ← 清洗输入、查缓存
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ classify_intent │  ← LLM 判断意图
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │ order_node  │  │ policy_node │  │ weather_node│
    │  订单查询    │  │  政策检索   │  │  天气查询   │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │ generate_reply  │  ← 生成最终回复
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ update_memory   │  ← 保存对话
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Output        │
                    └─────────────────┘
```

---

## Step 1: 环境准备与基础认知

### 1.1 安装依赖

```bash
cd apps/backend
source venv/bin/activate
pip install langgraph
```

### 1.2 理解三个核心概念

**State（状态）**
```python
class AgentState(TypedDict):
    messages: list          # 对话历史
    intent: str             # 识别出的意图: order | policy | weather | unknown
    tool_result: str        # 工具执行结果
    final_answer: str       # 最终回复
    session_id: str
```

**Node（节点）**
一个纯函数：接收 State，修改 State，返回 State。
```python
def order_node(state: AgentState) -> AgentState:
    # 查询订单
    result = get_order_status(state["order_id"])
    state["tool_result"] = result
    return state
```

**Edge（边）**
控制流：A 节点之后去哪个节点。
```python
# 无条件边
graph.add_edge("sanitize_input", "classify_intent")

# 条件边（路由）
graph.add_conditional_edges(
    "classify_intent",
    route_by_intent,          # 判断函数
    {
        "order": "order_node",
        "policy": "policy_node",
        "weather": "weather_node",
    }
)
```

---

## Step 2: 定义 State 和 Graph 骨架

### 2.1 新建文件 `backend/graph/agent_graph.py`

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# ─── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """整个 Agent 工作流共享的状态对象。"""
    messages: Annotated[list, operator.add]   # 自动合并列表
    user_input: str
    intent: str
    tool_result: str
    final_answer: str
    cached: bool

# ─── Nodes ───────────────────────────────────────────────────────────────────

def sanitize_input(state: AgentState) -> AgentState:
    """Step 1: 清洗输入、检查缓存。"""
    ...

def classify_intent(state: AgentState) -> AgentState:
    """Step 2: LLM 判断用户意图。"""
    ...

def order_node(state: AgentState) -> AgentState:
    """Step 3a: 订单查询。"""
    ...

def policy_node(state: AgentState) -> AgentState:
    """Step 3b: 政策检索。"""
    ...

def weather_node(state: AgentState) -> AgentState:
    """Step 3c: 天气查询。"""
    ...

def generate_reply(state: AgentState) -> AgentState:
    """Step 4: 生成最终回复。"""
    ...

def update_memory(state: AgentState) -> AgentState:
    """Step 5: 保存到记忆。"""
    ...

# ─── Conditional Router ──────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """根据 intent 字段决定下一步去哪个节点。"""
    return state["intent"]

# ─── Build Graph ─────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

# 注册节点
builder.add_node("sanitize_input", sanitize_input)
builder.add_node("classify_intent", classify_intent)
builder.add_node("order_node", order_node)
builder.add_node("policy_node", policy_node)
builder.add_node("weather_node", weather_node)
builder.add_node("generate_reply", generate_reply)
builder.add_node("update_memory", update_memory)

# 入口
builder.set_entry_point("sanitize_input")

# 边
builder.add_edge("sanitize_input", "classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "order": "order_node",
        "policy": "policy_node",
        "weather": "weather_node",
        "unknown": "generate_reply",   # 直接让 LLM 回答
    },
)
builder.add_edge("order_node", "generate_reply")
builder.add_edge("policy_node", "generate_reply")
builder.add_edge("weather_node", "generate_reply")
builder.add_edge("generate_reply", "update_memory")
builder.add_edge("update_memory", END)

# 编译
agent_graph = builder.compile()
```

### 2.2 可视化 Graph

```python
from IPython.display import Image, display

# 生成 Mermaid PNG
png = agent_graph.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(png)
```

运行后你会得到一张流程图，清清楚楚看到每一步的走向。

---

## Step 3: 逐个实现 Node

### 3.1 `sanitize_input` — 输入清洗 + 缓存检查

复用现有的 `clean_query()` 和 `get_cached_response()`：

```python
def sanitize_input(state: AgentState) -> AgentState:
    raw = state["user_input"]
    cleaned = clean_query(raw)
    state["user_input"] = cleaned

    # 检查语义缓存
    cached = get_cached_response(cleaned)
    if cached:
        state["final_answer"] = cached
        state["cached"] = True
    return state
```

**技巧：** 如果命中缓存，后续节点可以直接跳过。用条件边处理：

```python
def should_skip(state: AgentState) -> str:
    if state.get("cached"):
        return "update_memory"   # 跳过后续步骤，直接保存记忆
    return "classify_intent"

builder.add_conditional_edges("sanitize_input", should_skip, ...)
```

### 3.2 `classify_intent` — 意图识别

**方案 A：纯 LLM 判断（简单）**
```python
from langchain_core.messages import SystemMessage, HumanMessage

INTENT_PROMPT = """判断用户意图，只返回一个单词：
- order（订单相关）
- policy（政策相关）
- weather（天气相关）
- unknown（其他）

用户：{input}
意图："""

def classify_intent(state: AgentState) -> AgentState:
    prompt = INTENT_PROMPT.format(input=state["user_input"])
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()
    # 兜底
    if intent not in ("order", "policy", "weather", "unknown"):
        intent = "unknown"
    state["intent"] = intent
    return state
```

**方案 B：关键词 + LLM 混合（更快更省）**
```python
def classify_intent(state: AgentState) -> AgentState:
    text = state["user_input"].lower()
    if any(w in text for w in ["order", "订单", "status"]):
        state["intent"] = "order"
    elif any(w in text for w in ["policy", "return", "退货", "政策"]):
        state["intent"] = "policy"
    elif any(w in text for w in ["weather", "天气"]):
        state["intent"] = "weather"
    else:
        state["intent"] = "unknown"
    return state
```

**建议：** 先用方案 B 跑通，再升级到方案 A。

### 3.3 `order_node` / `policy_node` / `weather_node`

直接调用已有的 tools：

```python
def order_node(state: AgentState) -> AgentState:
    # 从用户输入中提取 order_id（简单正则或让 LLM 提取）
    # 这里先简化：假设输入就是 order_id
    result = order_status_tool.invoke(state["user_input"])
    state["tool_result"] = result
    return state

def policy_node(state: AgentState) -> AgentState:
    result = policy_retriever_tool.invoke(state["user_input"])
    state["tool_result"] = result
    return state

def weather_node(state: AgentState) -> AgentState:
    result = get_current_weather.invoke(state["user_input"])
    state["tool_result"] = result
    return state
```

### 3.4 `generate_reply` — 生成回复

```python
REPLY_PROMPT = """基于以下信息，用中文或英文（匹配用户语言）生成友好、简洁的回复。

用户问题：{question}
工具查询结果：{result}

回复："""

def generate_reply(state: AgentState) -> AgentState:
    # 如果已经有缓存的 final_answer，直接跳过
    if state.get("final_answer"):
        return state

    prompt = REPLY_PROMPT.format(
        question=state["user_input"],
        result=state.get("tool_result", "无额外信息"),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    state["final_answer"] = response.content.strip()
    return state
```

### 3.5 `update_memory` — 保存对话

复用现有的 `memory_store`：

```python
def update_memory(state: AgentState) -> AgentState:
    memory_store.add_user(state["user_input"])
    memory_store.add_agent(state["final_answer"])
    return state
```

---

## Step 4: 接入 FastAPI

### 4.1 同步调用（非流式）

```python
from backend.graph.agent_graph import agent_graph

@router.post("/chat")
def chat(request: ChatRequest):
    result = agent_graph.invoke({
        "user_input": request.message,
        "messages": [],
    })
    return ChatResponse(response=result["final_answer"])
```

### 4.2 流式调用（streaming）

LangGraph 支持 `astream()`，可以观察每个节点的执行：

```python
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async for event in agent_graph.astream({
            "user_input": request.message,
            "messages": [],
        }):
            # event 包含当前节点名和状态快照
            node_name = list(event.keys())[0]
            state = event[node_name]

            # 只在 generate_reply 节点完成后发送最终答案
            if node_name == "generate_reply" and state.get("final_answer"):
                for word in state["final_answer"].split(" "):
                    yield f"data: {word} \n\n"
                    await asyncio.sleep(0.03)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## Step 5: 添加高级功能

### 5.1 循环控制（防止无限循环）

```python
# 在 State 中添加计数器
class AgentState(TypedDict):
    ...
    loop_count: int

# 在 classify_intent 中计数
def classify_intent(state: AgentState) -> AgentState:
    state["loop_count"] = state.get("loop_count", 0) + 1
    ...

# 条件边：超过 3 轮直接结束
def check_loop_limit(state: AgentState) -> str:
    if state["loop_count"] > 3:
        state["final_answer"] = "问题较复杂，已为您转接人工客服。"
        return "update_memory"
    return "classify_intent"
```

### 5.2 Human-in-the-Loop（人工审核）

场景：当订单金额 > $1000 时，需要人工确认。

```python
builder.add_node("human_review", human_review_node)

def need_human_review(state: AgentState) -> str:
    if is_high_value_order(state["order_id"]):
        return "human_review"
    return "generate_reply"

builder.add_conditional_edges("order_node", need_human_review, {
    "human_review": "human_review",
    "generate_reply": "generate_reply",
})
```

LangGraph 的 `interrupt` 机制可以在任意节点暂停，等人输入后再继续。

### 5.3 持久化状态（断点续跑）

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 每次 invoke 后自动保存状态到 SQLite
memory = SqliteSaver.from_conn_string(":memory:")
agent_graph = builder.compile(checkpointer=memory)

# 后续可以用 thread_id 恢复对话
config = {"configurable": {"thread_id": "user-123"}}
result = agent_graph.invoke({"user_input": "hi"}, config=config)
```

---

## Step 6: 替换旧代码

### 6.1 文件结构调整

```
backend/
├── agent.py              # 保留现有工具 + 辅助函数，移除 create_agent
├── graph/
│   ├── __init__.py
│   ├── agent_graph.py    # StateGraph 定义 + 编译
│   ├── nodes.py          # 所有 node 函数实现
│   └── router.py         # 条件路由函数
```

### 6.2 迁移清单

- [ ] 安装 `langgraph`
- [ ] 新建 `backend/graph/` 包
- [ ] 定义 `AgentState` TypedDict
- [ ] 迁移 `clean_query` → `sanitize_input` node
- [ ] 实现 `classify_intent` node（先关键词方案）
- [ ] 迁移工具调用 → `order_node` / `policy_node` / `weather_node`
- [ ] 实现 `generate_reply` node
- [ ] 连接所有 edges
- [ ] 编译 graph 并导出可视化 PNG
- [ ] 修改 `api/routes.py` 使用新 graph
- [ ] 删除旧 `create_agent` 代码
- [ ] 运行测试，对比新旧行为一致性

---

## 预期收益

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| 意图识别准确率 | 依赖 LLM 黑盒 | 可调试、可替换策略 |
| 响应延迟（天气） | 3s（等 LLM 决策+调用） | 1s（直接路由到 weather_node） |
| 新增工具成本 | 改 prompt + 祈祷 LLM 用对 | 加 node + 加一条边 |
| 可观测性 | 只能看日志 | 可导出执行路径图 |
| 人工介入 | 不支持 | 任意节点可中断等人确认 |

---

## 下一步

要我直接按这个计划开始实现吗？建议从 **Step 2 + Step 3** 开始（搭骨架 + 实现节点），大概 30 分钟就能跑通第一个版本。
