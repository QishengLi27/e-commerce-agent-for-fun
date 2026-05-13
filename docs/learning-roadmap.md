# LangChain & Agent 持续学习路线图

基于你现有的电商客服 Agent 项目，以下是一个由浅入深的开发路线图。每个阶段都有具体可落地的功能。

---

## Phase 1: 强化基础（1-2 周）

### 1.1 真正的 LLM Streaming

**现状：** 你的 `/chat/stream` 是先等 LLM 返回完整结果，再逐词发送给前端。

**目标：** 让前端实时看到 LLM 逐字输出的效果（真正的 token-by-token 流式传输）。

**学习点：**
- LangChain 的 `astream()` / `astream_events()`
- FastAPI 的 `StreamingResponse` 与 SSE 协议
- 前端 `ReadableStream` 的实时渲染

**实现思路：**
```python
# 用 astream_events 获取实时 token
async for event in agent.astream_events(...):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

---

### 1.2 Agent 返回结构化数据

**现状：** Agent 返回纯文本字符串。

**目标：** 让 Agent 可以返回 JSON、表格等结构化数据，方便前端渲染。

**学习点：**
- Pydantic Output Parser
- `with_structured_output()` 方法
- LangChain 的 `create_structured_output_runnable`

**应用场景：**
- 用户问"我的订单列表" → 返回 `{"orders": [...]}`
- 前端直接渲染成表格，而不是解析文本

---

### 1.3 添加 Evaluation（评估框架）

**现状：** 靠人工测试每个改动。

**目标：** 建立自动化测试集，每次改代码后自动跑评估。

**学习点：**
- LangSmith 的 trace 和 evaluation
- 自定义 evaluator（准确率、相关性、幻觉检测）
- 数据集管理

**实现：**
```python
# 创建一个测试数据集
dataset = [
    {"question": "订单1001的状态", "expected": "Delivered", "tools": ["order_status"]},
    {"question": "退货政策", "expected": "30-day", "tools": ["policy_retriever"]},
]
```

---

## Phase 2: Agent 进阶架构（2-3 周）

### 2.1 从 ReAct 升级到 LangGraph

**现状：** 使用 `create_agent()`，黑盒运行。

**目标：** 用 **LangGraph** 构建可视化的 Agent 状态机，精确控制每一步。

**学习点：**
- StateGraph、Node、Edge 的概念
- 条件边（Conditional Edges）
- 循环和状态持久化

**架构改造：**
```
User Input
    → [理解意图] → 路由节点
        → 订单查询 → [order_node] → 结束
        → 政策查询 → [policy_node] → 结束
        → 天气查询 → [weather_node] → 结束
        → 模糊/复杂 → [llm_reasoning_node] → 结束
```

**好处：**
- 简单意图直接走专用节点（更快、更准）
- 复杂意图才走 LLM 推理
- 可以添加"人工审核"节点（human-in-the-loop）

---

### 2.2 多 Agent 协作（Multi-Agent）

**目标：** 不是一个大 Agent 做所有事，而是多个专职 Agent 协作。

**设计：**

| Agent | 职责 | 工具 |
|-------|------|------|
| **Router Agent** | 判断用户意图，分发给其他 Agent | 无 |
| **Order Agent** | 处理所有订单相关查询 | order_status, list_orders |
| **Policy Agent** | 处理政策/退换货问题 | policy_retriever |
| **Weather Agent** | 天气查询 | get_current_weather |
| **Escalation Agent** | 检测到用户不满时，转人工 | 发送通知 |

**学习点：**
- Supervisor 模式（一个总指挥 Agent）
- Agent 间通信（传递 State）
- LangGraph 的 `Send` API

---

### 2.3 向量记忆（Vector Memory）

**现状：** `memory_store.json` 存最近 N 条对话，是线性存储。

**目标：** 用向量数据库做语义记忆，Agent 能"回忆起"一周前聊过的相关内容。

**学习点：**
- `VectorStoreRetrieverMemory`
- 对话摘要 + 向量检索的混合记忆
- ConversationBufferWindowMemory vs ConversationSummaryMemory

**实现：**
```python
# 每次对话存入向量库
memory_vectorstore = PGVector(
    collection_name="conversation_memory",
    ...
)

# 用户新提问时，检索历史相关对话
relevant_history = memory_vectorstore.similarity_search(query, k=3)
```

---

## Phase 3: RAG 进阶（2-3 周）

### 3.1 Query Transformation（查询变换）

**现状：** 用户 query 直接拿去检索。

**目标：** 在检索前用 LLM 优化/改写 query，提高召回率。

**技术栈：**
- **HyDE** (Hypothetical Document Embeddings): 让 LLM 先写一个理想答案，再用这个答案做向量检索
- **Multi-Query**: 从一个问题生成 3-5 个不同表述的查询，分别检索后合并
- **Step-back Prompting**: 从具体问题抽象到通用概念，先查通用再查具体

---

### 3.2 Self-RAG / Corrective RAG

**目标：** Agent 能判断检索结果是否有用，如果没有就换策略。

**流程：**
```
1. 检索政策文档
2. LLM 判断："这些文档能回答问题吗？"
   → 能 → 生成答案
   → 不能 → 改写 query 重新检索（或 fallback 到通用知识）
```

**学习点：**
- LangGraph 的循环结构
- Self-RAG 论文实现
- CRAG (Corrective RAG) 模式

---

### 3.3 Agentic RAG（Agent + RAG 深度融合）

**目标：** RAG 不再是独立的检索模块，而是 Agent 的一个子工作流。

**场景：** 用户问"我能退掉订单1001里的电子产品吗？"

**Agent 自主执行：**
```
1. 查询订单1001状态 → "Delivered on 2023-10-15"
2. 计算是否超过30天 → "已超过30天"
3. 检索电子产品退货政策 → "14天退货期"
4. 综合判断 → "很遗憾，您的订单已超出退货期限..."
```

这需要 Agent 能**串行调用多个工具并做中间推理**。

---

## Phase 4: 工程化与部署（2-3 周）

### 4.1 LangServe 部署

**目标：** 用 LangServe 将 Agent 封装为标准化服务。

**学习点：**
- `add_routes(app, agent_chain, path="/agent")`
- Playground 调试界面
- 自动生成的 OpenAPI 文档

### 4.2 LangSmith 全流程监控

**目标：** 线上环境全链路可观测。

**接入点：**
- 每次 LLM 调用自动记录（latency, token usage, cost）
- 每次工具调用记录输入/输出
- 错误追踪和报警
- A/B 测试不同 prompt

### 4.3 提示词版本管理

**目标：** Prompt 不再是硬编码字符串，而是可管理、可回滚的配置。

**方案：**
- Prompt 存入数据库/配置文件
- 支持多版本（v1, v2, v3）
- 线上灰度：50% 流量用 v2 prompt，对比效果

---

## Phase 5: 前沿探索（持续）

### 5.1 本地小模型 + 大模型混合

**场景：** 简单意图识别用本地 Llama 3（免费、快），复杂推理才调 GLM-4/GPT-4（贵、慢）。

### 5.2 视觉 Agent

**场景：** 用户上传商品图片问"这个能退吗？"，Agent 需要 OCR + 视觉理解 + 政策检索。

### 5.3 自主 Agent（AutoGPT 模式）

**场景：** 用户说"帮我处理所有待发货订单"，Agent 自主规划、执行、验证，不需要逐步确认。

---

## 推荐学习顺序

```
Week 1-2:  真正 Streaming + 结构化输出 + Evaluation
Week 3-4:  LangGraph 重构 + 多 Agent 架构
Week 5-6:  向量记忆 + Query Transformation + Self-RAG
Week 7-8:  LangServe + LangSmith + Prompt 版本管理
Week 9+:   本地模型混合 / 视觉 / 自主 Agent（选感兴趣的）
```

---

## 每个阶段的具体交付物

| 阶段 | 可运行的功能 | 学习资料 |
|------|-------------|---------|
| 1.1 | 前端实时看到 LLM "打字"效果 | LangChain `astream_events` 文档 |
| 1.2 | `/chat` 返回 JSON，前端渲染卡片 | Pydantic + `with_structured_output` |
| 1.3 | `pytest tests/evaluation/` 自动跑 20 条测试 | LangSmith Evaluations |
| 2.1 | 可导出 LangGraph 状态图 PNG | LangGraph 官方教程 |
| 2.2 | "转人工"按钮触发 Escalation Agent | Multi-Agent 示例代码 |
| 2.3 | Agent 能引用昨天聊过的内容 | VectorStoreRetrieverMemory |
| 3.1 | 同一问题换 3 种说法检索，结果合并 | Multi-Query Retriever |
| 3.2 | 检索不到时自动改写 query 重试 | LangGraph 循环 + Self-RAG |
| 4.1 | `http://localhost:8000/agent/playground` | LangServe 文档 |
| 4.2 | LangSmith 看每次调用的 trace | LangSmith 官方 |

---

## 下一步建议

如果你想立刻开始，我建议从 **Phase 1.1（真正的 Streaming）** 入手：
- 改动范围小（只改 backend streaming + frontend reader）
- 效果明显（用户体验质的飞跃）
- 能学到 LangChain 最核心的 `astream_events` API

需要我现在就帮你实现真正的 token-by-token streaming 吗？
