# Agent 框架选型对比与取舍分析

> 本文基于当前电商客服 AI Agent 项目的架构特征，对主流 Agent 框架进行系统性对比，为技术决策提供依据。
> 
> 分析日期：2026-05-18

---

## 一、项目需求画像

当前项目不是一个简单的 "LLM + Prompt" 原型，而是一个面向生产环境的**确定性客服工作流**。以下是从代码和架构中提取的核心需求：

| 需求维度 | 当前实现 | 对框架的要求 |
|---------|---------|-------------|
| **流程控制** | 严格 DAG：`sanitize → classify → tool → generate → validate → [loop?] → memory` | 必须支持显式节点定义、条件边、循环回退 |
| **状态持久化** | Postgres checkpointer（session 级恢复、time-travel） | 原生支持 checkpoint / 状态恢复 |
| **自纠错机制** | `generate_reply → validate_reply → route_after_validation → [retry]` | 支持反馈环和条件重试 |
| **多意图路由** | 6 条分支：order / list_orders / policy / weather / knowledge / unknown | 精确、可审计的路由控制 |
| **RAG** | Hybrid（dense + BM25 + RRF + LLM rerank），自研实现 | 框架不强制 RAG 实现方式 |
| **知识图谱** | 用于意图实体校验（防止 LLM 幻觉商品名） | 可与外部 KG 模块集成 |
| **弹性工程** | Circuit breaker、retry、semantic cache（带 intent 防过期） | 不阻塞自定义弹性逻辑 |
| **模型选择** | 智谱 GLM-4-Flash（OpenAI-compatible，但非 OpenAI/Claude） | 模型无关，不 vendor 锁定 |
| **部署形态** | FastAPI + SSE streaming，面向 C 端用户 | 支持异步流式、并发 session |

**核心结论**：我们需要的是**"状态机编排框架"**，而不是"多 Agent 聊天框架"或"角色扮演框架"。

---

## 二、主流框架对比矩阵

| 框架 | 核心模型 | 状态持久化 | 流程控制 | 模型依赖 | 与项目匹配度 | 关键结论 |
|------|---------|-----------|---------|---------|-------------|---------|
| **LangGraph**（当前） | 有向状态图 + 条件边 | ✅ 原生 Checkpointer | ✅ 节点级精确控制 | 完全无关 | ⭐⭐⭐⭐⭐ | **当前最优解** |
| **CrewAI** | 角色-任务-流程（Role-Task-Process） | ❌ 无 checkpointer | ⚠️ 黑盒流程 | 无关 | ⭐⭐ | 不适合确定性客服 |
| **AutoGen / AG2** | Agent 群聊对话 | ⚠️ 内存默认 | ❌ 对话驱动非确定 | 无关 | ⭐⭐ | 适合研究/辩论 |
| **OpenAI Agents SDK** | Handoff + Context vars | ❌ 默认 ephemeral | ⚠️ 简单路由 | OpenAI-only | ⭐⭐ | Vendor 锁定，且用 GLM |
| **Claude Agent SDK** | Tool-use chain + Sub-agents | ⚠️ 需自建 | ⚠️ 中等 | Claude-only | ⭐⭐ | 模型不匹配 |
| **LlamaIndex Workflows** | 事件驱动（Event-driven） | ⚠️ 事件回溯 | ⚠️ 异步管道 | 无关 | ⭐⭐⭐ | RAG 检索最佳，编排次之 |
| **PydanticAI** | 类型安全函数调用 | ❌ 无 | ❌ 无状态图 | 无关 | ⭐⭐ | 适合简单结构化输出 |
| **Semantic Kernel** | Plugin + Planner | ✅ 有 | ⚠️ Planner 黑盒 | 无关 | ⭐⭐ | .NET/Azure 首选 |
| **原生实现**（直接调 API）| 自己写状态机 | 自己实现 | 完全控制 | 完全无关 | ⭐⭐⭐ | 控制最强，维护成本最高 |

---

## 三、逐个框架深度分析

### 1. LangGraph ✅ — 当前选择，继续保留

**为什么与项目高度匹配：**

- **确定性路由**：`classify_intent → route_by_intent` 在 LangGraph 里是显式条件边（`add_conditional_edges`），每条分支可审计、可单元测试、可打断点。这对客服场景的 SLA 至关重要。
- **自纠错循环**：`generate_reply → validate_reply → route_after_validation → [generate_reply | update_memory]` 是 LangGraph 的原生能力。CrewAI、AutoGen 无法优雅表达这种"验证失败回退重试"的语义。
- **状态持久化**：`AsyncPostgresSaver` 让 session 在服务器重启或扩容后精确恢复。对多轮客服对话来说，这是生产必备。
- **流式可控**：`astream_events()` 支持按节点粒度流式输出，与当前 FastAPI SSE streaming 完美契合。
- **Human-in-the-loop**：未来"转人工"需求，LangGraph 的 `interrupt` 机制是目前业界最成熟的方案。

**生产信号**：Klarna（8500 万用户客服）、LinkedIn、Uber、Replit、Elastic 均在生产环境使用 LangGraph 处理类似的高确定性客服/支持场景。

**劣势**（接受的成本，不是换框架的理由）：
- 学习曲线较陡，需要理解 StateGraph、reducer、checkpoint 等概念
- 样板代码比 CrewAI 多

---

### 2. CrewAI ❌ — 角色扮演型，不适合客服

**模型差异**：CrewAI 的抽象是"招聘一个团队"——定义 Agent 的 Role、Goal、Backstory，再分配 Task，由 Process（sequential/hierarchical）驱动协作。

**为什么不适合当前项目**：
- 客服流程是"用户问订单 → 查数据库 → 格式化 → 校验事实"，不是"几个专家开会讨论"
- 无 checkpointer，session 丢失无法恢复
- 无法做精确的验证回环（validation loop）
- 内部路由是黑盒，出问题难以 debug 和审计

**适用场景**：内容生成（研究员写大纲 → 写手扩写 → 编辑润色）、竞品分析、多视角研究任务。

---

### 3. AutoGen / AG2 / Microsoft Agent Framework ❌ — 对话驱动，过于自由

**模型差异**：核心抽象是 GroupChat——多个 Agent 在一个聊天室里发消息，由一个 selector（可能是 LLM 或规则）决定谁下一步发言。

**为什么不适合**：
- **非确定性**：对话流向由 LLM 动态决定，不适合"用户查订单必须返回订单状态"这种刚性流程
- **2025 年已合并**：AutoGen 与 Semantic Kernel 于 2025 年 10 月合并为 **Microsoft Agent Framework**，目标 2026 Q1 GA，深度绑定 Azure/OpenAI
- 当前技术栈是 Python + 智谱 GLM + pgvector，不在微软生态内

**适用场景**：代码生成（Agent 写代码 → 另一个 Agent review）、研究辩论、需要 emergent collaboration 的开放性问题。

---

### 4. OpenAI Agents SDK ❌ — Vendor 锁定 + 模型不匹配

- **模型锁定**：SDK 的很多特性（tracing、guardrails、内置工具）深度依赖 OpenAI Platform。当前项目使用**智谱 GLM-4-Flash**，虽然兼容 OpenAI API 协议，但无法使用 SDK 的专有特性。
- **能力边界**：handoff 机制过于简单，无法表达复杂的 validation loop 和多条件路由。
- **状态 ephemeral**：默认无持久化，需自建。

**结论**：除非未来全面迁移到 OpenAI 模型家族，否则不适用。

---

### 5. Claude Agent SDK ❌ — 同样是 Vendor 锁定

- Anthropic 原生，对 Claude 的 tool-use、extended thinking、MCP（Model Context Protocol）支持最好
- **Claude-only**，当前 GLM 模型不匹配
- 如果未来考虑迁移到 Claude 3.7/4 系列做主模型，可重新评估其 MCP 工具生态

---

### 6. LlamaIndex Workflows ⚠️ — RAG 检索层最强，编排层可互补

**优势**：
- 事件驱动模型（`Event → Step → Event`）适合构建复杂的数据管道
- RAG 基础设施（indexing、hybrid search、reranking、self-correction）比 LangChain 原生实现更成熟
- 社区共识：**LlamaIndex 做检索，LangGraph 做编排** 是生产最佳实践

**劣势**：
- 编排能力弱于 LangGraph，无原生 time-travel 和 human-in-the-loop 中断
- 事件驱动在调试长链路时心智负担较高

**对当前项目的建议**：
- 当前 `retrieval.py` 已自研 dense+BM25+RRF+rerank，效果可控，无需替换
- **如果未来政策/知识文档膨胀到万级/十万级**，可评估把 retrieval layer 迁移到 LlamaIndex Workflows，但**外层编排仍保留 LangGraph**

---

### 7. PydanticAI ⚠️ — 轻量但能力边界明显

- 2025 年后兴起的"类型安全 Agent 框架"，强调 Python 类型系统贯穿始终
- 适合：**简单工具调用 + 强类型输出**（如"提取订单信息并返回 Pydantic 模型"）
- 不适合：复杂状态机、多步循环、持久化 session

**对当前项目**：可用来替换某些单个节点的结构化输出实现（如意图提取的 JSON schema 校验），但无法替代 LangGraph 的全局编排。

---

### 8. Semantic Kernel ⚠️ — .NET/Azure 生态专用

- 微软企业级 SDK，抽象为 Plugin + Planner
- Python 版本存在但生态薄弱，Planner 的决策过程偏黑盒
- 当前栈是 Python + 智谱 + pgvector，无 Azure 依赖，不匹配

**适用场景**：已有 .NET 后端、使用 Azure OpenAI、需要 SSO/企业安全合规的环境。

---

### 9. 原生实现（直接调 HTTP API）⚠️ — 控制最强，成本最高

**优势**：
- 零框架依赖，零 vendor 锁定
- 性能最优（无 LangChain/LangGraph 中间层开销）
- 可精确按业务需求实现状态机

**劣势**：
- 需自研：checkpointer、流式 parser、retry 逻辑、状态序列化、可视化调试、并发控制
- 当前项目已在 LangGraph 上投入大量工程（checkpoint、validation loop、cache intent 防过期、SSE streaming），迁移成本极高
- 框架的 bug 修正是社区共同承担，原生实现的 bug 全是维护成本

**结论**：LangGraph 目前不是性能或能力瓶颈，重写为原生没有 ROI。

---

## 四、明确取舍建议

### 短期（保持现状）

> **继续使用 LangGraph 作为核心编排框架，不要迁移。**

当前架构（确定性 DAG + validation loop + Postgres checkpoint + SSE streaming）与 LangGraph 高度匹配，且已积累大量定制逻辑（intent-aware cache、hybrid classifier、KG validation）。任何迁移都会带来 regression 风险。

### 中期（增强而非替换）

| 方向 | 策略 | 框架选择 |
|------|------|---------|
| **RAG 增强** | 文档量爆炸时，把检索子系统升级为 LlamaIndex Workflows | LlamaIndex + LangGraph 混用 |
| **多 Agent 协作** | 如果未来需要"政策专家 Agent + 订单专家 Agent + 售后专家 Agent"并行处理复杂问题 | 在 LangGraph 内用 **Supervisor Pattern**（无需引入 CrewAI） |
| **模型切换** | 如果未来切 Claude/GPT 作为主模型 | 评估对应 Vendor SDK 的 MCP 生态，但编排层仍保留 LangGraph |

### 长期（架构演进愿景）

```
┌─────────────────────────────────────────┐
│           外层编排：LangGraph              │  ← 不变：session、路由、校验、人工介入
│  sanitize → classify → [sub-graph] → ... │
├─────────────────────────────────────────┤
│           子系统可替换层                  │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ RAG 检索     │  │ 多 Agent 协作    │  │
│  │ LlamaIndex? │  │ Supervisor模式   │  │
│  │ (可选升级)   │  │ (LangGraph原生)  │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

---

## 五、一句话总结

> **LangGraph 是当前项目的"正确选择"，不是"过渡方案"。**
>
> 我们的需求是"精确控制的状态机"，不是"自由对话的多 Agent 系统"。CrewAI/AutoGen 在那条路上更强，但那不是我们走的路。继续深耕 LangGraph 的 checkpoint、human-in-the-loop 和 streaming，把单点能力（如 RAG）通过混合架构增强，而不是推翻重来。

---

## 参考

- [LangGraph vs CrewAI vs AutoGen vs Semantic Kernel vs LlamaIndex Workflows](https://theplanettools.ai/tools/langgraph) — 2026-05
- [Best Multi-Agent Frameworks in 2026: LangGraph, CrewAI, OpenAI SDK and Google ADK](https://gurusup.com/blog/best-multi-agent-frameworks-2026) — 2026-05
- [AI Agent Frameworks 2026: Production-Tested Ranking](https://alicelabs.ai/en/insights/best-ai-agent-frameworks-2026) — 2026-04
- [Top AI Agent Frameworks in 2026: A Production-Ready Comparison](https://pub.towardsai.net/top-ai-agent-frameworks-in-2026-a-production-ready-comparison-7ba5e39ad56d) — 2026-04
