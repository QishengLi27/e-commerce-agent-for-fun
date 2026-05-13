## 0. Recently Implemented

- **ReAct Agent**: ✅ Switched to ReAct-style agent with tool calling and reasoning steps.
- **Conversation Memory**: ✅ Added persistent memory module for multi-turn conversations.
- **Project Roadmap**: ✅ Created this plans document for future development guidance.

## 1. High Priority Enhancements

- **User Interface**: Build a customer-facing web chat interface using FastAPI or Streamlit.
- **Conversation Memory**: Add persistent session memory so the agent can remember recent user context.
- **Dynamic Policy Updates**: Allow store policies to be updated without code changes.
- **Hybrid Retrieval**: Combine semantic retrieval with sparse search using BM25.
- **Logging and Metrics**: Track query types, response latency, cache hit rate, and feedback.

## 2. Medium Priority Improvements

- **Order History Context**: Personalize responses using order history and customer profile data.
- **Tool Expansion**: Add tools for inventory check, shipment ETA, returns eligibility, and refund policy.
- **Multi-turn Dialogue**: Support follow-up questions that use previous conversation context.
- **Feedback Loop**: Allow thumbs-up/down feedback and use it to improve prompts or reranking.
- **Authentication**: Add a user login/session layer for personalized support.

## 3. Agent Architecture & Quality Assurance

- **Validation Agent (LLM-as-a-Judge)**: Add a post-generation validation node that checks answers for factual accuracy, hallucination, tone compliance, and question relevance before delivery. Failed validations trigger a regeneration loop.
- **Sub-Agent / Expert Graphs**: Replace monolithic tool nodes with specialized sub-graphs (e.g., a `PolicyExpert` that performs multi-hop reasoning across multiple policy documents before concluding).
- **Multi-Agent Team (Supervisor Pattern)**: Scale to a supervisor graph that delegates to domain-specific sub-agents (order expert, policy expert, weather expert) each with independent tool sets and prompts.

## 4. Advanced Capabilities

- **Cloud Deployment**: Move to managed vector databases or cloud-hosted PostgreSQL with pgvector.
- **Fine-tuning**: Fine-tune the model with domain-specific e-commerce and policy data.
- **Multi-language Support**: Add translation support for international customers.
- **Human Escalation**: Add a fallback path to human support when the agent is unsure.
- **Compliance**: Add GDPR-style data handling and deletion of conversation memory.

## 5. Quick Wins for Prototype Development

- Add a simple CLI wrapper for interactive testing.
- Persist user-agent memory to a local JSON file.
- Add `plans/future_plans.md` to surface roadmap ideas for the AI.
- Create a reusable `memory.py` module for conversation history.
- Update `README.md` to point to the plans file and memory module.

## 6. Notes for AI Assistance

- Use this roadmap to suggest incremental implementation steps.
- Prefer building features in small, testable pieces.
- Keep the prototype lightweight and modular.
- Prioritize memory and retrieval improvements before moving to cloud deployment.
