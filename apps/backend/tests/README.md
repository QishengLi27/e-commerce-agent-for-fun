# Backend Tests (Not Yet Implemented)

## Overview

This directory will contain the test suite for the e-commerce support agent backend.

## Planned Structure

```
tests/
├── unit/
│   ├── test_memory.py
│   ├── test_resilience.py
│   ├── test_retrieval.py
│   └── test_db.py
├── integration/
│   ├── test_agent.py
│   └── test_api.py          # once API layer exists
├── fixtures/
│   ├── mock_orders.sql
│   └── sample_policies.txt
└── conftest.py
```

## Tech Stack

- **Runner**: `pytest`
- **Coverage**: `pytest-cov`
- **Mocking**: `unittest.mock` + `pytest-mock`
- **LLM mocking**: `langchain-core`'s `FakeListLLM` for deterministic agent tests
- **DB fixtures**: `pytest-postgresql` or Docker-based PostgreSQL service container

## Key Test Scenarios

1. **Memory**: persistence, truncation at max_history, JSON corruption recovery
2. **Resilience**: circuit breaker state transitions, retry backoff, fallback invocation
3. **Retrieval**: BM25 index rebuild, RRF fusion correctness, LLM re-rank mocking
4. **Agent**: tool routing (order vs policy), cache hit/miss, typo correction
5. **API**: endpoint status codes, request validation, error handling

## Running Tests

```bash
cd apps/backend
pytest tests/ -v --cov=backend --cov-report=term-missing
```
