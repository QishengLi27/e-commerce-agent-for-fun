# Prompt Management System — Design Doc

**Status:** Draft  
**Date:** 2026-06-18  
**Scope:** Centralized prompt storage, hot reload, version control, admin API for the e-commerce AI agent.

---

## Problem

Prompts are currently embedded as Python string constants across 6 files:

```python
# nodes.py
_REPLY_PROMPT = """You are a helpful..."""       # line 422
_STRICT_REPLY_PROMPT = """You are a helpful...""" # line 438
_VALIDATION_PROMPT = """You are an accuracy...""" # line 493

# keyword.py
_INTENT_PROMPT = """You are an intent..."""       # line 221

# llm_hybrid.py
_LLM_EXTRACT_PROMPT = """You are an intent..."""  # line 34

# semantic.py
_SEMANTIC_PROMPT = """You are an intent..."""     # line 60

# retrieval.py
prompt = ("""Rate how relevant...""")             # line 197

# nodes.py (summarization)
summary_prompt = ("""Summarize...""")             # line 340
```

**Problems:**
1. **Can't modify without redeploy** — every prompt change requires a code push + server restart
2. **No version history** — can't see who changed what or rollback
3. **No A/B testing** — can't compare prompt versions on the same query
4. **Scattered** — grep for `prompt =` across 6 files to find all prompts
5. **No validation** — `.format()` silently drops missing variables
6. **No preview** — can't test a prompt without running the full agent

---

## Design

### Architecture

```
┌─────────────────────────────────────────────────┐
│  prompts/templates/*.j2  (Version Controlled)   │
│  ┌───────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ reply.j2  │ │intent.j2 │ │ validation.j2  │  │
│  │ v3        │ │ v1       │ │ v1             │  │
│  └───────────┘ └──────────┘ └───────────────┘  │
│  ... 8 prompt files, one per template           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  prompts/loader.py                              │
│  ┌───────────────────────────────────────────┐ │
│  │ PromptLoader.load_all()                   │ │
│  │  · Reads all .j2 files                   │ │
│  │  · Parses JSON frontmatter → metadata    │ │
│  │  · Body → template string                │ │
│  │  · Returns dict[name, PromptTemplate]    │ │
│  └───────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  prompts/__init__.py  (Registry)                │
│  ┌───────────────────────────────────────────┐ │
│  │ _registry: dict[str, PromptTemplate]      │ │
│  │                                           │ │
│  │ init_prompts()        — startup load      │ │
│  │ get_prompt(name)      — fetch by name     │ │
│  │ reload_prompts()      — hot reload all    │ │
│  │ reload_one(name)      — hot reload one    │ │
│  └───────────────────────────────────────────┘ │
└────────────────┬───────┬────────────────────────┘
                 │       │
     ┌───────────┘       └───────────┐
     ▼                               ▼
┌──────────────┐            ┌──────────────────┐
│ Agent code   │            │ Admin API        │
│              │            │                  │
│ get_prompt(  │            │ GET  /admin/     │
│   "reply")   │            │      prompts     │
│   .format(   │            │                  │
│   ...)       │            │ POST /admin/     │
│              │            │      prompts/    │
│              │            │      reload      │
│              │            │                  │
│              │            │ POST .../preview │
└──────────────┘            └──────────────────┘
```

### Key Data Structure

```python
@dataclass(frozen=True)
class PromptTemplate:
    name: str              # "reply", "validation"
    description: str       # Human-readable purpose
    template: str          # The actual prompt text with {variables}
    version: int           # Monotonic, incremented on change
    author: str            # Who last modified
    model: str             # Which LLM this was tuned for
    temperature: float     # Recommended temperature
    variables: list[str]   # Expected format variables
    created_at: str        # ISO timestamp
    updated_at: str        # ISO timestamp

    def format(self, **kwargs) -> str: ...
    def validate(self) -> list[str]: ...  # Check all variables present
```

Frozen dataclass — immutable by design. "Modify" means create a new version.

### File Format: JSON Frontmatter + Template Body

```json
---
{
  "name": "reply",
  "description": "Generate the final reply to the user",
  "version": 3,
  "author": "qisheng",
  "variables": ["history", "question", "result"],
  "model": "glm-4-flash",
  "temperature": 0.0
}
---
You are a helpful e-commerce support agent...
{history}
User question: {question}
Relevant information: {result}
Your reply:
```

**Why JSON frontmatter + body, not YAML or pure JSON?**
- JSON is a strict subset of YAML, no parser ambiguity
- Frontmatter separators (`---`) work with any text editor
- Template body is plain text — `git diff` shows meaningful changes
- One file = one concern. Easy to find, edit, version, review

**Why `.j2` extension?**
- Signals "this is a template with variables" to editors and reviewers
- Not a Jinja2 dependency — uses Python `.format()` internally
- Easy to migrate to Jinja2 later if conditional logic is needed

### Hot Reload: How It Works

```
1. User edits reply.j2, changes version from 3 → 4
2. POST /admin/prompts/reload
3. PromptLoader reads reply.j2, parses frontmatter, creates PromptTemplate(v4)
4. _registry["reply"] = new PromptTemplate
5. Next request: get_prompt("reply") returns v4
6. Zero downtime. No server restart. In-flight requests finish with v3.
```

### Version Control: Git as Prompt History

```
$ git log --oneline apps/backend/prompts/templates/reply.j2
a1b2c3d feat: add strict guardrails to reply prompt (v3)
d4e5f6g fix: clarify citation requirement (v2)
g7h8i9j feat: initial reply prompt (v1)

$ git diff a1b2c3d d4e5f6g -- apps/backend/prompts/templates/reply.j2
# Shows exactly what text changed between v2 and v3
```

No database needed. Git is the version control. Prompt history = git history.

---

## Technology Choices

| Decision | Choice | Why not alternative |
|----------|--------|-------------------|
| **Storage** | Filesystem (.j2 files) | DB (PostgreSQL): adds migration overhead. Git already tracks files. 8 prompts × 2KB = trivial I/O |
| **Format** | JSON frontmatter + text body | YAML frontmatter: whitespace sensitivity causes bugs. Pure JSON file: template body becomes escape-hell |
| **Template engine** | Python `.format()` | Jinja2: more powerful but introduces dependency + sandbox concerns. `.format()` is built-in, fast, and sufficient for prompt templates |
| **Registry** | In-memory dict with hot reload | DB-backed: added latency per prompt lookup. Prompts are read-heavy, write-rarely — in-memory is ideal |
| **Versioning** | Git (file-based) | DB version table: duplicates git. Prompt files are already in git — every commit is a version snapshot |
| **API** | FastAPI router (same process) | Separate service: overkill for 8 prompts. Same process = zero network overhead for prompt lookup |
| **Data model** | Frozen dataclass | Pydantic BaseModel: heavier, validation not needed at this layer. Dataclass is simpler for immutable value objects |
| **Validation** | `validate()` method checks variable presence | Runtime-only: `.format()` throws KeyError if variable missing. Adding explicit `validate()` catches errors at load time, not in production |

---

## What This Design Does NOT Do

- **A/B testing** — comparing prompt versions on the same input requires a separate evaluation framework
- **Prompt optimization** — DSPy or automatic prompt tuning is out of scope
- **Multi-environment** — dev/staging/prod prompt variants are not needed at this scale
- **Access control** — prompt editing is admin-only, enforced at the API level

---

## File Map

```
apps/backend/prompts/
├── __init__.py           # Registry: get_prompt(), reload_prompts(), init_prompts()
├── models.py             # PromptTemplate (frozen dataclass) + PromptVersion
├── loader.py             # PromptLoader: filesystem → PromptTemplate objects
├── api.py                # Admin router: GET list, GET detail, POST reload, POST preview
└── templates/            # One .j2 file per prompt — git-tracked
    ├── reply.j2          # Generate final answer from tool results
    ├── reply_strict.j2   # Regenerate after validation failure
    ├── validation.j2     # LLM auditor: check grounding
    ├── intent.j2         # Classify user intent
    ├── llm_extract.j2    # LLM hybrid: intent + entity extraction
    ├── semantic.j2       # Semantic classifier with product context
    ├── rerank.j2         # Batch re-rank: score documents 1-10
    └── summarize.j2      # Compress conversation history
```
