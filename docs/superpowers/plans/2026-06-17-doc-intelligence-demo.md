# Document Intelligence Demo Project — Implementation Plan

**Goal:** A self-contained, runnable demo that reads a PDF (with tables and images), preserves formatting, extracts key information, and performs semantic reasoning — suitable for a GitHub portfolio piece.

**Design Principle:** One command runs the full pipeline. Every step prints what it did and why. The output is a side-by-side comparison (original PDF vs structured extraction) and a reasoning trace.

---

## Project Structure

```
doc-intelligence-demo/
├── README.md                        # What it does, how to run, screenshots
├── requirements.txt                 # Pinned versions
├── demo.py                          # python demo.py sample/contract.pdf — one-command demo
├── src/
│   ├── parsing.py                   # PDF → structured Markdown (MinerU + Marker)
│   ├── ingestion.py                 # Markdown → chunks (LlamaIndex)
│   ├── extraction.py                # Chunks → key entities (LLM CoT)
│   ├── reasoning.py                 # Entities → semantic answers (LLM)
│   └── display.py                   # Rich terminal output + HTML report
├── sample/
│   ├── contract.pdf                 # Sample PDF (2-3 pages, tables + text)
│   └── invoice.pdf                  # Second sample (scanned or image-heavy)
├── output/                          # Generated output per run
│   ├── contract_markdown.md
│   ├── contract_entities.json
│   ├── contract_reasoning.txt
│   └── contract_report.html
└── docs/
    └── pipeline.md                  # Architecture explanation (link to html diagram)
```

---

## Pipeline — 4 Stages

### Stage 1: PDF → Structured Markdown (parsing.py)

```
PDF → MinerU (Chinese + table-preserving) / Marker (English)
    → Markdown with tables, headings, reading order preserved
```

**What happens:**
1. Detect document language (Chinese → MinerU, English → Marker)
2. Parse PDF → Markdown with preserved tables
3. Save `output/{name}_markdown.md`
4. Print: "Parsed 3 pages → 2 tables, 8 sections, 1 image description"

**Why MinerU + Marker, not just one:**
- MinerU: best for Chinese documents, preserves reading order
- Marker: best for English technical docs, fastest
- Fallback: both installed, auto-detect language

### Stage 2: Markdown → Semantic Chunks (ingestion.py)

```
Markdown → LlamaIndex SentenceWindowNodeParser (window=5)
         → Metadata-enriched chunks stored in-memory
```

**What happens:**
1. Parse Markdown into sections (based on `##` headings)
2. SentenceWindowNodeParser chunks each section
3. Attach metadata: section_title, page_number, has_table (bool)
4. Print: "Created 15 chunks across 8 sections, 5 with tables"

### Stage 3: Chunks → Key Information (extraction.py)

```
Chunks → LLM Chain-of-Thought extraction → structured JSON
```

**What happens (per section type):**
- **Parties section:** Extract company names, signatories, dates
- **Payment section:** Extract amounts, currencies, payment terms, due dates
- **Obligation section:** Extract each obligation, responsible party, deadline
- **Table section:** Extract table as structured JSON

**CoT Prompt pattern:**
```
Step 1: Identify field type (company name / amount / date / obligation).
Step 2: Locate the exact text containing the value.
Step 3: Verify against surrounding context.
Step 4: Output with confidence score and evidence snippet.
```

**Output:** `{name}_entities.json`
```json
{
  "document_type": "Service Agreement",
  "parties": [
    {"name": "Acme Corp", "role": "supplier", "evidence": "Section 1, para 2"}
  ],
  "financial_terms": [
    {"description": "Monthly service fee", "amount": 150000, "currency": "USD",
     "frequency": "monthly", "term": "Net 30", "evidence": "Section 4.2"}
  ],
  "obligations": [
    {"party": "Supplier", "description": "Deliver monthly reports",
     "deadline": "5th of each month", "evidence": "Section 3.1"}
  ],
  "key_dates": [
    {"date_type": "effective", "value": "2024-06-01", "evidence": "Section 1"},
    {"date_type": "expiry", "value": "2026-05-31", "evidence": "Section 12"}
  ]
}
```

### Stage 4: Entities → Semantic Reasoning (reasoning.py)

```
Extracted entities → LLM reasoning → answers to business questions
```

**Built-in reasoning queries (no user input needed for demo):**

| Query | Reasoning Pattern |
|-------|------------------|
| "What are the total financial obligations?" | Sum all amounts, group by currency |
| "What are the risks in this contract?" | Flag: penalty clauses, auto-renewal, unlimited liability, missing termination |
| "Who owes what to whom?" | Cross-reference obligations with parties |
| "Are there any conflicting clauses?" | Compare different sections for contradictions |
| "What's the contract lifecycle?" | effective_date → key_milestones → expiry_date |

**Output:** `{name}_reasoning.txt` — natural language answers with evidence citations.

### Display (display.py)

**Two outputs:**

1. **Terminal (Rich):** Color-coded extraction summary + reasoning answers printed to console
2. **HTML Report:** Side-by-side view — original PDF (embedded) | structured Markdown | extracted entities | reasoning answers. Saved to `output/{name}_report.html`

---

## Sample PDF Requirements

Create 2 sample PDFs in `sample/`:

**1. contract.pdf** (2-3 pages, generated)
- Title page + Parties section
- Payment Terms table (columns: Item | Amount | Frequency | Due Date)
- Obligations section with bullet points
- Termination clause
- Contains a simple logo image

**2. invoice.pdf** (1 page, scanned look)
- Table of line items (Product | Qty | Unit Price | Total)
- Tax calculation at bottom
- Company letterhead with logo

*Generate these with Python (ReportLab or FPDF) so they're reproducible and don't have licensing issues.*

---

## Implementation Tasks

### Task 1: Project Scaffold (10 min)

- [ ] Create directory structure
- [ ] Write `requirements.txt`:
  ```
  magic-pdf>=0.2.0        # MinerU CLI
  marker-pdf>=1.0          # Alternative parser
  llama-index>=0.12        # Ingestion
  llama-index-llms-openai-like
  openai>=1.0              # LLM API (DeepSeek / GLM / OpenAI compatible)
  rich>=13.0               # Terminal display
  pydantic>=2.0            # Data models
  ```
- [ ] Create `sample/` with 2 generated PDFs using ReportLab
- [ ] Write `README.md` with overview + screenshot placeholder

### Task 2: Stage 1 — PDF Parsing (15 min)

- [ ] `src/parsing.py`:
  ```python
  def parse_pdf(filepath: str) -> str:
      """PDF → Markdown. Auto-detects Chinese (→ MinerU) vs English (→ Marker)."""
  ```
- [ ] Test: parse both sample PDFs, verify tables preserved in Markdown

### Task 3: Stage 2 — Ingestion (10 min)

- [ ] `src/ingestion.py`:
  ```python
  def ingest_markdown(md_text: str) -> list[Document]:
      """Markdown → LlamaIndex chunks with metadata."""
  ```

### Task 4: Stage 3 — Entity Extraction (25 min)

- [ ] `src/extraction.py`:
  ```python
  def extract_entities(chunks: list[Document]) -> dict:
      """CoT prompting per section type → structured JSON."""
  ```
- [ ] Define Pydantic models for extracted entities
- [ ] Implement section-type-aware CoT prompts (parties vs payment vs obligation)

### Task 5: Stage 4 — Semantic Reasoning (20 min)

- [ ] `src/reasoning.py`:
  ```python
  def reason(entities: dict, chunks: list[Document]) -> dict[str, str]:
      """Pre-built reasoning queries → answers with evidence."""
  ```
- [ ] 5 built-in reasoning queries (total obligations, risks, conflicts, lifecycle)

### Task 6: Display + Report (20 min)

- [ ] `src/display.py`: Rich terminal output
- [ ] HTML report template (CSS-styled, self-contained, one file)
- [ ] `demo.py` orchestrator that runs stages 1-6

### Task 7: Polish for GitHub (15 min)

- [ ] Screenshots of terminal output + HTML report
- [ ] `README.md` with: what it does, how to run, architecture diagram (ASCII), sample output
- [ ] `.env.example` for LLM API keys
- [ ] Push to GitHub

---

## Total Estimated Time: ~2 hours

---

## What This Demo Proves

| JD Requirement | How the demo proves it |
|---------------|----------------------|
| Complex document parsing | 2-tier parsing: MinerU (Chinese tables) + Marker (English speed) |
| Layout restoration | Side-by-side: original PDF vs structured Markdown with tables preserved |
| Key information extraction | CoT prompting per section type → structured JSON with evidence citations |
| Semantic reasoning | 5 built-in queries: obligations, risks, conflicts, lifecycle, party obligations |
| LLM workflow orchestration | 4-stage pipeline with explicit state passing (not ReAct) |
| AI Harness thinking | Structured output with confidence scores, evidence citations, audit trail |
| Production readiness | `.env` management, error handling, pinned versions, sample data |

---

## What NOT to include (keep it demo-sized)

- ❌ No Neo4j (graphs are explained in the architecture docs, this is a focused parsing demo)
- ❌ No FastAPI server (offline demo, one command)
- ❌ No Milvus/pgvector (in-memory chunk storage is enough for 2-3 page PDFs)
- ❌ No multi-agent (the 4-stage pipeline shows the orchestration pattern without complexity)
- ❌ No feature flags / A/B testing
