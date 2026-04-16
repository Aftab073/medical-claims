# Medical Claims Processor — Production FastAPI + LangGraph Service

A multi-agent PDF processing pipeline that classifies, extracts, and structures
medical insurance claim data using LangGraph orchestration and Llama 3 (via Groq).

---

## Folder Structure

```
medical-claims/
├── app/
│   ├── main.py                  # FastAPI app, /api/process endpoint, validation
│   ├── graph/
│   │   ├── state.py             # ClaimState TypedDict — shared pipeline state
│   │   └── workflow.py          # LangGraph StateGraph definition
│   ├── agents/
│   │   ├── segregator.py        # Page classifier — routes pages to agents
│   │   ├── id_agent.py          # Extracts patient identity fields
│   │   ├── discharge_agent.py   # Extracts discharge summary fields
│   │   ├── bill_agent.py        # Extracts itemized billing fields
│   │   └── aggregator.py        # Merges all outputs into final JSON
│   └── utils/
│       ├── pdf_utils.py         # PyMuPDF page extraction + table detection
│       └── llm_utils.py         # LLM factory, prompts, JSON extraction helper
├── requirements.txt
├── .env.example
└── sample_response.json
```

---

## Setup

```bash
# 1. Clone / extract the project
cd medical-claims

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

### POST /api/process

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-2024-00892" \
  -F "file=@/path/to/claim.pdf"
```

### Python client

```python
import httpx

with open("claim.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/process",
        data={"claim_id": "CLM-2024-00892"},
        files={"file": ("claim.pdf", f, "application/pdf")},
        timeout=120,
    )
print(response.json())
```

---

## LangGraph Workflow — Detailed Explanation

```
START
  │
  ▼
[segregator_node]
  • Parses PDF bytes with PyMuPDF (page-by-page)
  • Sends each page's text to the LLM for classification
  • Maps each page to ONE of 9 document types
  • Groups pages: { "discharge_summary": [page4], "itemized_bill": [page9], ... }
  • Falls back to keyword heuristics if LLM confidence < 0.45
  │
  ├────────────────────┬────────────────────┐
  ▼                    ▼                    ▼
[id_agent]    [discharge_agent]      [bill_agent]
  (parallel)       (parallel)          (parallel)
  ↓                    ↓                    ↓
  Only receives    Only receives       Only receives
  identity_doc     discharge_summary   itemized_bill
  + claim_forms    pages               pages
  ↓                    ↓                    ↓
  patient_info    discharge_summary    billing
  written to       written to state    written to state
  state
  │
  └────────────────────┴────────────────────┘
                        │
                        ▼
               [aggregator_node]
  • Merges patient_info + discharge_summary + billing
  • Computes per-agent and overall confidence scores
  • Produces final JSON with _meta, quality report
                        │
                        ▼
                       END
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Parallel fan-out | The 3 extraction agents are independent — running in parallel cuts latency by ~2/3 vs sequential |
| Page-level routing | Segregator never passes the full PDF to agents — strict isolation prevents cross-contamination |
| Confidence threshold (0.45) | Pages below this use keyword fallback to prevent silent misclassification |
| `_merge_optional` reducer | LangGraph requires reducers on `Annotated` fields written by parallel nodes |
| `operator.add` on errors | Safely accumulates errors from all nodes without overwriting |
| Arithmetic validation | Bill agent recalculates totals from line items and flags discrepancies |
| Null > hallucination | All agents are instructed to return `null` for missing fields, never invent values |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required** — your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model to use for all LLM calls |
| `LLM_TEMPERATURE` | `0.0` | Keep at 0 for deterministic extraction |
| `LLM_MAX_RETRIES` | `3` | Retry count on JSON parse failure |
| `LLM_RETRY_DELAY` | `2.0` | Seconds between retries |
| `MAX_PDF_SIZE_MB` | `20` | Upload size limit |

---

## Validation & Safety Layers

1. **Input validation** — magic bytes check (`%PDF`), size limit, empty file guard
2. **Type validation** — LLM output checked against the 9 canonical document types
3. **Confidence thresholding** — pages below 0.45 confidence fall back to keyword classifier
4. **JSON retry** — up to 3 LLM attempts with fence-stripping and regex fallback
5. **Arithmetic validation** — bill agent cross-checks qty×rate vs amount, and subtotal+tax-discount vs total
6. **Date sanity** — discharge agent flags admission_date > discharge_date
7. **Error accumulation** — `processing_errors` list surfaces all non-fatal issues in the response