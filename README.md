# ContractRAG — Contract Intelligence & Risk Review Platform

A RAG system for contract analysis built on real commercial contracts (CUAD dataset).
Covers the full RAG lifecycle: ingestion, hybrid retrieval, reranking, LLM generation,
inline evaluation, and a production-style FastAPI serving layer.

Built as a portfolio project demonstrating production ML engineering practices —
not a tutorial chatbot.

---

## Problem

Legal and procurement teams spend hours manually reviewing contracts to find risk clauses,
compare vendor terms against templates, and track renewal obligations. Generic LLMs cannot
reliably search across hundreds of contracts with traceable citations.

This system retrieves the right clause from the right contract and grounds every answer
in cited source text.

---

## Dataset

CUAD (Contract Understanding Atticus Dataset) — 510 real commercial contracts
annotated by legal experts across 41 clause types. Publicly available on HuggingFace.

- Contracts indexed: 50 (configurable via config.yaml)
- Chunks in Qdrant: 7,873
- Contract types: MSA, NDA, Employment, Lease, Distribution, License, Joint Venture

---

## Architecture

```text
PDF Contracts (CUAD)
|
[Ingestion Pipeline]
PyMuPDF parser -> clause-aware chunker -> regex metadata extractor
|
BGE-large-en-v1.5 embedder (dense 1024-dim) + BM25 sparse vectors
|
Qdrant (dense vector + sparse vector + metadata payload per chunk)
|
[Retrieval Pipeline]
Query -> hybrid search (RRF dense+sparse) -> BGE reranker (cross-encoder)
|
[Generation Pipeline]
Top-5 chunks -> Mistral-7B-Instruct (Ollama) -> structured JSON output
|
[Inline Evaluation]
Retrieval score + grounding score + faithfulness flag per response
|
[FastAPI]
/ask  /risk  /compare  /health-score  /evaluate
```

---

## Stack

| Layer        | Component                        | Role                                      |
|--------------|----------------------------------|-------------------------------------------|
| Parsing      | PyMuPDF, python-docx             | PDF and DOCX text extraction              |
| Chunking     | LangChain RecursiveTextSplitter  | Clause-boundary aware splits              |
| Metadata     | Regex extractors                 | Clause type, contract type, dates, parties|
| Embeddings   | BAAI/bge-large-en-v1.5           | Dense 1024-dim vectors, normalized        |
| Sparse       | BM25 via Qdrant sparse vectors   | Exact legal term matching                 |
| Vector DB    | Qdrant (Docker)                  | Hybrid dense+sparse, metadata filtering   |
| Reranker     | BAAI/bge-reranker-v2-m3          | Cross-encoder rescoring of top-20 chunks  |
| LLM          | Mistral-7B-Instruct via Ollama   | Structured JSON generation, local         |
| Evaluation   | Custom inline scorer + RAGAS     | Per-response quality scorecard            |
| API          | FastAPI + Pydantic               | Typed endpoints, structured responses     |

---

## Project Structure

```text
LEGALRAG/
├── contract_rag/
│   ├── ingestion/
│   │   ├── parser.py       # PDF/DOCX -> page-level text objects
│   │   ├── chunker.py      # Clause-aware recursive splitting
│   │   ├── metadata.py     # Regex extraction of clause/contract type, dates, parties
│   │   ├── embedder.py     # BGE-large batch embedder + query encoder
│   │   ├── indexer.py      # Qdrant upsert with idempotent manifest tracking
│   │   └── pipeline.py     # Orchestrates full ingestion flow
│   ├── retrieval/
│   │   ├── searcher.py     # Hybrid search via Qdrant query_points + RRF fusion
│   │   ├── reranker.py     # BGE cross-encoder reranker (transformers)
│   │   └── retriever.py    # Embed -> search -> rerank orchestrator
│   ├── generation/
│   │   ├── prompts.py      # Task-specific prompt templates (ask/risk/compare/health)
│   │   ├── llm.py          # Ollama HTTP client with timeout and health check
│   │   ├── generator.py    # JSON extraction, Pydantic validation, generation methods
│   │   ├── risk.py         # RiskAnalyser: 5 risk-focused queries, deduplication
│   │   ├── compare.py      # ContractComparer: clause-level diff generation
│   │   └── health.py       # HealthScorer: 0-100 weighted contract health score
│   ├── evaluation/
│   │   ├── scorer.py       # Inline retrieval + grounding + faithfulness scoring
│   │   ├── ragas_eval.py   # RAGAS pipeline over gold QA set (pending run)
│   │   └── gold_qa.json    # 50 hand-crafted QA pairs from CUAD annotations
│   ├── api/
│   │   ├── models.py       # Pydantic request/response schemas + QualityScorecard
│   │   ├── routes.py       # FastAPI route handlers with scorecard wiring
│   │   └── main.py         # App lifespan, component init, structured logging
│   ├── configs/
│   │   └── config.yaml     # Single source of truth for all parameters
│   └── data/
│       ├── raw/            # 510 CUAD PDFs
│       └── processed/      # manifest.json, ablation_results.json (pending)
├── logs/
│   └── app.log             # Structured JSON logs
├── .env
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker Desktop running
- Ollama installed: https://ollama.com/download
- Mistral pulled: `ollama pull mistral`

### Setup

```powershell
# Clone and activate
git clone https://github.com/YOUR_USERNAME/ContractRAG
cd ContractRAG
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install uv
uv pip install -r requirements.txt

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Run ingestion (indexes 50 contracts by default)
python -m contract_rag.ingestion.pipeline

# Start API
python -m uvicorn contract_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint             | Description                               |
|--------|----------------------|-------------------------------------------|
| GET    | /health              | Qdrant + Ollama health check              |
| POST   | /api/v1/ask          | Free-form Q&A with citations              |
| POST   | /api/v1/risk         | Risk clause detection and flagging        |
| POST   | /api/v1/compare      | Clause-level vendor vs template diff      |
| POST   | /api/v1/health-score | 0-100 contract health score               |
| POST   | /api/v1/evaluate     | RAGAS ablation evaluation (offline)       |

Every response includes a `quality_scorecard`:

```json
"quality_scorecard": {
  "retrieval_score": 0.356,
  "grounding_score": 0.71,
  "faithfulness_flag": true,
  "num_chunks_used": 5
}
```

---

## Example Responses

### /api/v1/ask

Request:
```json
{
  "question": "What are the liability clauses?",
  "mode": "hybrid"
}
```

Response (truncated):
```json
{
  "answer": "The liability clauses are found in Clause 15 (Liability and Indemnity) on pages 27-28...",
  "citations": [
    {"clause": "liability", "page": 27, "source_file": "AzulSa...Maintenance Agreement.pdf", "score": 0.41}
  ],
  "confidence": 1.0,
  "risk_level": "LOW",
  "quality_scorecard": {"retrieval_score": 0.356, "grounding_score": 0.71, "faithfulness_flag": true, "num_chunks_used": 5}
}
```

### /api/v1/risk

Request:
```json
{"cuad_id": "Array BioPharma Inc. - LICENSE...AGREEMENT.PDF"}
```

Response flags:
```json
{
  "risk_level": "HIGH",
  "flags": ["Imbalanced indemnification provisions", "Lack of clarity in some obligations"],
  "health_score": null
}
```

### /api/v1/compare

Returns clause-level diffs:
```json
{
  "clause_diffs": [
    {
      "clause_type": "liability",
      "vendor_summary": "Neither party is liable for consequential damages...",
      "template_summary": "Provider shall indemnify and hold harmless...",
      "risk_delta": "MORE_RISKY",
      "recommendation": "Negotiate to align vendor liability clause with template indemnification."
    }
  ]
}
```

---

## Retrieval — Ablation Study

Four retrieval configurations evaluated against a 50-pair gold QA set.
Production config is Hybrid + Reranker.

| Config            | Recall@10 | Context Precision | Faithfulness | Avg Latency |
|-------------------|-----------|-------------------|--------------|-------------|
| Dense only        | [PENDING] | [PENDING]         | [PENDING]    | [PENDING]   |
| BM25 only         | [PENDING] | [PENDING]         | [PENDING]    | [PENDING]   |
| Hybrid (RRF)      | [PENDING] | [PENDING]         | [PENDING]    | [PENDING]   |
| Hybrid + Reranker | [PENDING] | [PENDING]         | [PENDING]    | [PENDING]   |

Run evaluation:
```powershell
python -m contract_rag.evaluation.ragas_eval
```

---

## Inline Quality Scorecard

Every API response includes three lightweight evaluation signals computed
without any external calls:

| Signal            | Method                                              | Range  |
|-------------------|-----------------------------------------------------|--------|
| retrieval_score   | Average cross-encoder reranker score of top-k chunks| 0 to 1 |
| grounding_score   | Lexical overlap between answer tokens and context   | 0 to 1 |
| faithfulness_flag | True if grounding >= 0.55 and retrieval >= 0.3      | bool   |

---

## Key Engineering Decisions

**Hybrid retrieval over pure dense** — Legal contracts use precise terminology
(indemnification, force majeure, governing law). BM25 catches exact term matches
that dense retrieval misses. RRF fusion combines both without manual weight tuning.

**Clause-aware chunking** — Fixed-size chunking splits clauses mid-sentence.
Splitting on legal section boundaries (ARTICLE, Section N, numbered clauses)
keeps clause semantics intact per chunk.

**Cross-encoder reranker as separate stage** — Bi-encoder retrieval (BGE) is fast
but imprecise. Running a cross-encoder over top-20 candidates adds ~10s on CPU
but significantly improves top-5 precision for clause-specific queries.

**Idempotent ingestion** — manifest.json tracks indexed files. Re-running the
pipeline skips already-indexed contracts. Safe to run incrementally as new
contracts are added.

**Structured JSON output enforced** — All prompts instruct Mistral to return
only a JSON object matching a fixed schema. Pydantic validates every response.
Malformed outputs are caught and returned with a degraded-mode fallback.

**Config-driven** — Every parameter (chunk size, top-k, reranker threshold,
model name, collection name) lives in config.yaml. Zero hardcoded values.

---

## Limitations

- Mistral-7B on CPU generates at 2-5 tokens/sec. Each request takes 3-6 minutes.
  For production, run on a GPU or swap Mistral for a faster quantized model.
- Grounding score uses lexical overlap, not semantic similarity. It underscores
  paraphrased answers even when they are factually grounded.
- 50 contracts indexed for this demo. Qdrant scales to millions of vectors
  without architecture changes.
- RAGAS ablation table pending full run.

---

## Environment Variables

QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
