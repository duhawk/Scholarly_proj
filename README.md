# Scholarly — Academic RAG Pipeline

A production-grade Retrieval-Augmented Generation system built from scratch — no LangChain, no LlamaIndex. Answers natural language questions over academic PDFs with cited, grounded responses. Every component of the retrieval pipeline is implemented manually.

---

## Features

| Feature | Implementation |
|---|---|
| PDF Ingestion | PyMuPDF text extraction, sentence-aware chunking with overlap |
| Dense Retrieval | pgvector cosine similarity search via raw SQL |
| Sparse Retrieval | BM25 implemented from scratch (term frequency, IDF, length normalization) |
| Hybrid Retrieval | Reciprocal Rank Fusion merging dense + sparse ranked lists |
| HyDE | Claude generates a hypothetical answer, embeds it, uses that for vector search |
| Cross-Encoder Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks RRF candidates |
| Agentic Loop | Iterative retrieval — Claude decides if context is sufficient, refines query up to N times |
| Answer Generation | Claude `claude-sonnet-4-20250514` with inline citations `[Title, chunk N]` |
| SSE Streaming | `StreamingResponse(text/event-stream)` — streams tokens + citations + metadata |
| Background Ingestion | `BackgroundTasks` — `/ingest` returns `{job_id}` immediately, poll for status |
| Faithfulness Evaluation | One Claude API call per claim sentence — scores grounded vs. hallucinated claims |

---

## Tech Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn
- **Database:** PostgreSQL 16 + pgvector extension
- **ORM / Migrations:** SQLAlchemy (async), Alembic
- **Embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2` (384 dimensions)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Anthropic Claude API (`claude-sonnet-4-20250514`)
- **PDF Parsing:** PyMuPDF (fitz)
- **Containerization:** Docker + Docker Compose

---

## Project Structure

```
scholarly/
├── docker-compose.yml
├── Dockerfile
├── alembic.ini
├── requirements.txt
├── .env.example
├── README.md
│
├── app/
│   ├── main.py                        # FastAPI app factory, router registration, model warmup
│   ├── config.py                      # Pydantic BaseSettings — loads from .env
│   ├── database.py                    # Async SQLAlchemy engine + session factory
│   ├── models.py                      # ORM models + in-memory IngestionJob store
│   │
│   ├── ingestion/
│   │   ├── parser.py                  # PDF text extraction via PyMuPDF
│   │   ├── chunker.py                 # Fixed-size chunking with sentence-boundary snapping
│   │   └── pipeline.py               # Orchestrates: parse → chunk → embed → store
│   │
│   ├── retrieval/
│   │   ├── embedder.py                # SentenceTransformer singleton, unit-normalized embeddings
│   │   ├── vector_search.py           # pgvector <=> cosine similarity, raw SQL
│   │   ├── bm25.py                    # BM25 from scratch: BM25Index class + bm25_search()
│   │   ├── reranker.py                # CrossEncoder singleton, rerank() function
│   │   └── hybrid.py                  # HyDE → dense → BM25 → RRF → cross-encoder
│   │
│   ├── generation/
│   │   ├── prompt.py                  # System prompt, context block builder, sufficiency prompt
│   │   └── generator.py              # Claude API: generate_answer() + stream_answer() SSE
│   │
│   ├── agentic/
│   │   └── query_loop.py             # While loop with REFINE: detection, chunk deduplication
│   │
│   ├── evaluation/
│   │   ├── faithfulness.py           # Per-claim Claude verification, faithfulness score
│   │   ├── retrieval_quality.py      # Precision / recall / F1 metrics
│   │   └── runner.py                 # Runs eval and persists EvalResult to DB
│   │
│   └── routes/
│       ├── ingest.py                  # POST /ingest, GET /ingest/{job_id}
│       ├── query.py                   # POST /query — SSE streaming response
│       ├── documents.py               # GET /documents
│       └── eval.py                    # GET /eval?query_id=...
│
└── migrations/
    ├── env.py                         # Alembic async migration runner
    └── versions/
        └── 001_initial.py            # CREATE EXTENSION vector + all 4 tables
```

---

## Data Models

### `Document`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| filename | String | Original PDF filename |
| title | String | Extracted or user-provided |
| authors | String | Nullable |
| ingested_at | DateTime | |
| chunk_count | Integer | Total chunks produced |

### `Chunk`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| document_id | FK → Document | Cascade delete |
| content | Text | Raw chunk text |
| embedding | Vector(384) | pgvector column, unit-normalized |
| chunk_index | Integer | Position within document |
| token_count | Integer | Approximate token count |
| bm25_tokens | JSONB | Tokenized form for BM25 |
| page_num | Integer | Source page |

### `Query`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| question | Text | |
| answer | Text | Generated answer |
| citations | JSONB | `[{document_title, chunk_index, excerpt}]` |
| retrieved_chunk_ids | JSONB | All chunk IDs considered |
| iterations | Integer | Agentic loop count |
| faithfulness_score | Float | 0.0–1.0 |
| created_at | DateTime | |

### `EvalResult`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| query_id | FK → Query | Cascade delete |
| metric | String | e.g. `"faithfulness"` |
| score | Float | |
| detail | JSONB | Per-claim breakdown |
| evaluated_at | DateTime | |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```env
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/scholarly
ANTHROPIC_API_KEY=your-anthropic-key
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=8
MAX_ITERATIONS=3
RRF_K=60
BM25_K1=1.5
BM25_B=0.75
RERANKER_TOP_N=5
```

---

## Running the Project

### 1. Configure environment
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Start services
```bash
docker-compose up --build
```

### 3. Run database migrations
```bash
docker-compose exec api alembic upgrade head
```

### 4. Verify services are healthy
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## API Reference

### `POST /ingest`
Upload a PDF and start background ingestion.

```bash
curl -F file=@paper.pdf \
     -F title="My Paper" \
     -F authors="Jane Doe" \
     http://localhost:8000/ingest
```
```json
{"job_id": "uuid", "document_id": "uuid", "status": "pending"}
```

### `GET /ingest/{job_id}`
Poll ingestion job status.

```bash
curl http://localhost:8000/ingest/{job_id}
```
```json
{"job_id": "uuid", "status": "done", "result": {"chunk_count": 142, "ingestion_time_ms": 3241}}
```

### `POST /query`
Ask a question. Streams SSE tokens.

```bash
curl -N -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What estimation strategies are used to handle endogeneity?"}'
```
```
data: {"type": "token", "content": "Several"}
data: {"type": "token", "content": " papers"}
...
data: {"type": "citations", "citations": [{"document_title": "...", "chunk_index": 3, "excerpt": "..."}]}
data: {"type": "metadata", "faithfulness_score": 0.91, "iterations": 2, "chunks_retrieved": 8}
data: [DONE]
```

### `GET /documents`
List all ingested documents.

```bash
curl http://localhost:8000/documents
```

### `GET /eval?query_id={uuid}`
Retrieve faithfulness evaluation for a query.

```bash
curl "http://localhost:8000/eval?query_id={uuid}"
```
```json
{
  "query_id": "uuid",
  "faithfulness_score": 0.91,
  "claims": [
    {"claim": "Several papers use IV strategies", "supported": true, "justification": "..."},
    {"claim": "DiD designs exploit policy variation", "supported": true, "justification": "..."}
  ]
}
```

---

## System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Pipeline — POST /ingest"]
        A[PDF Upload] --> B[PyMuPDF Text Extraction]
        B --> C[Sentence-Aware Chunker<br/>512 tokens / 64 overlap]
        C --> D[Batch Embed<br/>all-MiniLM-L6-v2 384-dim]
        C --> E[BM25 Tokenizer<br/>lowercase, stopwords, JSONB]
        D --> F[(PostgreSQL + pgvector<br/>Document / Chunk / Embedding)]
        E --> F
    end

    subgraph Query["Query Pipeline — POST /query"]
        G[User Question] --> H[HyDE: Claude generates<br/>hypothetical answer]

        subgraph AgLoop["Agentic Loop — max 3 iterations"]
            H --> I[Embed hypothetical answer]
            I --> J[Dense Vector Search<br/>pgvector cosine similarity]
            G --> K[BM25 Sparse Search<br/>from-scratch inverted index]
            J --> L[Reciprocal Rank Fusion<br/>RRF k=60]
            K --> L
            L --> M[Cross-Encoder Rerank<br/>ms-marco-MiniLM-L-6-v2]
            M --> N{Claude: context sufficient?}
            N -- REFINE --> H
        end

        N -- Yes --> O[Answer Generation<br/>claude-sonnet-4-20250514]
        O --> P[Faithfulness Evaluation<br/>per-claim Claude verification]
        P --> Q[SSE Stream to Client<br/>tokens + citations + metadata]
    end

    F --> J
    F --> K

    subgraph Persistence["Persisted Results"]
        Q --> R[(Query record<br/>answer / citations / score)]
        P --> S[(EvalResult<br/>per-claim breakdown)]
    end
```

## Retrieval Pipeline Deep Dive

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  Agentic Query Loop                  │
│  ┌──────────────────────────────────────────────┐   │
│  │  1. HyDE: Claude generates hypothetical ans  │   │
│  │  2. Embed hypothetical answer                │   │
│  │  3. Dense vector search (pgvector <=>)       │   │
│  │  4. BM25 sparse search (from-scratch)        │   │
│  │  5. Reciprocal Rank Fusion merge             │   │
│  │  6. Cross-encoder rerank top candidates      │   │
│  └──────────────────────────────────────────────┘   │
│  Claude checks sufficiency → REFINE? → loop (max 3) │
└─────────────────────────────────────────────────────┘
     │
     ▼
Answer Generation (Claude claude-sonnet-4-20250514)
     │
     ▼
Faithfulness Evaluation (per-claim)
     │
     ▼
SSE Stream to Client
```

---

## BM25 Formula

Implemented from scratch in `app/retrieval/bm25.py`:

```
score(D, Q) = Σ IDF(qi) * f(qi,D) * (k1 + 1)
                        / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))

IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)

k1 = 1.5, b = 0.75
```
