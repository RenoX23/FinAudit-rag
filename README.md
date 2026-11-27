# FinAudit-RAG

FinAudit-RAG is a source-cited question answering project for financial filings and regulatory circulars. It is designed for an analyst who needs to check a reported number, compare two financial periods, or locate a requirement in its source document.

The first release will cover a fixed collection of public, digital-text Indian company filings and RBI or SEBI documents. A policy lookup should cite the PDF page. A calculation should show its cited inputs, period, units, and Python-computed result. A question without sufficient evidence should yield an explicit abstention.

**Status** — PDF ingestion, retrieval, and answering API. Benchmark results and the public demo are pending.

## Development setup

Use Python 3.11 or later. Create and activate a virtual environment with `python -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell.

```sh
python -m pip install ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest
python -m finaudit_rag.corpus data/corpus.json
python -m build
```

## Corpus and evaluation

The corpus manifest records the original publisher URL, PDF page count, and SHA-256 of every selected snapshot. Save the PDFs locally as `data/local/corpus/<id>.pdf`. Verify the files with:

```sh
python -m finaudit_rag.corpus data/corpus.json --pdf-dir data/local/corpus
```

To inspect extracted text and tables, export the verified snapshots as JSON Lines:

```sh
python -m finaudit_rag.ingestion data/corpus.json --pdf-dir data/local/corpus --output data/local/chunks-table-aware.jsonl
python -m finaudit_rag.ingestion data/corpus.json --pdf-dir data/local/corpus --output data/local/chunks-baseline.jsonl --mode baseline
```

Both exports preserve one-based PDF page references. The table-aware export keeps numeric columns in order and carries the applicable source headings with each group of rows. The parser skips lines with undecodable glyphs and ruled tables that contain those glyphs. PDFs remain in the ignored local directory.

### Search the corpus

Copy `.env.example` to `.env`, choose a local database password, and start pgvector with `docker compose up -d db`. Set `FINAUDIT_DATABASE_URL` to the connection URL for user `finaudit`, database `finaudit`, and `127.0.0.1:5433`. If that port is occupied, set `FINAUDIT_POSTGRES_PORT` and use the same port in the connection URL. URL-encode special characters in the password.

```sh
python -m finaudit_rag.retrieval index data/local/chunks-table-aware.jsonl --mode table-aware
python -m finaudit_rag.retrieval index data/local/chunks-baseline.jsonl --mode baseline
python -m finaudit_rag.retrieval search COLLECTION_ID "Find SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/80"
```

Use the ID printed by the corresponding index command in place of `COLLECTION_ID`. The baseline returns cosine matches. Table-aware retrieval combines BM25 and cosine ranks, then applies a CPU reranker. Model snapshots download on first use to the ignored model cache. Queries and evidence are processed locally by both retrieval models. The search command prints evidence and PDF page numbers.

The benchmark will run fixed-size chunking and vector-only retrieval against the upgraded pipeline on the same held-out questions. The [evaluation protocol](evals/README.md) defines the labels, measurements, and release criteria.

## Run the answering API

Start the database and index the table-aware export. Set `FINAUDIT_DATABASE_URL` and the collection ID printed by indexing in `FINAUDIT_COLLECTION_ID`. Add the HTTPS Chat Completions URL, model ID, and API key for your provider to the ignored local `.env`. The variable names are in [.env.example](.env.example).

```sh
python -m uvicorn finaudit_rag.api:create_app --factory --host 127.0.0.1 --port 8000
```

`GET /healthz` checks that the process is responding. Send a JSON object with a `question` field to `POST /query`. An answer contains page-cited source excerpts or a Python calculation and its cited operands. An abstention includes the reason. Calculations require annual figures labeled in INR crore with consistent source headings. Provider and database failures return structured 503 errors.

PAN, Aadhaar, email addresses, and labeled account numbers are masked before the model receives queries or excerpts. Set `FINAUDIT_REDIS_URL` to enable the optional cache. It keys entries to the exact question, indexed corpus, and model configuration. Queries with detected identifiers bypass caching.

## Build and release plan

1. Lock the corpus, evaluation protocol, and repository checks.
2. Preserve PDF structure and implement baseline and upgraded retrieval.
3. Deliver cited answers, deterministic calculations, and abstention through FastAPI.
4. Run the benchmark and document the measured results and failure cases.
5. Release the public web app with Docker, a tagged revision, and a redeploy and rollback runbook.

See [agent.md](agent.md) for phase reviews and [deployment constraints](docs/deployment.md) for the hosting shortlist and release checks.
