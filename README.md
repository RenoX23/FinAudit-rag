# FinAudit-RAG

FinAudit-RAG is a source-cited question answering project for financial filings and regulatory circulars. It is designed for an analyst who needs to check a reported number, compare two financial periods, or locate a requirement in its source document.

The first release will cover a fixed collection of public, digital-text Indian company filings and RBI or SEBI documents. A policy lookup should cite the PDF page. A calculation should show its cited inputs, period, units, and Python-computed result. A question without sufficient evidence should yield an explicit abstention.

**Status** — PDF ingestion and retrieval. The answering API, evaluation results, and public demo are pending.

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

Both exports preserve one-based PDF page references. The table-aware export also keeps numeric columns in order and carries the source headings with each group of rows. PDFs remain in the ignored local directory.

### Search the corpus

Copy `.env.example` to `.env`, choose a local database password, and start pgvector with `docker compose up -d db`. Set `FINAUDIT_DATABASE_URL` to the connection URL for user `finaudit`, database `finaudit`, and `127.0.0.1:5433`. URL-encode special characters in the password.

```sh
python -m finaudit_rag.retrieval index data/local/chunks-table-aware.jsonl --mode table-aware
python -m finaudit_rag.retrieval index data/local/chunks-baseline.jsonl --mode baseline
python -m finaudit_rag.retrieval search COLLECTION_ID "Find SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/80"
```

Use the ID printed by the corresponding index command in place of `COLLECTION_ID`. The baseline returns cosine matches. Table-aware retrieval combines BM25 and cosine ranks, then applies a CPU reranker. Model snapshots download on first use to the ignored model cache. Queries and evidence are processed locally by both retrieval models. The search command prints evidence and PDF page numbers.

The benchmark will run fixed-size chunking and vector-only retrieval against the upgraded pipeline on the same held-out questions. The [evaluation protocol](evals/README.md) defines the labels, measurements, and release criteria.

## Build and release plan

1. Lock the corpus, evaluation protocol, and repository checks.
2. Preserve PDF structure and implement baseline and upgraded retrieval.
3. Deliver cited answers, deterministic calculations, and abstention through FastAPI.
4. Run the benchmark and document the measured results and failure cases.
5. Release the public web app with Docker, a tagged revision, and a redeploy and rollback runbook.

See [agent.md](agent.md) for phase reviews and [deployment constraints](docs/deployment.md) for the hosting shortlist and release checks.
