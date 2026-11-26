# FinAudit-RAG

FinAudit-RAG is a source-cited question answering project for financial filings and regulatory circulars. It is designed for an analyst who needs to check a reported number, compare two financial periods, or locate a requirement in its source document.

The first release will cover a fixed collection of public, digital-text Indian company filings and RBI or SEBI documents. A policy lookup should cite the PDF page. A calculation should show its cited inputs, period, units, and Python-computed result. A question without sufficient evidence should yield an explicit abstention.

**Status** — Repository foundation. The query API, evaluation results, and public demo are pending.

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

The benchmark will run fixed-size chunking and vector-only retrieval against the upgraded pipeline on the same held-out questions. The [evaluation protocol](evals/README.md) defines the labels, measurements, and release criteria.

## Build and release plan

1. Lock the corpus, evaluation protocol, and repository checks.
2. Preserve PDF structure and implement baseline and upgraded retrieval.
3. Deliver cited answers, deterministic calculations, and abstention through FastAPI.
4. Run the benchmark and document the measured results and failure cases.
5. Release the public web app with Docker, a tagged revision, and a redeploy and rollback runbook.

See [agent.md](agent.md) for phase reviews and [deployment constraints](docs/deployment.md) for the hosting shortlist and release checks.
