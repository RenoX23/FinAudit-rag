# FinAudit-RAG repository instructions

## Setup and checks

- Setup with Python 3.11 or later and `python -m pip install ".[dev]"` inside a virtual environment.
- Run the foundation tool with `python -m finaudit_rag.corpus data/corpus.json`.
- Run the API with `python -m uvicorn finaudit_rag.api:create_app --factory` after configuring the database, collection, and model provider in `.env`.
- Test with `python -m pytest`.
- Lint with `python -m ruff check .` and `python -m ruff format --check .`.
- Type-check with `python -m mypy src tests`.
- Build with `python -m build`.

## Directory map

- `src/finaudit_rag/` contains Python modules.
- `tests/` contains automated checks.
- `data/corpus.json` pins the source PDFs. `data/local/` contains ignored downloads.
- `evals/` describes the evaluation and will contain its cases and artifacts.
- `docs/` documents technical and operating decisions.
- `.github/workflows/` contains CI.

## Conventions

Use pytest, Ruff, and typed public Python functions. Keep local notes, source PDFs, and credentials outside tracked files. Use small commits with imperative subjects and retain their history through phase integration. Apply the independent review process in [agent.md](agent.md) at each phase boundary.
