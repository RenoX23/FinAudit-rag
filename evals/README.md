# Evaluation protocol

Use 120 or more held-out cases over the pinned corpus. The first 120 target 35 regulatory lookups, 30 financial lookups, 30 calculations, and 25 unanswerable questions. Use a separate development set for parser changes, reranker settings, and abstention thresholds. Freeze the held-out set before scoring the release candidate.

Each case records its ID, query, category, and expected answer or abstention reason. For answerable cases, label the supporting document IDs and **one-based PDF page numbers**. Record the entity, reporting period, consolidation scope, currency, scale, and units when they affect the answer. For calculations, record the exact input cells, formula, expected result, and case-specific rounding tolerance. For unanswerable questions, document the missing period, entity, metric, or rule, or the unresolved ambiguity. Check that evidence is absent from the entire pinned corpus.

## Comparison

- **Baseline** uses fixed-size text chunks and cosine vector search.
- **Candidate** uses table-aware chunks, hybrid lexical and vector retrieval, reranking, and Python arithmetic on structured cited inputs.

Run both against the same immutable corpus and questions. Keep the embedding and generation models, retrieved context budget, and request concurrency fixed where they are shared. Disable the answer cache for the main run. Log corpus and dataset revisions, prompt and model versions, parameters, seed where supported, and retrieval settings. Record existing provider or network failures separately from evidence-based abstention. Count those failures as unsuccessful cases in the quality denominators.

| Metric | Release target and denominator |
|---|---|
| Citation correctness | At least 90% of answered, answerable cases cite sufficient source pages for all material claims. Score using the labeled evidence and manual review of any alternative page. |
| Numerical accuracy | At least 90% of all numeric cases match the expected result within the labeled tolerance and use the expected units. An abstention or service error fails the case. |
| Correct abstention | At least 90% of the unanswerable set returns an evidence-based abstention. |
| Answerable coverage | At least 90% of the answerable set returns a substantive answer. A service error fails the case. |
| Context precision and faithfulness | Report how much retrieved context is relevant and how many claims are supported by that context. Publish the scoring method and sample-check any model-judged results. |
| Latency and cost | Target warm uncached p95 at or below 10 seconds and provider inference cost at or below USD 0.50 per 100 queries. Report the cold path separately. Include all billed calls and report hosting costs separately from inference. |

Report raw counts alongside percentages and include representative failures. Version the results and generate the README comparison table from the saved run. Run the same cases against both pipelines after any correction to a reference label.

## Frozen cases and automatic scoring

The four JSONL files in this directory contain 120 unique cases. Lookup references use groups of acceptable answer terms. A case matches when every group has at least one term in the answer. Calculations compare the decimal result within the labeled tolerance and require the expected unit. The source pages for calculations also identify the original metric, scope, fiscal year, and amount of each operand.

The automatic scorer checks cited physical PDF pages against the reference list. Context precision is the proportion of supplied chunks from reference pages, including repeated chunks. Reference-grounded accuracy requires matching answer content and citations. It is an automatic proxy for faithfulness. Review additional or alternate valid evidence against the PDF before interpreting apparent errors.

Run the numerical preflight after exporting the structured chunks:

```sh
python -m evals.preflight data/local/chunks-table-aware.jsonl
```

The preflight supplies verified reference cells from the labeled page and exercises the application checks without retrieval ranking or a model request. It diagnoses whether the application can use the required table evidence before a live run. Its counts describe this controlled check.

Generate a summary from complete, saved observation files with:

```sh
python -m evals.score path/to/baseline.jsonl path/to/candidate.jsonl
```

## Running the comparison

Index both exports in pgvector. Save the baseline collection ID in `data/local/baseline-collection.txt` and set the table-aware ID in `FINAUDIT_COLLECTION_ID`. Configure the same provider model for both runs, then run each pipeline sequentially:

```sh
python -m evals.run baseline data/local/benchmark/baseline.jsonl
python -m evals.run candidate data/local/benchmark/candidate.jsonl
python -m evals.score data/local/benchmark/baseline.jsonl data/local/benchmark/candidate.jsonl
```

Use `--pilot` with separate filenames to exercise the runner on two cases per category. A run saves completed observations and its configuration so the same command can resume after a provider rate limit. The runner records usage reported by successful provider responses and measures per-question time before the pause between requests.

The numerical preflight identified reporting metadata missing from the text extraction. The application recovers verified annual headers for four TCS statement pages and the standalone heading for Infosys page 264 from the exact pinned source snapshots. The preflight cases informed that change. Report them as a fixed regression benchmark and reserve fresh cases for a blind assessment.
