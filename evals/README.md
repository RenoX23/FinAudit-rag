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

New runs also record source-code and manifest digests, embedding and reranker revisions, and context limits. Resuming requires the same configuration and code bytes. Each HTTP response appends a usage record to the adjacent `.attempts.jsonl` file before response validation. This preserves reported token usage when a successful response is followed by a rate-limited retry. The scorer reports `usage_accounting_complete` and uses the attempt ledger for token totals when every saved case has a completion marker.

The original baseline run began at acc0192 with the earlier metadata format and is resumed with that revision's baseline runner. Its records are retained, and its usage accounting is incomplete because the earlier runner did not preserve successful attempts followed by quota failures. Treat its recorded token totals as a lower bound. Candidate runs with the revised accounting use separate output files.

Structured lookup observations preserve the amount, fiscal year, reporting scope, unit, and source citation. Their content score verifies the typed amount, year, scope, and unit against the reference labels. Free-text lookup responses use the documented term proxy and require manual review for year and scope before publishing a faithfulness claim.

The numerical preflight identified reporting metadata missing from the text extraction. The application recovers verified annual headers for four TCS statement pages and the standalone heading for Infosys page 264 from the exact pinned source snapshots. The preflight cases informed that change. Report them as a fixed regression benchmark and reserve fresh cases for a blind assessment.

### Pilot findings

The source-cell preflight verified 29 of 30 calculations. In the separate eight-case live pilot with evidence-selection-v3, the vector baseline matched 2 of 6 answerable cases and the candidate matched 3 of 6. Both pipelines correctly abstained on the two unanswerable cases and missed both calculations. The candidate's calculation context contained Infosys pages 31 or 32, while both reference calculations required page 264. The staged pilot resumed after provider rate limits. The complete comparison remains pending.

The second candidate run used the same eight questions and the same model with evidence-selection-v4 and hybrid-annual-tables-v1. Calculation retrieval restricted the candidate pool to annual tables using the requested issuer, fiscal years, reporting scope, and PDF page when specified. The reference calculation pages appeared in the top five for all 30 fixed cases. This measures retrieval only.

| Live pilot metric | Vector baseline | Updated candidate |
|---|---:|---:|
| Answer content matched | 2 / 6 | 5 / 6 |
| Answered cases with reference citations | 3 / 5 | 4 / 6 |
| Correct content and reference citations | 1 / 6 | 4 / 6 |
| Correct calculations | 0 / 2 | 2 / 2 |
| Correct abstentions | 2 / 2 | 2 / 2 |

The two financial lookup responses cited pages 31 and 74. Their excerpts presented numbers across multiple fiscal years and scopes, while the labeled answer requires a single standalone amount tied to its fiscal year. Both failed the reference citation check. The content term check accepted one of those excerpts, which illustrates why the score needs source review.

The candidate with hybrid-annual-tables-v2 handles recognized single-amount questions locally. In a warm, uncached API check against the existing index, all 30 financial lookups matched the reference amount, year, scope, and unit. Request latency at p95 was 4.68 ms on the local test host, measured through FastAPI's in-process test client. The provider endpoint was disabled for this check.

Twenty-six citations matched the original page labels. For the other four, the candidate used the consolidated statement on physical PDF page 277 of the Infosys annual report. Direct source review confirmed that page 277 and the labeled page 347 report the same FY2025 and FY2024 revenue and profit-before-tax values under the same scope and units. The automatic scorer retains the frozen reference pages. The runner records the verified cell's page as context for a structured lookup.
