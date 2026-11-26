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
