# Fixed-corpus evaluation, November 28, 2026

Both pipelines completed the same 120 cases over the pinned corpus. The candidate answer implementation is commit 44d8ab92a403b10ac357206d02ffe5f602922b07. These cases informed implementation changes and form a regression benchmark. A blind assessment on fresh cases is pending.

| Automatic metric | Vector baseline | Candidate |
|---|---:|---:|
| Correct calculations | 2/30 (6.7%) | 29/30 (96.7%) |
| Correct abstentions | 21/25 (84.0%) | 23/25 (92.0%) |
| Answerable coverage | 71/95 (74.7%) | 92/95 (96.8%) |
| Strict citation matches | 40/71 (56.3%) | 84/92 (91.3%) |
| Answer content matches | 32/95 (33.7%) | 91/95 (95.8%) |
| Service errors | 6 | 0 |
| Local uncached pipeline p95 | 1.063 s | 2.375 s |
| Reference-page context precision | 17.3% | 50.9% |

The candidate uses direct verified cells for supported financial requests. Policy answers use hybrid retrieval, reranking, and Groq openai/gpt-oss-20b. Both pipelines share the embedding model, provider model where used, five-chunk context limit, 2,400 characters per chunk, temperature zero, and request concurrency of one. Answer caching is disabled.

The context metric counts the selected source pages for direct financial answers and the passages supplied to the model for policy answers. The branch mix affects the aggregate. Report policy retrieval separately when comparing retrieval quality.

## Source review and scoring limits

The automatic scorer uses exact reference-page equality and a term proxy for free-text answers. Eight answered candidate cases cite valid alternate pages. The independently checked financial exceptions are calc-08, calc-11, fin-11, fin-12, fin-17, and fin-18. Policy exceptions reg-05 and reg-35 also cite valid alternate evidence. Reg-25 provides the correct deadline despite failing the term check because table columns interleave in the extracted text. The original labels and observations are retained.

Independent manual review of all 30 baseline financial lookups found 15 correct supported answers, nine wrong answers, two abstentions, and four errors. Seven of the correct answers use valid alternate or additional pages. The automatic content and citation proxies undercount some baseline answers. Manual policy review covered the five strict exceptions, with three supported answers and two answerable abstentions. Full manual faithfulness review of every returned policy answer is pending.

## Remaining failures

- Calc-26 abstains because the extracted TCS FY2024 consolidated statement lacks the profit-before-tax row.
- Reg-11 and reg-28 abstain despite answers existing in the source PDFs.
- Unanswerable-05 returns an answer for an unsupported Reliance revenue question.
- Unanswerable-13 returns dated KYC guidance for a question asking what is in force today.

## Latency and inference estimate

The p95 figures measure local pipeline requests with cached model artifacts and no answer cache. Initialization precedes those timers. A separate fresh-process API check took 4.4447 seconds to initialize and 1.8613 seconds for its first policy request. Independently measured total wall time was 6.3380 seconds, including test-client overhead. The model files were already cached. Public deployment and first-download startup latency are pending.

The baseline recorded 154,082 input tokens and 10,957 output tokens. The list-price recorded lower bound is USD 0.01237 per 100 queries.

The candidate recorded 101,617 input tokens and 3,722 output tokens. The list-price reported-response estimate is USD 0.00728 per 100 queries.

The estimate uses USD 0.075 per million input tokens and USD 0.30 per million output tokens, checked November 28, 2026. It excludes hosting and the local CPU cost. Actual account billing has not been inspected. The original baseline runner lost usage from some successful attempts followed by quota failures, so its complete inference cost is unknown. Candidate usage includes the persisted response ledger. [Groq model pricing](https://console.groq.com/docs/model/openai/gpt-oss-20b)

## Reproduce the automatic summary

```sh
python -m evals.score evals/results/2026-11-28/baseline.jsonl evals/results/2026-11-28/candidate.jsonl
```

Use summary.json for the regenerated counts. Candidate metadata records the exact code and corpus digests and model revisions. Baseline metadata uses the legacy format from acc0192. Source-review files identify the reviewed cases, original source checksums, and observation snapshot hashes. Those snapshots were taken during the run. The final observations preserve those reviewed rows. Normalize JSONL line endings to LF when verifying snapshot hashes across checkouts.
