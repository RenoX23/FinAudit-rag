# Phase 3 acceptance

Independent review passed the fixed-corpus evaluation deliverable on November 28, 2026. The reviewed evidence revision is 892b776554abc0fd0de7227c9e0f3a2791b497d1. The evaluated candidate answer implementation is 44d8ab92a403b10ac357206d02ffe5f602922b07.

| Acceptance check | Candidate result | Target |
|---|---:|---:|
| Numerical accuracy | 29/30, 96.7% | At least 90% |
| Correct abstention | 23/25, 92.0% | At least 90% |
| Answerable coverage | 92/95, 96.8% | At least 90% |
| Strict citation matches | 84/92, 91.3% | At least 90% |
| Local uncached pipeline p95 | 2.375 s | At most 10 s |
| Estimated provider inference per 100 queries | USD 0.00728 | At most USD 0.50 |

The reviewer regenerated the saved summary exactly, checked case and corpus digests, verified candidate retry accounting, and checked the reviewed source rows against their pinned PDF hashes. Eight alternate candidate citations were verified. The policy content proxy's missed deadline answer was also checked against the source. Source-review artifacts record the exact examined cases and the limits of that manual review.

The code review passed 59 tests including pgvector, Ruff, standard and expanded mypy checks, and isolated wheel and source builds. GitHub Actions passed for the reviewed evidence revision.

Two false answers and three answerable abstentions remain in the evaluation. The baseline's recorded inference usage is a lower bound because its original runner lost some retry usage. The cases informed implementation changes and form a regression benchmark. Fresh blind assessment, deployment latency, public hosting, and release verification remain pending.

The [evaluation report](../evals/results/2026-11-28/README.md) includes saved observations, usage records, automatic scores, source reviews, and failure examples. [CI evidence](https://github.com/RenoX23/FinAudit-rag/actions/runs/46161947677)
