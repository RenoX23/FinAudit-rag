# Independent phase review

Use **Plan → Action → Scrutinize → Execute**. The implementer records the phase scope and evidence, then assigns an independent agent to review the final candidate commit. The reviewer inspects the diff and reruns the critical commands from a clean checkout or equivalent isolation. Integrate the reviewed revision or advance to the next phase after a recorded `PASS`.

| Phase | Review focus and required evidence |
|---|---|
| 0. Foundation | Confirm repo and CI access, track the evaluation protocol, and verify the manifest and downloaded checksums. Run lint, type checks, tests, and build from a clean checkout. Confirm that the hosting plan fits the public demo's constraints and that `project.md`, secrets, and local PDFs are ignored. |
| 1. Ingestion and retrieval | Check source page mapping, table headings and cells, repeated ingestion, empty and malformed pages, and exact circular identifiers. Compare baseline and upgraded retrieval using the same development queries. |
| 2. Answering and API | Exercise the API for a policy answer, numerical calculation, and abstention. Check citation validation, unit and period handling, provider failures, and redaction of outbound input and retrieved context. Check cache behavior for different scopes and versions. |
| 3. Evaluation | Spot-check held-out labels against original pages, confirm equivalent benchmark conditions and the documented thresholds, and regenerate the published scores from the saved run. |
| 4. Release | Use the public URL in a fresh browser session for the three demo flows. Check the release revision, clean-clone setup, health checks, CI, and a redeploy or rollback drill. |

Record the commit ID, verdict, commands and observed results, actionable findings, and any access or evidence gaps. Use `FAIL` when a criterion is broken and rerun it after repair. Use `BLOCKED` when access, evidence, or a required decision is missing. Keep the current phase open in either case.
