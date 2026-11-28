"""Regenerate the benchmark summary from saved JSON Lines observations."""

import argparse
import json
from pathlib import Path

from finaudit_rag.corpus import load_manifest
from finaudit_rag.evaluation import Observation, load_cases, summarize


def main() -> None:
    """Score complete baseline and candidate observation files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus.json"))
    parser.add_argument("--cases", type=Path, default=Path("evals"))
    args = parser.parse_args()
    documents = {document.id: document for document in load_manifest(args.manifest)}
    cases = load_cases(args.cases, documents)
    results = {}
    for name, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        observations = [
            Observation.model_validate_json(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results[name] = summarize(cases, observations)
        ledger = source.with_suffix(".attempts.jsonl")
        attempts = (
            [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if ledger.exists()
            else []
        )
        completed = {item["case_id"] for item in attempts if item["event"] == "complete"}
        complete = completed == {item.id for item in observations}
        results[name]["usage_accounting_complete"] = complete
        if complete:
            results[name]["input_tokens"] = sum(item.get("input_tokens", 0) for item in attempts)
            results[name]["output_tokens"] = sum(item.get("output_tokens", 0) for item in attempts)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
