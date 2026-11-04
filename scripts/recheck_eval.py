"""Re-evaluate saved answers without calling Ollama or PostgreSQL."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from eval_runner import (
    DEFAULT_DATASET,
    assess_answer,
    has_unapplied_setup,
    load_cases,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/evaluations/baseline.json",
    )
    parser.add_argument(
        "--output",
        default="results/evaluations/baseline_rechecked.json",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
    )
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()

    if source == output:
        parser.error("Input and output must be different files.")

    data = json.loads(source.read_text(encoding="utf-8"))
    cases = load_cases(Path(args.dataset).resolve())
    case_by_id = {str(case["id"]): case for case in cases}

    updated = 0

    for result in data.get("cases", []):
        case_id = str(result.get("id", ""))
        case = case_by_id.get(case_id)

        if case is None or result.get("verdict") == "error":
            continue

        answer = str(result.get("answer", ""))
        verdict, checks, reasons = assess_answer(case, answer)

        if has_unapplied_setup(case) and verdict != "fail":
            reasons.append(
                "Required test setup was not applied; human review is needed."
            )
            verdict = "review_needed"

        result["verdict"] = verdict
        result["checks"] = checks
        result["reasons"] = reasons
        updated += 1

    data["recheck"] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source_file": str(source),
        "cases_rechecked": updated,
        "method": "offline rule-based re-evaluation; no model calls",
    }
    data["summary"] = summarize(data.get("cases", []))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Cases rechecked: {updated}")
    print(json.dumps(data["summary"], indent=2))
    print(f"Saved: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())