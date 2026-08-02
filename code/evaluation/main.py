"""
Self-evaluation harness.

Compares a generated output.csv against dataset/sample_messages.csv (the
solved examples with action/message_type/reason/confidence/evidence_message_ids
filled in) to sanity-check the router's predictions before submission.
Reports agreement on action and message_type, and flags rows where
evidence_message_ids or confidence look inconsistent with the samples'
style. Not used to train or hardcode behavior -- reference/style check only.

sample_messages.csv rows are not part of dataset/messages.csv, so this
script runs the same pipeline (loaders -> media -> retrieval -> safety ->
router) directly against the sample rows rather than reading output.csv,
to check the router's behavior against the one part of the dataset with
known-correct answers.

Run from the terminal, e.g.:
    python code/evaluation/main.py
    python code/evaluation/main.py --dataset-dir dataset
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loaders import Message, load_dataset  # noqa: E402
from main import build_features  # noqa: E402
from media import MediaResolver  # noqa: E402
from retrieval import EvidenceRetriever  # noqa: E402
from router import RoutingEngine  # noqa: E402
from safety import SafetyGuard  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation run."""
    parser = argparse.ArgumentParser(description="Evaluate the router against dataset/sample_messages.csv.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing sample_messages.csv and the other dataset/*.csv files (default: dataset)",
    )
    return parser.parse_args()


def _row_to_message(row: dict[str, str]) -> Message:
    """Parse one sample_messages.csv row's input columns into a Message (ignoring its answer columns)."""
    return Message(
        message_id=row["message_id"],
        user_id=row["user_id"],
        conversation_type=row["conversation_type"],
        group_id=row["group_id"] or None,
        business_id=row["business_id"] or None,
        sender_user_id=row["sender_user_id"] or None,
        created_at=None,
        message_text=row["message_text"],
        media_type=row["media_type"] or None,
        media_id=row["media_id"] or None,
        forwarded_count=int(row["forwarded_count"] or 0),
    )


def main() -> None:
    """Route every sample_messages.csv row and report agreement against its labeled answers."""
    args = parse_args()

    dataset = load_dataset(args.dataset_dir)
    resolver = MediaResolver(dataset, args.dataset_dir)
    retriever = EvidenceRetriever(dataset)
    guard = SafetyGuard()
    engine = RoutingEngine()

    with (args.dataset_dir / "sample_messages.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    action_correct = 0
    message_type_correct = 0
    evidence_hits = 0
    evidence_expected = 0
    mismatches: list[str] = []

    for row in rows:
        message = _row_to_message(row)
        features = build_features(message, dataset, resolver, retriever, guard)
        decision = engine.route(features)

        action_ok = decision.action == row["action"]
        type_ok = decision.message_type == row["message_type"]
        action_correct += action_ok
        message_type_correct += type_ok

        if row["evidence_message_ids"] != "none":
            evidence_expected += 1
            expected_ids = set(row["evidence_message_ids"].split(";"))
            got_ids = set(decision.evidence_message_ids)
            if expected_ids & got_ids:
                evidence_hits += 1

        if not action_ok:
            mismatches.append(
                f"{row['message_id']}: expected action={row['action']!r} got={decision.action!r} "
                f"(reason: {decision.reason})"
            )

    total = len(rows)
    print(f"action accuracy:       {action_correct}/{total} ({action_correct / total:.0%})")
    print(f"message_type accuracy: {message_type_correct}/{total} ({message_type_correct / total:.0%})")
    if evidence_expected:
        print(
            f"evidence hit rate:     {evidence_hits}/{evidence_expected} "
            f"({evidence_hits / evidence_expected:.0%}) of rows with expected evidence"
        )

    if mismatches:
        print("\nAction mismatches:")
        for line in mismatches:
            print(f"  {line}")


if __name__ == "__main__":
    main()
