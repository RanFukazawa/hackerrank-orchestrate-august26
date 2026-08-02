"""
Entry point for the Message Notification Router.

Orchestrates the full pipeline: loads all dataset/*.csv files via loaders.py,
resolves and inspects media via media.py, retrieves similar historical
messages and reaction patterns via retrieval.py, runs the injection/scam
guard via safety.py, routes each message to notify/digest/mute via router.py,
and writes the final predictions to dataset/output.csv.

The optional Gemini-backed reasoning layer (llm.py) is OFF by default --
RoutingEngine runs fully rule-based unless --use-llm is passed. This keeps
every run deterministic and free by default, and avoids spending the
configured GEMINI_API_KEY's tight daily call budget until explicitly asked.

Run from the terminal, e.g.:
    python code/main.py
    python code/main.py --dataset-dir dataset --output dataset/output.csv
    python code/main.py --use-llm
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loaders import Dataset, Message, load_dataset
from llm import LLMReasoner
from media import MediaResolver
from retrieval import EvidenceRetriever
from router import FeatureBundle, RoutingDecision, RoutingEngine
from safety import SafetyGuard

_OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the router run."""
    parser = argparse.ArgumentParser(description="Route dataset/messages.csv into notify/digest/mute predictions.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing messages.csv and the other dataset/*.csv files (default: dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write output.csv to (default: <dataset-dir>/output.csv)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable the optional Gemini-backed reasoning layer (requires GEMINI_API_KEY; tight daily call budget)",
    )
    return parser.parse_args()


def build_features(message: Message, dataset: Dataset, resolver: MediaResolver, retriever: EvidenceRetriever, guard: SafetyGuard) -> FeatureBundle:
    """Join context and run media/retrieval/safety for one message, assembling its FeatureBundle."""
    inspection = resolver.inspect(message)
    business = dataset.businesses.get(message.business_id) if message.business_id else None
    biz_history = (
        dataset.user_business_history.get((message.user_id, message.business_id)) if message.business_id else None
    )
    group = dataset.groups.get(message.group_id) if message.group_id else None
    membership = dataset.group_members.get((message.group_id, message.user_id)) if message.group_id else None
    user = dataset.users.get(message.user_id)
    evidence = retriever.find_similar(message, inspection.derived_text)
    safety_result = guard.is_unsafe(message, inspection.derived_text, business)

    return FeatureBundle(
        message=message,
        user=user,
        group=group,
        membership=membership,
        business=business,
        biz_history=biz_history,
        derived_text=inspection.derived_text,
        evidence=evidence,
        safety=safety_result,
    )


def route_messages(dataset: Dataset, dataset_dir: Path, engine: RoutingEngine) -> list[RoutingDecision]:
    """Build features and route every message in dataset.messages, returning one RoutingDecision per message."""
    resolver = MediaResolver(dataset, dataset_dir)
    retriever = EvidenceRetriever(dataset)
    guard = SafetyGuard()

    decisions = []
    for message in dataset.messages:
        features = build_features(message, dataset, resolver, retriever, guard)
        decisions.append(engine.route(features))
    return decisions


def write_output(decisions: list[RoutingDecision], output_path: Path) -> None:
    """Write one output.csv row per RoutingDecision, in the required column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_OUTPUT_COLUMNS)
        for decision in decisions:
            writer.writerow(
                [
                    decision.message_id,
                    decision.action,
                    decision.message_type,
                    decision.reason,
                    decision.confidence,
                    decision.evidence_field(),
                ]
            )


def main() -> None:
    """Load the dataset, route every message, and write predictions to output.csv."""
    args = parse_args()
    output_path = args.output or (args.dataset_dir / "output.csv")

    dataset = load_dataset(args.dataset_dir)
    reasoner = LLMReasoner() if args.use_llm else None
    engine = RoutingEngine(reasoner=reasoner)

    decisions = route_messages(dataset, args.dataset_dir, engine)
    write_output(decisions, output_path)

    print(f"Wrote {len(decisions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
