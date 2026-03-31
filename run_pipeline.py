from __future__ import annotations

import argparse
import json

from pipeline import ActionTargetPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intent classification and target extraction.")
    parser.add_argument("--intent-model", required=True, help="Path to intent model.")
    parser.add_argument("--target-model", required=True, help="Path to target selector model.")
    parser.add_argument("--people-config", required=True, help="Path to people.json.")
    parser.add_argument(
        "--text",
        action="append",
        nargs="+",
        required=True,
        help="One or more text inputs. Repeat --text for multiple sentences.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive threshold for target selection.")
    args = parser.parse_args()

    pipeline = ActionTargetPipeline(
        intent_model_path=args.intent_model,
        target_model_path=args.target_model,
        people_config_path=args.people_config,
        target_threshold=args.threshold,
    )

    texts = [" ".join(parts).strip() for parts in args.text]
    outputs = [pipeline.predict(item) for item in texts if item]
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
