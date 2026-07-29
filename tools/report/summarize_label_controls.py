import argparse
import csv
import json
import sys
from pathlib import Path


def phase_by_name(summary: dict, name: str) -> dict:
    for phase in summary.get("phases", []):
        if phase.get("phase") == name:
            return phase
    return {}


def infer_case(path: Path, root: Path) -> str:
    rel = path.parent.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2 and parts[-1].startswith("lr_"):
        return "/".join(parts[:-1])
    return "/".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize server label-control sweep summary.json files."
    )
    parser.add_argument("root", type=Path, help="Control matrix output directory.")
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        before = phase_by_name(summary, "eval_before")
        train = phase_by_name(summary, "train")
        after = phase_by_name(summary, "eval_after")
        case = infer_case(path, args.root)
        row = {
            "case": case,
            "learning_rate": summary.get("learning_rate"),
            "optimizer": summary.get("optimizer", "adamw"),
            "sgd_momentum": summary.get("sgd_momentum", 0.0),
            "sgd_dampening": summary.get("sgd_dampening", 0.0),
            "sgd_weight_decay": summary.get("sgd_weight_decay", 0.0),
            "sgd_nesterov": summary.get("sgd_nesterov", False),
            "grad_clip": summary.get("grad_clip", 1.0),
            "num_chunks": summary.get("num_chunks"),
            "train_chunks": ",".join(str(x) for x in summary.get("train_chunks", [])),
            "train_schedule": summary.get("train_schedule", "fifo"),
            "pipeline_window": summary.get("pipeline_window", 1),
            "alpha": summary.get("alpha"),
            "train_steps": summary.get("train_steps"),
            "eval_before_acc": before.get("choice_accuracy"),
            "eval_after_acc": after.get("choice_accuracy"),
            "delta_acc": (after.get("choice_accuracy", 0.0) - before.get("choice_accuracy", 0.0)),
            "eval_before_loss": before.get("avg_loss"),
            "eval_after_loss": after.get("avg_loss"),
            "delta_loss": (after.get("avg_loss", 0.0) - before.get("avg_loss", 0.0)),
            "train_acc": train.get("choice_accuracy"),
            "train_loss": train.get("avg_loss"),
            "summary": str(path),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No summary.json files found under {args.root}")

    fieldnames = list(rows[0].keys())
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
