from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--physical_batch_size", type=int, required=True)
    parser.add_argument("--microbatches", type=int, required=True)
    parser.add_argument("--records", type=int, required=True)
    args = parser.parse_args()

    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())

    expected_effective = args.physical_batch_size * args.microbatches
    expected_batches = args.records // args.physical_batch_size
    expected_windows = args.records // expected_effective

    assert summary["runner"] in {"bpfree-clean-schedule-v0", "bpfree-clean-schedule-v1-stage", "bpfree-clean-schedule-v2-runtime"}, summary["runner"]
    assert summary["physical_request_batch"] == args.physical_batch_size
    assert summary["gradient_accumulation_steps"] == args.microbatches
    assert summary["microbatches"] == args.microbatches
    assert summary["effective_optimizer_batch"] == expected_effective
    assert summary["completed_records"] == args.records
    assert summary["optimizer_steps"] == expected_windows

    train_phase = next(p for p in summary["phases"] if p["phase"] == "train")
    assert train_phase["completed_records"] == args.records
    assert train_phase["optimizer_steps"] == expected_windows
    assert train_phase["batches"] == expected_batches

    bad_order = []
    bad_numbering = []

    for path in sorted(args.run_dir.glob("train.stage*.actions.csv")):
        rows = list(csv.DictReader(path.open()))

        by = defaultdict(dict)
        for r in rows:
            stage = int(r["stage_id"])
            batch_seq = int(r["global_batch_seq"])
            window_id = int(r["window_id"])
            mb_id = int(r["mb_id"])
            seq_start = int(r["seq_start"])

            if batch_seq >= 0:
                expected_window = batch_seq // args.microbatches
                expected_mb = batch_seq % args.microbatches
                expected_seq_start = batch_seq * args.physical_batch_size

                if (
                    window_id != expected_window
                    or mb_id != expected_mb
                    or seq_start != expected_seq_start
                ):
                    bad_numbering.append(
                        (
                            path.name,
                            stage,
                            r["action"],
                            batch_seq,
                            window_id,
                            mb_id,
                            seq_start,
                            expected_window,
                            expected_mb,
                            expected_seq_start,
                        )
                    )

            key = (stage, batch_seq)
            by[key].setdefault(r["action"], []).append(r)

        for (stage, batch_seq), acts in by.items():
            if "FWD_SEND_POST" in acts and "LOCAL_BACKWARD" in acts:
                send_start = float(acts["FWD_SEND_POST"][0]["start_perf"])
                bwd_start = float(acts["LOCAL_BACKWARD"][0]["start_perf"])
                if send_start > bwd_start:
                    bad_order.append((path.name, stage, batch_seq, send_start, bwd_start))

    if bad_numbering:
        print("BAD numbering")
        for item in bad_numbering[:20]:
            print(item)
        raise SystemExit(1)

    if bad_order:
        print("BAD: FWD_SEND_POST after LOCAL_BACKWARD")
        for item in bad_order[:20]:
            print(item)
        raise SystemExit(1)

    print("PASS clean-v0 trace checks")
    print(f"records={args.records}")
    print(f"physical_batch_size={args.physical_batch_size}")
    print(f"microbatches={args.microbatches}")
    print(f"effective_batch={expected_effective}")
    print(f"update_windows={expected_windows}")
    print(f"physical_microbatches={expected_batches}")


if __name__ == "__main__":
    main()
