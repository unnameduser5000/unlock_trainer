from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ALLOWED_RUNNERS = {
    "bpfree-clean-schedule-v0",
    "bpfree-clean-schedule-v1-stage",
    "bpfree-clean-schedule-v2-runtime",
    "bpfree-clean-schedule-v3-body-send-head",
    "bpfree-clean-schedule-v4-perfmode",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--physical_batch_size", type=int, required=True)
    parser.add_argument("--microbatches", type=int, required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument(
        "--require_body_send_head",
        action="store_true",
        help="Require BODY_FORWARD -> FWD_SEND_POST -> LOCAL_HEAD_LOSS -> LOCAL_BACKWARD.",
    )
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text())

    expected_effective = args.physical_batch_size * args.microbatches
    expected_batches = args.records // args.physical_batch_size
    expected_windows = args.records // expected_effective

    assert summary["runner"] in ALLOWED_RUNNERS, summary["runner"]
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

    bad_numbering = []
    bad_order = []
    bad_bwd_p2p = []
    missing_body_split = []

    for path in sorted(args.run_dir.glob("train.stage*.actions.csv")):
        rows = list(csv.DictReader(path.open()))
        by = defaultdict(lambda: defaultdict(list))

        for r in rows:
            stage = int(r["stage_id"])
            batch_seq = int(r["global_batch_seq"])
            window_id = int(r["window_id"])
            mb_id = int(r["mb_id"])
            seq_start = int(r["seq_start"])
            action = r["action"]

            if "BWD_RECV" in action or "BWD_SEND" in action:
                bad_bwd_p2p.append((path.name, stage, batch_seq, action))

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
                            action,
                            batch_seq,
                            window_id,
                            mb_id,
                            seq_start,
                            expected_window,
                            expected_mb,
                            expected_seq_start,
                        )
                    )

            by[(stage, batch_seq)][action].append(r)

        for (stage, batch_seq), acts in by.items():
            if "FWD_SEND_POST" in acts and "LOCAL_BACKWARD" in acts:
                send_start = float(acts["FWD_SEND_POST"][0]["start_perf"])
                bwd_start = float(acts["LOCAL_BACKWARD"][0]["start_perf"])
                if send_start > bwd_start:
                    bad_order.append(
                        ("FWD_SEND_POST_AFTER_LOCAL_BACKWARD", path.name, stage, batch_seq)
                    )

            if args.require_body_send_head:
                if "FWD_SEND_POST" in acts:
                    required = ["BODY_FORWARD", "FWD_SEND_POST", "LOCAL_HEAD_LOSS", "LOCAL_BACKWARD"]
                    if not all(name in acts for name in required):
                        missing_body_split.append((path.name, stage, batch_seq, sorted(acts)))

                    else:
                        body_end = float(acts["BODY_FORWARD"][0]["end_perf"])
                        send_start = float(acts["FWD_SEND_POST"][0]["start_perf"])
                        head_start = float(acts["LOCAL_HEAD_LOSS"][0]["start_perf"])
                        bwd_start = float(acts["LOCAL_BACKWARD"][0]["start_perf"])

                        if not (body_end <= send_start <= head_start <= bwd_start):
                            bad_order.append(
                                (
                                    "BAD_BODY_SEND_HEAD_ORDER",
                                    path.name,
                                    stage,
                                    batch_seq,
                                    body_end,
                                    send_start,
                                    head_start,
                                    bwd_start,
                                )
                            )

    if bad_numbering:
        print("BAD numbering")
        for item in bad_numbering[:20]:
            print(item)
        raise SystemExit(1)

    if bad_bwd_p2p:
        print("BAD: BP-free trace contains backward P2P action")
        for item in bad_bwd_p2p[:20]:
            print(item)
        raise SystemExit(1)

    if missing_body_split:
        print("BAD: missing BODY_FORWARD/FWD_SEND_POST/LOCAL_HEAD_LOSS/LOCAL_BACKWARD split")
        for item in missing_body_split[:20]:
            print(item)
        raise SystemExit(1)

    if bad_order:
        print("BAD action order")
        for item in bad_order[:20]:
            print(item)
        raise SystemExit(1)

    print("PASS schedule trace checks")
    print(f"runner={summary['runner']}")
    print(f"records={args.records}")
    print(f"physical_batch_size={args.physical_batch_size}")
    print(f"microbatches={args.microbatches}")
    print(f"effective_batch={expected_effective}")
    print(f"update_windows={expected_windows}")
    print(f"scheduled_physical_batches={expected_batches}")


if __name__ == "__main__":
    main()
