#!/usr/bin/env python3
"""Run flexible scheduler lab scenarios from a JSON file.

The scenario runner is deliberately thin: every expanded case still launches
the canonical orchestrated runtime and trains/evaluates with real PyTorch workers. The
JSON file only describes how to vary the scheduler, topology, recovery, failure,
and stage-local update settings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.e5_recovery.scheduler_lab.run_scheduler_benchmark import aggregate_rows, flatten_run_result, write_csv


LAB_MODULE = "sg_exe_trainer.runtime.bpfree.orchestrated_runtime"
REPO_ROOT = Path(__file__).resolve().parents[3]

RUNNER_KEYS = {
    "case_args",
    "case_config_json",
    "description",
    "disabled",
    "name",
    "output_root",
    "seeds",
    "target_stage",
}

CLI_ORDER = [
    "model_name",
    "manifest",
    "eval_manifest",
    "output_dir",
    "num_chunks",
    "stage_devices",
    "topology",
    "workers",
    "standby_worker_ids",
    "limit",
    "eval_limit",
    "max_inflight",
    "scheduler_policy",
    "task_timeout_ms",
    "timeout_policy",
    "recovery_policy",
    "max_attempts",
    "worker_rejoin_delay_ms",
    "checkpoint_interval",
    "failure_mode",
    "failure_rate",
    "failure_stage",
    "failure_seq",
    "failure_attempt",
    "failure_point",
    "failure_delay_ms",
    "offline_stage",
    "offline_start_seq",
    "offline_end_seq",
    "train_chunks",
    "stage_train_strides",
    "stage_update_policy",
    "stage_update_queue_thresholds",
    "learning_rate",
    "grad_clip",
    "gradient_accumulation_steps",
    "optimizer",
    "sgd_momentum",
    "sgd_dampening",
    "sgd_weight_decay",
    "sgd_nesterov",
    "belief_transport_mode",
    "alpha",
    "label_smoothing",
    "trainable_mode",
    "lora_rank",
    "lora_alpha",
    "lora_targets",
    "lora_init_std",
    "dtype",
    "seed",
    "request_prefix",
    "progress_interval",
]

STAGE_MAP_KEYS = {
    "stage_train_strides",
    "stage_update_queue_thresholds",
}


@dataclass(frozen=True)
class ScenarioCase:
    name: str
    config: dict[str, Any]
    seeds: list[int]
    args: list[str]


class FormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Scenario file must contain a JSON object.")
    return data


def as_list(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ValueError(f"{name} must be a list.")


def parse_seeds(value: Any) -> list[int]:
    if value is None:
        return [20260531]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        seeds = [int(item) for item in value]
    else:
        raise ValueError("seeds must be an int, comma-separated string, or list.")
    if not seeds:
        raise ValueError("seeds must not be empty.")
    return seeds


def sanitize_name(name: str) -> str:
    cleaned = []
    for char in name:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._-")
    if not result:
        raise ValueError(f"Invalid empty case name from {name!r}")
    return result


def render_templates(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(FormatDict(variables))
    if isinstance(value, list):
        return [render_templates(item, variables) for item in value]
    if isinstance(value, dict):
        return {
            str(render_templates(key, variables)): render_templates(item, variables)
            for key, item in value.items()
        }
    return value


def merge_configs(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    if "train_limit" in merged:
        merged["limit"] = merged.pop("train_limit")
    return merged


def expand_axes(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes.keys())
    products = itertools.product(*(axes[key] for key in keys))
    return [dict(zip(keys, values)) for values in products]


def expand_sweep(
    *,
    defaults: dict[str, Any],
    scenario_seeds: list[int],
    sweep: dict[str, Any],
) -> list[ScenarioCase]:
    if not isinstance(sweep, dict):
        raise ValueError("Each sweep must be an object.")
    axes = sweep.get("axes", {})
    if not isinstance(axes, dict):
        raise ValueError("sweep.axes must be an object.")
    normalized_axes = {str(key): as_list(value, name=f"sweep axis {key}") for key, value in axes.items()}
    cases: list[ScenarioCase] = []
    name_template = str(sweep.get("name_template", sweep.get("name", "case")))
    for variables in expand_axes(normalized_axes):
        template_vars = dict(defaults)
        template_vars.update(variables)
        raw_name = render_templates(name_template, template_vars)
        if raw_name == name_template and variables:
            suffix = "_".join(f"{key}{value}" for key, value in variables.items())
            raw_name = f"{raw_name}_{suffix}"
        overrides = render_templates(sweep.get("overrides", {}), template_vars)
        config = merge_configs(defaults, overrides)
        seeds = parse_seeds(sweep.get("seeds", scenario_seeds))
        args = [str(item) for item in render_templates(sweep.get("args", []), template_vars)]
        cases.append(
            ScenarioCase(
                name=sanitize_name(str(raw_name)),
                config=config,
                seeds=seeds,
                args=args,
            )
        )
    return cases


def expand_cases(scenario: dict[str, Any]) -> list[ScenarioCase]:
    defaults = merge_configs({}, scenario.get("defaults", {}))
    scenario_seeds = parse_seeds(scenario.get("seeds"))
    cases: list[ScenarioCase] = []

    for case in as_list(scenario.get("cases", []), name="cases"):
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object.")
        if case.get("disabled", False):
            continue
        if "name" not in case:
            raise ValueError("Each case requires a name.")
        variables = dict(defaults)
        variables.update(case.get("variables", {}))
        overrides = render_templates(case.get("overrides", {}), variables)
        config = merge_configs(defaults, overrides)
        args = [str(item) for item in render_templates(case.get("args", []), variables)]
        cases.append(
            ScenarioCase(
                name=sanitize_name(str(render_templates(case["name"], variables))),
                config=config,
                seeds=parse_seeds(case.get("seeds", scenario_seeds)),
                args=args,
            )
        )

    for sweep in as_list(scenario.get("sweeps", []), name="sweeps"):
        cases.extend(expand_sweep(defaults=defaults, scenario_seeds=scenario_seeds, sweep=sweep))

    if not cases:
        raise ValueError("Scenario must define at least one enabled case or sweep.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.name in seen:
            duplicates.append(case.name)
        seen.add(case.name)
    if duplicates:
        raise ValueError(f"Duplicate expanded case names: {sorted(set(duplicates))}")
    return cases


def stage_map_to_cli(value: dict[Any, Any]) -> str:
    return ",".join(f"{stage}:{setting}" for stage, setting in sorted(value.items(), key=lambda item: int(item[0])))


def workers_to_cli(value: list[Any]) -> str:
    specs: list[str] = []
    for item in value:
        if isinstance(item, str):
            specs.append(item)
        elif isinstance(item, dict):
            stage = item.get("stage", item.get("stage_id"))
            device = item.get("device")
            if stage is None or device is None:
                raise ValueError(f"Worker object must contain stage/stage_id and device: {item}")
            specs.append(f"{stage}:{device}")
        else:
            raise ValueError(f"Unsupported worker spec: {item!r}")
    return ",".join(specs)


def value_to_cli(key: str, value: Any) -> str:
    if key in STAGE_MAP_KEYS and isinstance(value, dict):
        return stage_map_to_cli(value)
    if key == "workers" and isinstance(value, list):
        return workers_to_cli(value)
    if key == "stage_devices" and isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def append_cli_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    flag = "--" + key
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, value_to_cli(key, value)])


def scheduler_command(case: ScenarioCase, *, seed: int, output_dir: Path) -> list[str]:
    config = dict(case.config)
    if "manifest" not in config:
        raise ValueError(f"Case {case.name} is missing required config: manifest")
    config["output_dir"] = output_dir
    config["seed"] = seed
    config.setdefault("request_prefix", f"scenario-{case.name}-seed{seed}")

    cmd = [sys.executable, "-m", LAB_MODULE]
    known = [key for key in CLI_ORDER if key in config]
    unknown = sorted(key for key in config if key not in CLI_ORDER and key not in RUNNER_KEYS)
    for key in known + unknown:
        append_cli_arg(cmd, key, config[key])
    cmd.extend(case.args)
    return cmd


def run_case_seed(
    *,
    case: ScenarioCase,
    seed: int,
    output_root: Path,
    force: bool,
    dry_run: bool,
    target_stage: int,
) -> dict[str, Any] | None:
    output_dir = output_root / f"{case.name}_seed{seed}"
    summary_path = output_dir / "scheduler_summary.json"
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = scheduler_command(case, seed=seed, output_dir=output_dir)

    if dry_run:
        print(shlex.join(cmd), flush=True)
        return None

    if summary_path.is_file() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        elapsed_s = 0.0
    else:
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + shlex.join(cmd) + "\n")
            log_handle.flush()
            process = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed_s = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(f"Case {case.name} seed {seed} failed; see {log_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    row = flatten_run_result(
        policy=case.name,
        seed=seed,
        output_dir=output_dir,
        summary=summary,
        target_stage=target_stage,
        elapsed_s=elapsed_s,
    )
    row["case"] = case.name
    row["case_config_json"] = json.dumps(case.config, sort_keys=True)
    return row


def write_expanded_scenario(path: Path, cases: list[ScenarioCase], output_root: Path) -> None:
    expanded = {
        "output_root": str(output_root),
        "cases": [
            {
                "name": case.name,
                "seeds": case.seeds,
                "config": case.config,
                "args": case.args,
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(expanded, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduler lab scenarios from JSON.")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--only", default="", help="Comma-separated expanded case names to run.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = load_json(args.scenario)
    cases = expand_cases(scenario)
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}
        cases = [case for case in cases if case.name in selected]
        missing = selected.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"--only selected unknown cases: {sorted(missing)}")
    output_root = args.output_root or Path(scenario.get("output_root", "results/e5_recovery/raw/scheduler_scenarios/default"))
    target_stage = int(scenario.get("target_stage", scenario.get("defaults", {}).get("target_stage", 2)))
    output_root.mkdir(parents=True, exist_ok=True)
    write_expanded_scenario(output_root / "scheduler_scenario_expanded.json", cases, output_root)

    rows: list[dict[str, Any]] = []
    for case in cases:
        for seed in case.seeds:
            print(f"[scenario] running case={case.name} seed={seed}", flush=True)
            row = run_case_seed(
                case=case,
                seed=seed,
                output_root=output_root,
                force=args.force,
                dry_run=args.dry_run,
                target_stage=target_stage,
            )
            if row is None:
                continue
            rows.append(row)
            print(
                f"[scenario] done case={case.name} seed={seed} "
                f"train_tput={float(row['train_throughput_per_s']):.3f} "
                f"eval_acc={float(row['eval_choice_accuracy']):.4f} "
                f"eval_loss={float(row['eval_avg_loss']):.4f} "
                f"target_updates={row['target_stage_updates']}",
                flush=True,
            )

    if args.dry_run:
        return

    aggregate = aggregate_rows(rows)
    write_csv(output_root / "scheduler_scenario_runs.csv", rows)
    write_csv(output_root / "scheduler_scenario_summary.csv", aggregate)
    (output_root / "scheduler_scenario_summary.json").write_text(
        json.dumps({"runs": rows, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
