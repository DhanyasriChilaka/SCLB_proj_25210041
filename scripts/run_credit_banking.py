"""Run the complete credit-banking experiment protocol.

The default invocation performs truthful-only radius calibration followed by
the 3 x 7 x 4 x 50 primary experiment. Results are checkpointed as JSONL, so
an interrupted run can be resumed safely. Use ``--quick`` for a smoke-sized
version and ``--stage`` to run calibration or evaluation separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.credit_banking import run_credit_banking_condition
from src.algorithms.bioptgtm import (
    BidirectionalOptGTM,
    WindowedBidirectionalOptGTM,
)
from src.algorithms.linucb_per_arm import PerArmLinUCB
from src.algorithms.optgtm import OneSidedOptGTM
from src.metrics import build_paired_summary
from src.plotting import (
    CONDITION_LABELS,
    MECHANISM_LABELS,
    plot_attacker_trigger_trace,
    plot_calibration,
    plot_grouped_estimates,
)
from src.strategies import (
    CreditBankerArm,
    OverReporterArm,
    PullCountCreditBankerArm,
    TruthfulArm,
    UnderReporterArm,
)


RADIUS_GRID = (0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
QUICK_RADIUS_GRID = (0.10, 0.20, 0.30)
MECHANISMS = ("linucb", "one_sided", "bidirectional", "windowed_w200")
ELIMINATION_MECHANISMS = MECHANISMS[1:]
CONDITIONS = (
    "truthful",
    "under_reporter",
    "over_reporter",
    "time_delayed",
    "time_banker",
    "pull_delayed",
    "pull_banker",
)


def strategy_factory(name, t0, bank_pulls):
    factories = {
        "truthful": TruthfulArm,
        "under_reporter": lambda: UnderReporterArm(delta=0.10),
        "over_reporter": lambda: OverReporterArm(delta=0.15),
        "time_delayed": lambda: CreditBankerArm(
            a_minus=0.0, a_plus=0.15, T0=t0
        ),
        "time_banker": lambda: CreditBankerArm(
            a_minus=0.10, a_plus=0.15, T0=t0
        ),
        "pull_delayed": lambda: PullCountCreditBankerArm(
            a_minus=0.0, a_plus=0.15, bank_pulls=bank_pulls
        ),
        "pull_banker": lambda: PullCountCreditBankerArm(
            a_minus=0.10, a_plus=0.15, bank_pulls=bank_pulls
        ),
    }
    return factories[name]


def learner_factory(name, radius_scale):
    def build(env):
        if name == "linucb":
            return PerArmLinUCB(K=env.K, d=env.d, alpha=1.0)
        common = dict(
            K=env.K,
            d=env.d,
            T=env.T,
            sigma=env.sigma,
            radius_scale=radius_scale,
        )
        if name == "one_sided":
            return OneSidedOptGTM(**common)
        if name == "bidirectional":
            return BidirectionalOptGTM(**common)
        if name == "windowed_w200":
            return WindowedBidirectionalOptGTM(**common, window=200)
        raise ValueError(f"Unknown mechanism: {name}")

    return build


def scenarios(horizon):
    base = {"T": horizon, "K": 6, "d": 5, "sigma": 0.1}
    return {
        "controlled_strong": {
            "attacker": 0,
            "attacker_role": "highest_quality",
            "env_kwargs": {**base, "context_mode": "controlled"},
        },
        "controlled_weak": {
            "attacker": 5,
            "attacker_role": "lowest_quality",
            "env_kwargs": {**base, "context_mode": "controlled"},
        },
        "iid_robustness": {
            "attacker": 0,
            "attacker_role": "exchangeable",
            "env_kwargs": {**base, "context_mode": "iid"},
        },
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def protocol_hash(protocol):
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, default=json_default).encode("utf-8")
    ).hexdigest()[:12]


def mean_interval(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return dict(n=0, mean=None, se=None, ci_low=None, ci_high=None,
                    median=None, q1=None, q3=None)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
    }


def wilson_interval(values):
    values = [bool(value) for value in values if value is not None]
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, se=None, ci_low=None, ci_high=None)
    successes = sum(values)
    p = successes / n
    z = 1.96
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "mean": p,
        "se": math.sqrt(p * (1 - p) / n),
        "ci_low": max(0.0, center - half),
        "ci_high": min(1.0, center + half),
    }


def aggregate(rows, group_keys, continuous, rates):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, members in sorted(groups.items()):
        result = dict(zip(group_keys, group))
        result["runs"] = len(members)
        for field in continuous:
            stats = mean_interval([row.get(field) for row in members])
            result.update({f"{field}_{key}": value for key, value in stats.items()})
        for field in rates:
            stats = wilson_interval([row.get(field) for row in members])
            result.update({f"{field}_{key}": value for key, value in stats.items()})
        output.append(result)
    return output


def calibration(args, output_dir, run_id):
    raw_path = output_dir / "raw" / "calibration_truthful_runs.jsonl"
    existing = load_jsonl(raw_path)
    completed = {
        (r["run_id"], r["mechanism"], r["radius_scale"], r["seed"])
        for r in existing
    }
    seeds = range(1000, 1000 + args.calibration_seeds)
    scales = QUICK_RADIUS_GRID if args.quick else RADIUS_GRID

    for mechanism in ELIMINATION_MECHANISMS:
        for scale in scales:
            for seed in seeds:
                key = (run_id, mechanism, scale, seed)
                if key in completed:
                    continue
                summary = run_credit_banking_condition(
                    seed=seed,
                    learner_factory=learner_factory(mechanism, scale),
                    strategy_factory=TruthfulArm,
                    attacker=0,
                    env_kwargs={
                        "T": args.horizon,
                        "K": 6,
                        "d": 5,
                        "sigma": 0.1,
                        "context_mode": "controlled",
                    },
                )
                row = {
                    **summary,
                    "run_id": run_id,
                    "mechanism": mechanism,
                    "radius_scale": scale,
                }
                append_jsonl(raw_path, row)
                existing.append(row)
                completed.add(key)

    current = [
        row for row in existing
        if row.get("run_id") == run_id
        and row["mechanism"] in ELIMINATION_MECHANISMS
        and row["radius_scale"] in scales
    ]
    aggregated = aggregate(
        current,
        ("mechanism", "radius_scale"),
        continuous=("false_eliminations", "cumulative_regret"),
        rates=("any_false_elimination", "complete_stop"),
    )
    for row in aggregated:
        row["run_false_elimination_rate"] = row["any_false_elimination_mean"]
        row["total_false_eliminations"] = int(
            sum(
                item["false_eliminations"]
                for item in current
                if item["mechanism"] == row["mechanism"]
                and item["radius_scale"] == row["radius_scale"]
            )
        )

    selected = {}
    selection_details = {}
    for mechanism in ELIMINATION_MECHANISMS:
        subset = sorted(
            (row for row in aggregated if row["mechanism"] == mechanism),
            key=lambda row: row["radius_scale"],
        )
        safe = [row for row in subset if row["total_false_eliminations"] == 0]
        if safe:
            boundary = safe[0]["radius_scale"]
            index = list(scales).index(boundary)
            chosen = scales[min(index + 1, len(scales) - 1)]
            status = "selected_one_grid_step_above_smallest_safe_scale"
        else:
            chosen = max(scales)
            status = "no_zero_false_elimination_scale_use_grid_max"
        selected[mechanism] = float(chosen)
        selection_details[mechanism] = {
            "selected_radius_scale": float(chosen),
            "status": status,
        }

    write_csv(output_dir / "tables" / "calibration_radius_sensitivity.csv", aggregated)
    write_json(
        output_dir / "selected_radius_scales.json",
        {"run_id": run_id, "scales": selected, "details": selection_details},
    )
    plot_calibration(
        aggregated,
        selected,
        output_dir / "figures" / "calibration_false_elimination_rate.png",
    )
    return selected


def load_selected_scales(output_dir, expected_calibration_run_id):
    path = output_dir / "selected_radius_scales.json"
    if not path.exists():
        raise FileNotFoundError(
            "No calibrated radius scales were found. Run "
            "`run_credit_banking.py --stage calibration` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != expected_calibration_run_id:
        raise ValueError(
            "The saved radius calibration was produced by a different "
            "experiment configuration. Re-run the calibration stage."
        )
    return payload["scales"]


def primary_runs(args, output_dir, run_id, selected_scales):
    raw_path = output_dir / "raw" / "primary_episode_metrics.jsonl"
    rows = load_jsonl(raw_path)
    completed = {
        (r["run_id"], r["scenario"], r["mechanism"], r["condition"], r["seed"])
        for r in rows
    }
    seeds = range(args.evaluation_seeds)
    t0 = int(round(0.20 * args.horizon))
    bank_pulls = 10 if args.quick else 100

    for scenario_name, scenario in scenarios(args.horizon).items():
        for mechanism in MECHANISMS:
            scale = 1.0 if mechanism == "linucb" else selected_scales[mechanism]
            for condition in CONDITIONS:
                for seed in seeds:
                    key = (run_id, scenario_name, mechanism, condition, seed)
                    if key in completed:
                        continue
                    summary = run_credit_banking_condition(
                        seed=seed,
                        learner_factory=learner_factory(mechanism, scale),
                        strategy_factory=strategy_factory(condition, t0, bank_pulls),
                        attacker=scenario["attacker"],
                        env_kwargs=scenario["env_kwargs"],
                    )
                    row = {
                        **summary,
                        "run_id": run_id,
                        "scenario": scenario_name,
                        "attacker_role": scenario["attacker_role"],
                        "context_mode": scenario["env_kwargs"]["context_mode"],
                        "mechanism": mechanism,
                        "condition": condition,
                        "positive_attack_gain": None,
                        "eliminated_under": summary["elimination_side"] == "under",
                        "eliminated_over": summary["elimination_side"] == "over",
                    }
                    append_jsonl(raw_path, row)
                    rows.append(row)
                    completed.add(key)

    current = [row for row in rows if row.get("run_id") == run_id]
    write_csv(output_dir / "tables" / "episode_metrics.csv", current)
    return current, t0, bank_pulls


def paired_tables(rows):
    lookup = {
        (r["scenario"], r["mechanism"], r["condition"], r["seed"]): r
        for r in rows
    }
    paired = []
    banking = []
    scenarios_present = sorted({row["scenario"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})

    for scenario in scenarios_present:
        for mechanism in MECHANISMS:
            for seed in seeds:
                truthful = lookup[(scenario, mechanism, "truthful", seed)]
                for condition in CONDITIONS[1:]:
                    attack = lookup[(scenario, mechanism, condition, seed)]
                    result = build_paired_summary(truthful, attack)
                    result.update({
                        "scenario": scenario,
                        "mechanism": mechanism,
                        "condition": condition,
                        "comparison": "versus_truthful",
                        "positive_attack_gain": result["attack_gain"] > 0,
                    })
                    paired.append(result)

                for banking_type, baseline_name, attack_name in (
                    ("time", "time_delayed", "time_banker"),
                    ("pull", "pull_delayed", "pull_banker"),
                ):
                    result = build_paired_summary(
                        lookup[(scenario, mechanism, baseline_name, seed)],
                        lookup[(scenario, mechanism, attack_name, seed)],
                    )
                    result.update({
                        "scenario": scenario,
                        "mechanism": mechanism,
                        "condition": attack_name,
                        "banking_type": banking_type,
                        "comparison": "banker_versus_delayed_control",
                        "positive_attack_gain": result["attack_gain"] > 0,
                    })
                    banking.append(result)
    return paired, banking


def aggregate_and_write(output_dir, episodes, paired, banking):
    group = ("scenario", "mechanism", "condition")
    episode_agg = aggregate(
        episodes,
        group,
        continuous=(
            "attacker_pulls",
            "cumulative_regret",
            "false_eliminations",
            "elimination_pull",
            "elimination_round",
            "detection_delay_global",
            "detection_delay_pull",
            "detection_delay_global_signed",
            "detection_delay_pull_signed",
            "post_switch_detection_delay_rounds",
            "post_switch_detection_delay_pulls",
        ),
        rates=(
            "attacker_eliminated",
            "any_false_elimination",
            "complete_stop",
            "eliminated_under",
            "eliminated_over",
            "eliminated_during_banking",
        ),
    )
    episode_groups = defaultdict(list)
    for row in episodes:
        episode_groups[tuple(row[key] for key in group)].append(row)
    for row in episode_agg:
        members = episode_groups[tuple(row[key] for key in group)]
        detected = [member for member in members if member["attacker_eliminated"]]
        row["detected_runs"] = len(detected)
        row["under_side_share_given_detected"] = (
            None
            if not detected
            else sum(member["elimination_side"] == "under" for member in detected)
            / len(detected)
        )
        row["over_side_share_given_detected"] = (
            None
            if not detected
            else sum(member["elimination_side"] == "over" for member in detected)
            / len(detected)
        )
    paired_agg = aggregate(
        paired,
        group,
        continuous=("attack_gain", "relative_attack_gain", "regret_increase"),
        rates=("positive_attack_gain", "attack_attacker_eliminated", "attack_complete_stop"),
    )
    banking_agg = aggregate(
        banking,
        ("scenario", "mechanism", "banking_type", "condition"),
        continuous=(
            "banking_specific_gain",
            "banking_cost",
            "spending_phase_gain",
            "banking_delay",
            "post_banking_selection_gain",
            "regret_increase",
        ),
        rates=("positive_attack_gain", "pair_censored", "attack_attacker_eliminated"),
    )

    tables = output_dir / "tables"
    write_csv(tables / "paired_metrics_vs_truthful.csv", paired)
    write_csv(tables / "banking_specific_paired_metrics.csv", banking)
    write_csv(tables / "aggregate_episode_metrics.csv", episode_agg)
    write_csv(tables / "aggregate_paired_metrics.csv", paired_agg)
    write_csv(tables / "aggregate_banking_specific_metrics.csv", banking_agg)
    return episode_agg, paired_agg, banking_agg


def create_summary_plots(
    output_dir, episode_agg, paired_agg, banking_agg, selected_scales
):
    figures = output_dir / "figures"
    radius_aware_labels = dict(MECHANISM_LABELS)
    for mechanism, scale in selected_scales.items():
        radius_aware_labels[mechanism] = (
            f"{MECHANISM_LABELS[mechanism]} (c={scale:g})"
        )

    for scenario in sorted({row["scenario"] for row in episode_agg}):
        paired_rows = [
            dict(row) for row in paired_agg if row["scenario"] == scenario
        ]
        episode_rows = [row for row in episode_agg if row["scenario"] == scenario]
        banking_rows = [row for row in banking_agg if row["scenario"] == scenario]

        censor_lookup = {
            (row["condition"], row["mechanism"]): row.get(
                "pair_censored_mean"
            )
            for row in banking_rows
        }
        for row in paired_rows:
            censor_rate = censor_lookup.get(
                (row["condition"], row["mechanism"])
            )
            if censor_rate is not None:
                row["pair_censored_mean"] = censor_rate

        # The banking-specific comparison is the primary outcome: it isolates
        # the value of banking from the later over-reporting phase.
        plot_grouped_estimates(
            banking_rows,
            "condition",
            "mechanism",
            "banking_specific_gain",
            figures / f"banking_specific_gain__{scenario}.png",
            f"Primary outcome: banking-specific gain — {scenario}",
            "Additional selections attributable to banking",
            category_order=["time_banker", "pull_banker"],
            series_order=list(MECHANISMS),
            category_labels=CONDITION_LABELS,
            series_labels=radius_aware_labels,
            censor_metric="pair_censored",
        )

        plot_grouped_estimates(
            paired_rows,
            "condition",
            "mechanism",
            "attack_gain",
            figures / f"attack_gain_vs_truthful__{scenario}.png",
            f"Descriptive total attacker selection gain — {scenario}",
            "Total selections gained versus truthful (not banking-specific)",
            category_order=list(CONDITIONS[1:]),
            series_order=list(MECHANISMS),
            category_labels=CONDITION_LABELS,
            series_labels=radius_aware_labels,
            censor_metric="pair_censored",
        )
        plot_grouped_estimates(
            paired_rows,
            "condition",
            "mechanism",
            "regret_increase",
            figures / f"regret_increase_vs_truthful__{scenario}.png",
            f"Paired cumulative-regret increase — {scenario}",
            "Regret increase versus truthful",
            category_order=list(CONDITIONS[1:]),
            series_order=list(MECHANISMS),
            category_labels=CONDITION_LABELS,
            series_labels=radius_aware_labels,
        )
        plot_grouped_estimates(
            episode_rows,
            "condition",
            "mechanism",
            "attacker_eliminated",
            figures / f"attacker_elimination_rate__{scenario}.png",
            f"Attacker elimination rate — {scenario}",
            "Elimination probability",
            category_order=list(CONDITIONS),
            series_order=list(MECHANISMS),
            category_labels=CONDITION_LABELS,
            series_labels=radius_aware_labels,
            horizontal_zero=False,
            zero_count_annotations=True,
        )
        plot_grouped_estimates(
            episode_rows,
            "condition",
            "mechanism",
            "any_false_elimination",
            figures / f"honest_arm_false_elimination_rate__{scenario}.png",
            f"Honest-arm collateral elimination rate — {scenario}",
            "Probability of at least one honest-arm elimination",
            category_order=list(CONDITIONS),
            series_order=list(MECHANISMS),
            category_labels=CONDITION_LABELS,
            series_labels=radius_aware_labels,
            horizontal_zero=False,
            zero_count_annotations=True,
        )

        time_rows = [
            row for row in banking_rows if row["banking_type"] == "time"
        ]
        decomposition = []
        for row in time_rows:
            for component, source, sign in (
                ("banking_loss", "banking_cost", -1.0),
                ("spending_gain", "spending_phase_gain", 1.0),
                ("net_gain", "banking_specific_gain", 1.0),
            ):
                transformed = {
                    "mechanism": row["mechanism"],
                    "component": component,
                }
                for suffix in ("mean", "ci_low", "ci_high"):
                    value = row[f"{source}_{suffix}"]
                    if sign < 0 and suffix == "ci_low":
                        value = -row[f"{source}_ci_high"]
                    elif sign < 0 and suffix == "ci_high":
                        value = -row[f"{source}_ci_low"]
                    else:
                        value = sign * value
                    transformed[f"phase_effect_{suffix}"] = value
                decomposition.append(transformed)

        plot_grouped_estimates(
            decomposition,
            "mechanism",
            "component",
            "phase_effect",
            figures / f"time_banking_phase_decomposition__{scenario}.png",
            f"Time-based banking phase decomposition — {scenario}",
            "Selection effect (loss shown below zero)",
            category_order=list(MECHANISMS),
            series_order=["banking_loss", "spending_gain", "net_gain"],
            category_labels=radius_aware_labels,
            series_labels={
                "banking_loss": "Banking-phase loss",
                "spending_gain": "Spending-phase gain",
                "net_gain": "Net banking-specific gain",
            },
        )


def representative_traces(args, output_dir, selected_scales, t0, bank_pulls):
    if args.no_traces:
        return
    for scenario_name, scenario in scenarios(args.horizon).items():
        for mechanism in ELIMINATION_MECHANISMS:
            for condition in ("time_banker", "pull_banker"):
                result = run_credit_banking_condition(
                    seed=args.trace_seed,
                    learner_factory=learner_factory(
                        mechanism, selected_scales[mechanism]
                    ),
                    strategy_factory=strategy_factory(condition, t0, bank_pulls),
                    attacker=scenario["attacker"],
                    env_kwargs=scenario["env_kwargs"],
                    summary_only=False,
                    collect_traces=True,
                )
                title = (
                    f"{CONDITION_LABELS[condition]} — "
                    f"{MECHANISM_LABELS[mechanism]} — {scenario_name}"
                )
                plot_attacker_trigger_trace(
                    result,
                    output_dir / "figures" / "traces"
                    / f"trigger_trace__{scenario_name}__{mechanism}__{condition}"
                      f"__seed_{args.trace_seed}.png",
                    title,
                )
                write_json(
                    output_dir / "trace_summaries"
                    / f"trace_summary__{scenario_name}__{mechanism}__{condition}"
                      f"__seed_{args.trace_seed}.json",
                    result["summary"],
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "primary", "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "credit_banking_experiment",
    )
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--calibration-seeds", type=int, default=50)
    parser.add_argument("--evaluation-seeds", type=int, default=50)
    parser.add_argument("--trace-seed", type=int, default=0)
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.horizon = min(args.horizon, 300)
        args.calibration_seeds = min(args.calibration_seeds, 3)
        args.evaluation_seeds = min(args.evaluation_seeds, 3)

    radius_grid = QUICK_RADIUS_GRID if args.quick else RADIUS_GRID
    t0 = int(round(0.20 * args.horizon))
    bank_pulls = 10 if args.quick else 100

    calibration_protocol = {
        "stage": "truthful_radius_calibration",
        "horizon": args.horizon,
        "calibration_seeds": args.calibration_seeds,
        "calibration_seed_start": 1000,
        "radius_grid": radius_grid,
        "mechanisms": ELIMINATION_MECHANISMS,
        "environment": {
            "context_mode": "controlled",
            "K": 6,
            "d": 5,
            "sigma": 0.1,
            "quality_jitter": 0.08,
        },
        "radius_scaling": "multiplies_entire_beta_root",
        "quick": args.quick,
    }
    calibration_run_id = protocol_hash(calibration_protocol)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("calibration", "all"):
        selected_scales = calibration(
            args, output_dir, calibration_run_id
        )
    else:
        selected_scales = load_selected_scales(
            output_dir, calibration_run_id
        )

    primary_protocol = {
        "stage": "primary_evaluation",
        "horizon": args.horizon,
        "evaluation_seeds": args.evaluation_seeds,
        "evaluation_seed_start": 0,
        "conditions": CONDITIONS,
        "mechanisms": MECHANISMS,
        "scenarios": tuple(scenarios(args.horizon)),
        "strategy_parameters": {
            "under_delta": 0.10,
            "over_delta": 0.15,
            "banker_a_minus": 0.10,
            "banker_a_plus": 0.15,
            "time_switch_round": t0,
            "pull_banking_target": bank_pulls,
        },
        "window_sizes": {"windowed_w200": 200},
        "selected_radius_scales": selected_scales,
        "calibration_run_id": calibration_run_id,
        "quick": args.quick,
    }
    primary_run_id = protocol_hash(primary_protocol)
    write_json(
        output_dir / "experiment_manifest.json",
        {
            "calibration_run_id": calibration_run_id,
            "primary_run_id": primary_run_id,
            "calibration_protocol": calibration_protocol,
            "primary_protocol": primary_protocol,
        },
    )

    if args.stage in ("primary", "all"):
        episodes, t0, bank_pulls = primary_runs(
            args, output_dir, primary_run_id, selected_scales
        )
        paired, banking = paired_tables(episodes)
        episode_agg, paired_agg, banking_agg = aggregate_and_write(
            output_dir, episodes, paired, banking
        )
        create_summary_plots(
            output_dir,
            episode_agg,
            paired_agg,
            banking_agg,
            selected_scales,
        )
        representative_traces(
            args, output_dir, selected_scales, t0, bank_pulls
        )

    print(f"Experiment outputs written to: {output_dir}")
    print(f"Calibration run id: {calibration_run_id}")
    print(f"Primary run id: {primary_run_id}")


if __name__ == "__main__":
    main()
