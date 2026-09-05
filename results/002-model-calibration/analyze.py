"""Aggregate saved metrics without refitting or selecting model checkpoints."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results.json").read_text())
runs = data["runs"]


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": float(np.mean(values)), "min": min(values), "max": max(values),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values": values,
    }


groups = []
for variant in ("open", "rooms"):
    for repeat in (1, 8):
        for count in (500, 2000):
            selected = [run for run in runs if run["variant"] == variant
                        and run["training_action_repeat"] == repeat and run["training_count"] == count]
            group = {
                "variant": variant, "training_action_repeat": repeat, "training_count": count,
                "seeds": [run["seed"] for run in selected],
                "coverage": stats([run["training_data_summary"]["square_coverage_fraction"] for run in selected]),
                "checkpoints": {},
            }
            for index, step in enumerate(data["protocol"]["fit_checkpoints"]):
                checkpoints = [run["checkpoints"][index] for run in selected]
                group["checkpoints"][str(step)] = {
                    "one_step_physical_mse": stats([c["one_step"]["physical_mse"] for c in checkpoints]),
                    "binary_brier": stats([c["one_step"]["binary_brier"] for c in checkpoints]),
                    "rollout8_physical_mse": stats([c["rollouts"]["8"]["endpoint_physical_mse"] for c in checkpoints]),
                    "rollout24_physical_mse": stats([c["rollouts"]["24"]["endpoint_physical_mse"] for c in checkpoints]),
                    "rollout24_position_mse": stats([c["rollouts"]["24"]["endpoint_position_mse"] for c in checkpoints]),
                    "disagreement_error_spearman": stats([c["one_step"]["disagreement_error_relation"]["spearman"] for c in checkpoints]),
                    "top_decile_error_lift": stats([c["one_step"]["disagreement_error_relation"]["top_decile_error_lift"] for c in checkpoints]),
                    "training_seconds": stats([c["cumulative_training_seconds"] for c in checkpoints]),
                    "velocity_drop_error_fraction": stats([c["one_step"]["observed_velocity_drop_events"]["fraction_of_total_physical_squared_error"] for c in checkpoints]),
                }
            groups.append(group)

final_step = str(data["protocol"]["fit_checkpoints"][-1])
summary = {
    "status": "development observations, three seeds per group; no confirmatory significance claims",
    "groups": groups,
    "fitting_comparison": {
        "one_step_improved_100_to_final": sum(run["checkpoints"][-1]["one_step"]["physical_mse"] < run["checkpoints"][1]["one_step"]["physical_mse"] for run in runs),
        "rollout24_improved_100_to_final": sum(run["checkpoints"][-1]["rollouts"]["24"]["endpoint_physical_mse"] < run["checkpoints"][1]["rollouts"]["24"]["endpoint_physical_mse"] for run in runs),
        "run_count": len(runs),
    },
    "identity_comparison": {
        "one_step_beats_identity": sum(run["checkpoints"][-1]["one_step"]["physical_mse"] < run["identity_baseline"]["one_step_physical_mse"] for run in runs),
        "rollout8_beats_identity": sum(run["checkpoints"][-1]["rollouts"]["8"]["endpoint_physical_mse"] < run["identity_baseline"]["rollouts"]["8"]["endpoint_physical_mse"] for run in runs),
        "rollout24_beats_identity": sum(run["checkpoints"][-1]["rollouts"]["24"]["endpoint_physical_mse"] < run["identity_baseline"]["rollouts"]["24"]["endpoint_physical_mse"] for run in runs),
    },
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
lines = [
    "All entries below are means across three development seeds; see summary.json for every value and range.",
    "",
    "| World | Hold | Transitions | Coverage | 1-step MSE | 8-step MSE | 24-step MSE | Spearman | Top10% error lift |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for group in groups:
    final = group["checkpoints"][final_step]
    lines.append(
        f"| {group['variant']} | {group['training_action_repeat']} | {group['training_count']} | "
        f"{100*group['coverage']['mean']:.1f}% | {final['one_step_physical_mse']['mean']:.3g} | "
        f"{final['rollout8_physical_mse']['mean']:.3g} | {final['rollout24_physical_mse']['mean']:.3g} | "
        f"{final['disagreement_error_spearman']['mean']:.3f} | {final['top_decile_error_lift']['mean']:.2f} |"
    )
(ROOT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(json.dumps(summary["fitting_comparison"]))
print(json.dumps(summary["identity_comparison"]))
for variant in ("open", "rooms"):
    selected = [run for run in runs if run["variant"] == variant]
    events = [run["checkpoints"][-1]["one_step"]["observed_velocity_drop_events"] for run in selected]
    print(variant, "velocity-drop event fraction", np.mean([event["count"] / 1536 for event in events]),
          "error fraction", np.mean([event["fraction_of_total_physical_squared_error"] for event in events]),
          "contact/other error ratio", np.mean([event["physical_mse"] / event["other_physical_mse"] for event in events]))
