"""Reproducible development study using only actual random-action experience.

Run with ``python -m rtga.model_study --output results/002-model-calibration``.
The evaluator creates separate real trajectories; neither their states nor any
simulator oracle are available to the model's training loop.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch

from .envs import PuckConfig, PuckLab
from .models import DynamicsEnsemble, EnsembleConfig, ReplayBuffer


SCALE = np.array([1, 1, .5, .5, 1, 1], dtype=np.float32)
PROBE_STARTS = [(x, y) for x in (.10, .35, .65, .90) for y in (.12, .32, .68, .88)]


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, including ties; no optional statistics dependency."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2.0)[inverse]


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranking_metrics(disagreement: np.ndarray, error: np.ndarray) -> dict:
    count = max(1, int(math.ceil(len(error) * .1)))
    selected = np.argsort(disagreement)[-count:]
    return {
        "spearman": _correlation(_rank(disagreement), _rank(error)),
        "pearson": _correlation(disagreement, error),
        "top_decile_error_lift": float(error[selected].mean() / max(error.mean(), 1e-15)),
        "top_decile_count": count,
    }


def collect_training(variant: str, seed: int, repeat: int, count: int):
    env = PuckLab(PuckConfig(variant=variant), seed=10_000 + seed)
    rng = np.random.default_rng(20_000 + seed)
    states = [env.observe()]
    actions = []
    for step in range(count):
        if step % repeat == 0:
            action = int(rng.integers(env.n_actions))
        actions.append(action)
        states.append(env.step(action))
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def collect_probes(variant: str, seed: int, length: int = 48):
    observations, action_sequences, repeats, starts = [], [], [], []
    for repeat in (1, 8):
        for index, start in enumerate(PROBE_STARTS):
            env = PuckLab(PuckConfig(variant=variant, start=start), seed=30_000 + seed * 100 + index)
            rng = np.random.default_rng(40_000 + seed * 100 + index + repeat * 10_000)
            states, actions = [env.observe()], []
            for step in range(length):
                if step % repeat == 0:
                    action = int(rng.integers(env.n_actions))
                actions.append(action)
                states.append(env.step(action))
            observations.append(states)
            action_sequences.append(actions)
            repeats.append(repeat)
            starts.append(start)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(action_sequences, dtype=np.int64),
        "action_repeats": np.asarray(repeats, dtype=np.int64),
        "starts": np.asarray(starts, dtype=np.float32),
    }


def data_summary(states: np.ndarray, actions: np.ndarray, probes: dict) -> dict:
    cells = np.minimum((states[:-1, :2] * 16).astype(int), 15)
    unique_cells = len(np.unique(cells, axis=0))
    distances = np.linalg.norm(probes["starts"][:, None, :] - states[None, :-1, :2], axis=-1)
    return {
        "actual_transitions": len(actions),
        "unique_position_cells_16x16": unique_cells,
        "square_coverage_fraction": unique_cells / 256,
        "right_half_fraction": float(np.mean(states[:-1, 0] > .5)),
        "per_coordinate_min": states[:-1].min(axis=0).tolist(),
        "per_coordinate_max": states[:-1].max(axis=0).tolist(),
        "action_counts": np.bincount(actions, minlength=6).tolist(),
        "binary_zero_to_one_events": int(np.sum(np.diff(states[:, 4]) > .5)),
        "probe_start_nearest_training_position_distance": distances.min(axis=1).tolist(),
    }


@torch.no_grad()
def evaluate_one_step(model: DynamicsEnsemble, probes: dict) -> dict:
    states = probes["observations"][:, :-1].reshape(-1, 6)
    targets = probes["observations"][:, 1:].reshape(-1, 6)
    actions = probes["actions"].reshape(-1)
    means, variances = model.predict(states, actions)
    mean = means.mean(axis=0)
    squared_error = np.square(mean - targets)
    normalized_physical_error = (squared_error[:, :4] / SCALE[:4] ** 2).mean(axis=1)
    # Every member sees the exact same actual state/action for this statistic.
    normalized_variance = np.var(means, axis=0) / SCALE ** 2
    local_disagreement = normalized_variance[:, :4].mean(axis=1)
    full_disagreement = normalized_variance.mean(axis=1)
    repeat_per_transition = np.repeat(probes["action_repeats"], probes["actions"].shape[1])
    velocity_drop = np.any((np.abs(states[:, 2:4]) > .025) & (np.abs(targets[:, 2:4]) < 1e-7), axis=1)
    result = {
        "physical_mse": float(squared_error[:, :4].mean()),
        "position_mse": float(squared_error[:, :2].mean()),
        "velocity_mse": float(squared_error[:, 2:4].mean()),
        "normalized_physical_mse": float(normalized_physical_error.mean()),
        "per_coordinate_mse": squared_error.mean(axis=0).tolist(),
        "binary_brier": float(squared_error[:, 4].mean()),
        "physical_disagreement_mean": float(local_disagreement.mean()),
        "all_coordinate_disagreement_mean": float(full_disagreement.mean()),
        "physical_aleatoric_variance_mean": float(variances[:, :, :4].mean()),
        "disagreement_error_relation": _ranking_metrics(local_disagreement, normalized_physical_error),
        "all_coordinate_disagreement_physical_error_relation": _ranking_metrics(full_disagreement, normalized_physical_error),
        "mixture_metrics": model.evaluate(states, actions, targets),
        "by_probe_action_repeat": {},
        "observed_velocity_drop_events": {
            "definition": "At least one velocity coordinate changes from magnitude >.025 to <1e-7; an observed-transition collision proxy, not an oracle label.",
            "count": int(velocity_drop.sum()),
            "physical_mse": float(squared_error[velocity_drop, :4].mean()) if velocity_drop.any() else None,
            "other_physical_mse": float(squared_error[~velocity_drop, :4].mean()),
            "fraction_of_total_physical_squared_error": float(squared_error[velocity_drop, :4].sum() / max(squared_error[:, :4].sum(), 1e-15)),
        },
        "per_transition": {
            "normalized_physical_squared_error": normalized_physical_error.tolist(),
            "normalized_physical_disagreement": local_disagreement.tolist(),
            "all_coordinate_disagreement": full_disagreement.tolist(),
        },
    }
    for repeat in (1, 8):
        mask = repeat_per_transition == repeat
        result["by_probe_action_repeat"][str(repeat)] = {
            "physical_mse": float(squared_error[mask, :4].mean()),
            "disagreement_error_relation": _ranking_metrics(local_disagreement[mask], normalized_physical_error[mask]),
        }
    return result


def identity_baseline(probes: dict) -> dict:
    """Post-hoc descriptive reference: predict the current observation forever."""
    observations = probes["observations"]
    return {
        "one_step_physical_mse": float(np.square(observations[:, 1:, :4] - observations[:, :-1, :4]).mean()),
        "rollouts": {
            str(horizon): {
                "endpoint_physical_mse": float(np.square(observations[:, horizon, :4] - observations[:, 0, :4]).mean()),
                "endpoint_position_mse": float(np.square(observations[:, horizon, :2] - observations[:, 0, :2]).mean()),
            }
            for horizon in (8, 24)
        },
    }


@torch.no_grad()
def evaluate_rollouts(model: DynamicsEnsemble, probes: dict, horizons=(8, 24)) -> dict:
    batch = len(probes["observations"])
    members = model.config.members
    state = torch.as_tensor(probes["observations"][:, 0], device=model.device)
    state = state.unsqueeze(0).expand(members, -1, -1).clone()
    actions = torch.as_tensor(probes["actions"], device=model.device)
    normalized_scale = torch.as_tensor(SCALE, device=model.device)
    predictions, local_disagreements = [], []
    member_indices = torch.arange(members, device=model.device)[:, None]
    branch_indices = torch.arange(members * batch, device=model.device).reshape(members, batch)
    for step in range(max(horizons)):
        flat_state = state.reshape(members * batch, 6)
        flat_actions = actions[:, step].repeat(members)
        all_means, _ = model.predict_tensor(flat_state, flat_actions)
        # Conditional disagreement is computed at each common branch input,
        # never by subtracting endpoints of diverged branches.
        local = (all_means.var(dim=0, unbiased=False) / normalized_scale.square())[:, :4].mean(dim=-1)
        local_disagreements.append(local.reshape(members, batch).mean(dim=0).cpu().numpy())
        # Each member carries its own forecast consistently across the rollout.
        state = all_means[member_indices, branch_indices]
        predictions.append(state.cpu().numpy())
    predictions = np.stack(predictions, axis=2)  # [K, trajectory, time, D]
    local_disagreements = np.stack(local_disagreements, axis=1)
    mean_prediction = predictions.mean(axis=0)
    true_future = probes["observations"][:, 1:max(horizons) + 1]
    result = {}
    for horizon in horizons:
        error = np.square(mean_prediction[:, :horizon, :4] - true_future[:, :horizon, :4])
        endpoint = error[:, -1].mean(axis=1)
        local_mean = local_disagreements[:, :horizon].mean(axis=1)
        result[str(horizon)] = {
            "endpoint_physical_mse": float(endpoint.mean()),
            "endpoint_position_mse": float(error[:, -1, :2].mean()),
            "endpoint_velocity_mse": float(error[:, -1, 2:4].mean()),
            "path_physical_mse": float(error.mean()),
            "member_endpoint_physical_mse": float(np.square(
                predictions[:, :, horizon - 1, :4] - true_future[None, :, horizon - 1, :4]
            ).mean()),
            "conditional_disagreement_mean": float(local_mean.mean()),
            "conditional_disagreement_endpoint_error_relation": _ranking_metrics(local_mean, endpoint),
            "per_trajectory": {
                "endpoint_physical_squared_error": endpoint.tolist(),
                "conditional_disagreement_mean": local_mean.tolist(),
                "predicted_endpoint_mean": mean_prediction[:, horizon - 1].tolist(),
                "actual_endpoint": true_future[:, horizon - 1].tolist(),
            },
        }
    return result


def run_study(output: Path, seeds: list[int], fit_steps: int = 500) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    datasets = output / "datasets"
    checkpoints = output / "checkpoints"
    datasets.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    source_files = ("model_study.py", "models.py", "envs.py")
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in source_files}
    result = {
        "study": "002-model-calibration",
        "protocol": {
            "seeds": seeds, "variants": ["open", "rooms"], "training_counts": [500, 2000],
            "training_action_repeats": [1, 8], "probe_action_repeats": [1, 8],
            "fit_checkpoints": sorted(set([0, min(100, fit_steps), fit_steps])),
            "batch_size_per_member": 128, "torch_threads": 1,
            "train_start": [.2, .5], "probe_starts": PROBE_STARTS,
            "probe_length": 48, "probe_trajectory_count": 32,
            "training_policy": "Uniform random action, held for repeat actual environment steps; no reward, goals or reset.",
            "data_separation": "500-transition training data is the prefix of a 2000-step real run. Independent real probe trajectories, initial positions and RNG streams; never fit on probes.",
            "initialization": "Fresh model and replay per condition. Same member initialization per seed for paired comparisons.",
            "rollouts": "Conditional-mean rollouts, with fixed member identity per branch; 8/24 steps from probe episode starts. Gaussian process noise is not sampled in these deterministic environments.",
            "disagreement": "Variance across predictions receiving the same input, normalized by declared observation scale. Physical statistic uses dimensions 0:4. Endpoint branch spread is not used as epistemic uncertainty.",
            "binary_limitation": "Door flag is always 1 in open/rooms, so Brier tests a constant flag, not switching or mechanism learning.",
            "posthoc_diagnostics": "After the initial unchanged-configuration run, add a constant-state baseline and observed nonzero-to-zero velocity-event breakdown to interpret residual errors. No model settings or training data were changed.",
        },
        "provenance": {
            "command": " ".join([sys.executable, "-m", "rtga.model_study", *sys.argv[1:]]),
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "source_sha256": hashes,
        },
        "runs": [],
    }
    for variant in ("open", "rooms"):
        for seed in seeds:
            probes = collect_probes(variant, seed)
            probe_path = datasets / f"probes-{variant}-seed{seed}.npz"
            np.savez_compressed(probe_path, **probes)
            for repeat in (1, 8):
                observations, actions = collect_training(variant, seed, repeat, 2000)
                training_path = datasets / f"training-{variant}-hold{repeat}-seed{seed}.npz"
                np.savez_compressed(training_path, observations=observations, actions=actions)
                for count in (500, 2000):
                    identifier = f"{variant}-hold{repeat}-n{count}-seed{seed}"
                    replay = ReplayBuffer(count, 6, seed=50_000 + seed)
                    for state, action, next_state in zip(observations[:count], actions[:count], observations[1:count + 1]):
                        replay.append(state, int(action), next_state)
                    config = EnsembleConfig(
                        state_dim=6, n_actions=6, members=3, hidden=64,
                        seed=seed, observation_scale=tuple(SCALE), binary_dims=(4,),
                        observation_low=(0, 0, -.5, -.5, 0, -1),
                        observation_high=(1, 1, .5, .5, 1, 1),
                    )
                    model = DynamicsEnsemble(config)
                    entry = {
                        "id": identifier, "variant": variant, "seed": seed,
                        "training_action_repeat": repeat, "training_count": count,
                        "ensemble_config": asdict(config),
                        "training_data": str(training_path.relative_to(output)),
                        "probe_data": str(probe_path.relative_to(output)),
                        "training_data_summary": data_summary(observations[:count + 1], actions[:count], probes),
                        "identity_baseline": identity_baseline(probes),
                        "checkpoints": [],
                    }
                    training_seconds = 0.0
                    for checkpoint in result["protocol"]["fit_checkpoints"]:
                        before_training = perf_counter()
                        train_metrics = model.train_steps(replay, checkpoint - model.version, 128)
                        training_seconds += perf_counter() - before_training
                        before_evaluation = perf_counter()
                        entry["checkpoints"].append({
                            "fit_batches": checkpoint, "training_metrics": train_metrics,
                            "cumulative_training_seconds": training_seconds,
                            "one_step": evaluate_one_step(model, probes),
                            "rollouts": evaluate_rollouts(model, probes),
                        })
                        entry["checkpoints"][-1]["evaluation_seconds"] = perf_counter() - before_evaluation
                    model_path = checkpoints / f"{identifier}.pt"
                    model.save(model_path)
                    entry["final_model_checkpoint"] = str(model_path.relative_to(output))
                    result["runs"].append(entry)
                    # Every completed condition is preserved if execution stops.
                    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    final = entry["checkpoints"][-1]
                    print(json.dumps({
                        "run": identifier, "physical_mse": final["one_step"]["physical_mse"],
                        "rollout24_mse": final["rollouts"]["24"]["endpoint_physical_mse"],
                        "spearman": final["one_step"]["disagreement_error_relation"]["spearman"],
                        "training_seconds": training_seconds,
                    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/002-model-calibration"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fit-steps", type=int, default=500)
    args = parser.parse_args()
    if args.fit_steps < 0:
        parser.error("--fit-steps must be nonnegative")
    run_study(args.output, args.seeds, args.fit_steps)


if __name__ == "__main__":
    main()
