"""Reproducible experiments; evaluator information stays outside agents."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

from .envs import PuckLab, PuckConfig, ExactPuckModel
from .planning import EvolutionPlanner, PlannerConfig, Evaluation


def provenance():
    def git(*args):
        return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'command': list(sys.orig_argv),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.glob('*.py')}}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def environment_description(env):
    c = asdict(env.config)
    c.update(width=1, height=1, door_y=c['doorway_y'],
             action_names=['no-op', 'left', 'right', 'up', 'down', 'interact'])
    return c


class ExactGoalEvaluator:
    def __init__(self, model, observation, goal, discount=.99):
        self.model, self.observation = model, observation.copy()
        self.goal, self.discount = np.asarray(goal), discount

    def __call__(self, plans):
        paths = self.model.rollout(self.observation, plans)
        distances = np.linalg.norm(paths[..., :2] - self.goal, axis=-1)
        weights = self.discount ** np.arange(plans.shape[1])
        scores = -(distances * weights).sum(axis=1) / weights.sum()
        # The objective is explicitly task-directed in this diagnostic only.
        return Evaluation(scores, paths, model_transitions=plans.size)


def trace_frame(step, state, action, decision, next_state, metrics):
    expected = None if decision.selected_path is None else decision.selected_path[0]
    return {'step': step, 'state': np.asarray(state).round(6).tolist(), 'action': int(action),
            'objective': decision.score, 'plans': decision.plans[:24].tolist(),
            'paths': [] if decision.paths is None else decision.paths[:8].round(6).tolist(),
            'selected_path': [] if decision.selected_path is None else decision.selected_path.round(6).tolist(),
            'gene_ages': decision.gene_ages[:24].tolist(),
            'prediction_error': None if expected is None else float(np.mean((expected - next_state) ** 2)),
            'disagreement': decision.disagreement, 'latency_ms': decision.latency_ms,
            'metrics': metrics}


def run_oracle(seed, method, variant='rooms', steps=100, planner_config=None, capture_trace=False):
    rng = np.random.default_rng(seed + 1701)
    start = (.15, float(rng.uniform(.18, .82)))
    goal = np.array([.85, rng.uniform(.18, .82)])
    env = PuckLab(PuckConfig(variant=variant, start=start), seed=seed)
    model = ExactPuckModel(env)
    c = replace(planner_config or PlannerConfig(), method=method, seed=seed + 7101)
    planner = EvolutionPlanner(env.n_actions, c)
    obs = env.observe()
    distances, latencies, actions, frames = [], [], [], []
    transitions, first_success = 0, None
    begin = perf_counter()
    for step in range(steps):
        decision = planner.plan(ExactGoalEvaluator(model, obs, goal))
        next_obs = env.step(decision.action)
        distance = float(np.linalg.norm(next_obs[:2] - goal))
        if first_success is None and distance <= .06:
            first_success = step + 1
        distances.append(distance)
        latencies.append(decision.latency_ms)
        transitions += decision.model_transitions
        actions.append(decision.action)
        if capture_trace:
            frame = trace_frame(step, obs, decision.action, decision, next_obs, {'distance': distance})
            frames.append(frame)
        obs = next_obs
    record = {'seed': seed, 'method': method, 'variant': variant,
              'steps': steps, 'success': first_success is not None, 'first_success_step': first_success,
              'final_distance': distances[-1], 'mean_distance': float(np.mean(distances)),
              'model_transitions': transitions, 'wall_seconds': perf_counter() - begin,
              'latency_p50_ms': float(np.percentile(latencies, 50)),
              'latency_p95_ms': float(np.percentile(latencies, 95)),
              'distance_curve': distances, 'latency_ms': latencies, 'actions': actions,
              'start': list(start), 'goal': goal.tolist(), 'planner': asdict(c)}
    trace = {'schema_version': 1, 'metadata': {'title': 'RTGA oracle navigation', 'method': method,
              'seed': seed, 'config': asdict(c), 'goal': goal.tolist(),
              'metrics': {k: record[k] for k in ['success', 'first_success_step', 'final_distance']}},
              'environment': environment_description(env), 'frames': frames}
    trace['environment']['goal'] = goal.tolist()
    return record, trace


def run_oracle_suite(output, seeds=range(10), methods=('persistent', 'fresh', 'random', 'cem'),
                     variants=('open', 'rooms'), steps=100, planner_config=None):
    output = Path(output)
    data = {'experiment': 'oracle_navigation', 'provenance': provenance(), 'records': [],
            'protocol': {'steps': steps, 'success_radius': .06, 'objective': 'discounted mean distance',
                         'discount': .99, 'planner': asdict(planner_config or PlannerConfig()),
                         'seeds': list(seeds), 'methods': list(methods), 'variants': list(variants)}}
    for variant in variants:
        for seed in seeds:
            for method in methods:
                record, trace = run_oracle(seed, method, variant, steps, planner_config,
                                           capture_trace=(seed == min(seeds)))
                data['records'].append(record)
                write_json(output / 'results.json', data)
                print(json.dumps({k: record[k] for k in ['variant', 'seed', 'method', 'success',
                                                         'final_distance', 'wall_seconds']}), flush=True)
                if trace['frames']:
                    write_json(output / f'trace-{variant}-{method}.json', trace)
                    from .viewer import write_viewer
                    write_viewer(trace, output / f'viewer-{variant}-{method}.html')
    return data
