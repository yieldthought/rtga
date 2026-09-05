"""Offline rare-event learning diagnostics on one unchanged real dataset.

The only neural difference is a generic high-change sampling distribution.
Simulator probes remain evaluator-only. The nearest-neighbour candidate is
an empirical-memory diagnostic and is not used to collect training data.
"""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys

import numpy as np
import torch


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_runtime(output):
    source = output / 'source'
    source.mkdir(exist_ok=True)
    for name in ['__init__.py', 'agent.py', 'models.py', 'planning.py', 'envs.py',
                 'event_learning_study.py']:
        shutil.copy2(Path(__file__).parent / name, source / name)
    package = '_rtga_event_learning_snapshot'
    spec = importlib.util.spec_from_file_location(
        package, source / '__init__.py', submodule_search_locations=[str(source.resolve())])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    runtime = {name: importlib.import_module(f'{package}.{name}')
               for name in ['agent', 'models', 'planning', 'envs']}
    return runtime, {path.name: sha256(path) for path in source.glob('*.py')}


class OfflineReplay:
    """Experiment-local sampler; never changes the retained dataset."""

    def __init__(self, data, scale, mode, seed, event_index):
        self.states, self.actions, self.next_states = data
        self.state_dim = self.states.shape[1]
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        change = np.max(np.abs((self.next_states - self.states) / scale), axis=1)
        self.tail = np.argsort(-change, kind='stable')[:int(np.ceil(len(self) * .01))]
        self.event_index = event_index
        self.exposure = np.zeros(3, dtype=np.int64)
        self.draws = np.zeros(3, dtype=np.int64)

    def __len__(self):
        return len(self.actions)

    def sample(self, batch_size, members=1):
        if self.mode == 'uniform':
            indices = self.rng.integers(len(self), size=(members, batch_size))
        elif self.mode == 'high_change':
            half = batch_size // 2
            uniform = self.rng.integers(len(self), size=(members, half))
            tail = self.tail[self.rng.integers(len(self.tail), size=(members, batch_size - half))]
            indices = np.concatenate([uniform, tail], axis=1)
        else:
            raise ValueError(self.mode)
        self.exposure[:members] += (indices == self.event_index).sum(axis=1)
        self.draws[:members] += batch_size
        return self.states[indices], self.actions[indices], self.next_states[indices]


class KNNResidual:
    """Action-conditioned distance-weighted residual memory, exposed as one member."""

    def __init__(self, data, config):
        self.config = replace(config, members=1)
        self.device = torch.device('cpu')
        self.version = 0
        self.scale = torch.tensor(config.observation_scale, dtype=torch.float32)
        self.low = torch.tensor(config.observation_low, dtype=torch.float32)
        self.high = torch.tensor(config.observation_high, dtype=torch.float32)
        states, actions, targets = data
        self.memory = {}
        for action in range(config.n_actions):
            selected = actions == action
            s = torch.tensor(states[selected])
            n = torch.tensor(targets[selected])
            self.memory[action] = (s / self.scale, n - s, n)

    @torch.no_grad()
    def predict_tensor(self, states, actions):
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        mean = torch.empty_like(states)
        variance = torch.empty_like(states)
        for action, (memory, residual, targets) in self.memory.items():
            mask = actions == action
            if not mask.any():
                continue
            query = states[mask]
            distance = ((query[:, None] / self.scale - memory[None]) ** 2).sum(dim=-1)
            d2, neighbours = torch.topk(distance, k=min(5, len(memory)), largest=False)
            weights = 1 / (d2 + 1e-6)
            weights /= weights.sum(dim=-1, keepdim=True)
            predictions = query[:, None] + residual[neighbours]
            predictions[:, :, list(self.config.binary_dims)] = targets[neighbours][
                :, :, list(self.config.binary_dims)]
            value = (weights[:, :, None] * predictions).sum(dim=1)
            spread = (weights[:, :, None] * (predictions - value[:, None]) ** 2).sum(dim=1)
            for dim in self.config.binary_dims:
                spread[:, dim] = value[:, dim] * (1 - value[:, dim])
            mean[mask] = value.clamp(self.low, self.high)
            variance[mask] = spread
        return mean[None], variance[None]

    def predict(self, states, actions):
        means, variances = self.predict_tensor(states, actions)
        return means.numpy(), variances.numpy()


def real_targets(runtime, states, actions):
    env = runtime['envs'].PuckLab(runtime['envs'].PuckConfig(variant='mechanism'), seed=60600)
    targets = []
    for state, action in zip(states, actions):
        snap = env.snapshot()
        snap['state'] = state.copy()
        env.restore(snap)
        targets.append(env.step(int(action)))
    return np.asarray(targets)


def make_probes(runtime, event):
    velocities = [[0., 0.], [-.12, 0.], [.12, 0.], [0., -.12], [0., .12],
                  event['state'][2:4]]
    states, actions, velocity_ids = [], [], []
    for x in np.linspace(.18, .42, 13):
        for y in np.linspace(.38, .62, 13):
            for velocity_id, velocity in enumerate(velocities):
                for action in range(6):
                    states.append([x, y, *velocity, 0., 0.])
                    actions.append(action)
                    velocity_ids.append(velocity_id)
    states, actions = np.asarray(states), np.asarray(actions)
    grid = {'states': states, 'actions': actions,
            'targets': real_targets(runtime, states, actions),
            'velocity_id': np.array(velocity_ids), 'velocities': np.array(velocities)}
    rng = np.random.default_rng(60601)
    states, actions = [], []
    for x in [.1, .3, .7, .9]:
        for y in [.2, .5, .8]:
            for door in [0., 1.]:
                for action in range(6):
                    states.append([x, y, *rng.uniform(-.2, .2, size=2), door, 0.])
                    actions.append(action)
    states, actions = np.asarray(states), np.asarray(actions)
    motion = {'states': states, 'actions': actions,
              'targets': real_targets(runtime, states, actions)}
    return grid, motion


def predict(model, states, actions):
    means, variances = [], []
    for start in range(0, len(states), 512):
        mu, var = model.predict(states[start:start + 512], actions[start:start + 512])
        means.append(mu)
        variances.append(var)
    return np.concatenate(means, axis=1), np.concatenate(variances, axis=1)


def binary_metrics(probability, target):
    positive = target >= .5
    negative = ~positive
    return {'count': len(target), 'positives': int(positive.sum()), 'negatives': int(negative.sum()),
            'recall_at_half': float((probability[positive] >= .5).mean()) if positive.any() else None,
            'false_positive_rate_at_half': float((probability[negative] >= .5).mean()) if negative.any() else None,
            'positive_mean_probability': float(probability[positive].mean()) if positive.any() else None,
            'negative_mean_probability': float(probability[negative].mean()) if negative.any() else None,
            'positive_brier': float(((probability[positive] - 1) ** 2).mean()) if positive.any() else None,
            'negative_brier': float((probability[negative] ** 2).mean()) if negative.any() else None}


def evaluate(model, event, grid, motion, output):
    event_mu, _ = model.predict(np.array([event['state']]), np.array([event['action']]))
    grid_mu, grid_var = predict(model, grid['states'], grid['actions'])
    motion_mu, motion_var = predict(model, motion['states'], motion['actions'])
    probability = grid_mu.mean(axis=0)[:, 4]
    target = grid['targets'][:, 4]
    interact = grid['actions'] == 5
    positive_positions = np.linalg.norm(grid['states'][:, :2] - [.3, .5], axis=1) <= .09
    error = motion_mu.mean(axis=0) - motion['targets']
    scale = np.array(model.config.observation_scale)
    result = {
        'event_probability_members': event_mu[:, 0, 4].tolist(),
        'event_probability_mean': float(event_mu[:, 0, 4].mean()),
        'grid_all': binary_metrics(probability, target),
        'grid_interact': binary_metrics(probability[interact], target[interact]),
        'grid_inside_wrong_action': binary_metrics(probability[positive_positions & ~interact],
                                                  target[positive_positions & ~interact]),
        'grid_outside_interact': binary_metrics(probability[~positive_positions & interact],
                                               target[~positive_positions & interact]),
        'grid_by_velocity': [binary_metrics(probability[grid['velocity_id'] == index],
                                            target[grid['velocity_id'] == index])
                             for index in range(len(grid['velocities']))],
        'motion': {
            'count': len(error), 'position_mse': float((error[:, :2] ** 2).mean()),
            'velocity_mse': float((error[:, 2:4] ** 2).mean()),
            'continuous_normalized_mse': float(((error[:, [0, 1, 2, 3, 5]]
                                               / scale[[0, 1, 2, 3, 5]]) ** 2).mean()),
            'door_brier': float((error[:, 4] ** 2).mean()),
        },
    }
    np.savez_compressed(output, grid_mean_members=grid_mu, grid_variance_members=grid_var,
                        motion_mean_members=motion_mu, motion_variance_members=motion_var)
    return result


def frozen_navigation(runtime, model, event, planner_seed, start_kind):
    env = runtime['envs'].PuckLab(runtime['envs'].PuckConfig(variant='mechanism'), seed=60620)
    if start_kind == 'experienced_event':
        snap = env.snapshot()
        snap['state'] = np.array(event['state'])
        env.restore(snap)
    planner = runtime['planning'].EvolutionPlanner(6, runtime['planning'].PlannerConfig(
        population=48, horizon=12, generations=3, seed=planner_seed))
    observation = env.observe()
    goal = np.array([.8, .5])
    frames = []
    opened = reached = None
    transitions = 0
    begin = perf_counter()
    for step in range(100):
        decision = planner.plan(runtime['agent'].LearnedEvaluator(model, observation, mode='goal', goal=goal))
        target = env.step(decision.action)
        transitions += decision.model_transitions
        if opened is None and target[4] >= .5:
            opened = step + 1
        if reached is None and np.linalg.norm(target[:2] - goal) <= .06:
            reached = step + 1
        frames.append({'step': step, 'state': observation.tolist(), 'action': decision.action,
                       'next_state': target.tolist(), 'selected_plan': decision.plan.tolist(),
                       'imagined_path': decision.selected_path.tolist()})
        observation = target
    return {'planner_seed': planner_seed, 'start': start_kind, 'door_open_step': opened,
            'goal_reached_step': reached, 'success': reached is not None,
            'final_distance': float(np.linalg.norm(observation[:2] - goal)),
            'steps': 100, 'model_transitions': transitions, 'wall_seconds': perf_counter() - begin,
            'frames': frames}


def run(output, dataset):
    torch.set_num_threads(1)
    output, dataset = Path(output), Path(dataset)
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / 'PROTOCOL.md'
    if not protocol.exists():
        raise ValueError('The declared PROTOCOL.md must exist before fitting')
    dataset_hash = sha256(dataset)
    with np.load(dataset) as archive:
        data = tuple(archive[name].copy() for name in ['states', 'actions', 'next_states'])
    if len(data[0]) != 2500:
        raise ValueError('This study requires the fixed 2,500-transition dataset')
    event_index = np.flatnonzero((data[0][:, 4] == 0) & (data[2][:, 4] == 1))
    if len(event_index) != 1:
        raise ValueError('Expected exactly one naturally observed opening')
    event_index = int(event_index[0])
    event = {'index': event_index, 'state': data[0][event_index].tolist(),
             'action': int(data[1][event_index]), 'next_state': data[2][event_index].tolist()}
    runtime, source_hashes = snapshot_runtime(output)
    env = runtime['envs'].PuckLab(runtime['envs'].PuckConfig(variant='mechanism'))
    scale = np.maximum(np.maximum(np.abs(env.observation_low), np.abs(env.observation_high)), 1e-3)
    base_config = runtime['models'].EnsembleConfig(
        6, 6, members=3, hidden=64, binary_dims=(4,), observation_scale=tuple(scale),
        observation_low=tuple(env.observation_low), observation_high=tuple(env.observation_high))
    grid, motion = make_probes(runtime, event)
    np.savez_compressed(output / 'grid-probes.npz', **grid)
    np.savez_compressed(output / 'motion-probes.npz', **motion)
    results = {'experiment': 'offline_event_learning', 'command': sys.orig_argv,
               'protocol_sha256': sha256(protocol), 'dataset': str(dataset),
               'dataset_sha256': dataset_hash, 'source_sha256': source_hashes,
               'scope': 'Offline fitting on full unchanged data; not exploration evidence',
               'model_config': asdict(base_config), 'event': event,
               'model_seeds': [0, 1, 2], 'optimizer_checkpoints': [1000, 3000],
               'records': [], 'navigation': []}
    models = {}
    for mode in ['uniform', 'high_change']:
        for seed in [0, 1, 2]:
            model = runtime['models'].DynamicsEnsemble(replace(base_config, seed=seed))
            replay = OfflineReplay(data, scale, mode, seed + 101, event_index)
            results['high_change_indices'] = replay.tail.tolist()
            for budget in [1000, 3000]:
                begin = perf_counter()
                training = model.train_steps(replay, budget - model.version, batch_size=128)
                fit_seconds = perf_counter() - begin
                key = f'{mode}-seed{seed}-updates{budget}'
                metrics = evaluate(model, event, grid, motion, output / f'{key}-predictions.npz')
                record = {'candidate': mode, 'seed': seed, 'optimizer_updates': budget,
                          'new_fit_seconds': fit_seconds, 'training': training,
                          'sample_draws_per_member': replay.draws.tolist(),
                          'opening_event_draws_per_member': replay.exposure.tolist(), **metrics}
                results['records'].append(record)
                model.save(output / f'{key}.pt')
                write_json(output / 'results.json', results)
                print(json.dumps({'candidate': mode, 'seed': seed, 'updates': budget,
                                  'event_p': record['event_probability_mean'],
                                  'grid_recall': metrics['grid_all']['recall_at_half'],
                                  'outside_fp': metrics['grid_outside_interact']['false_positive_rate_at_half']}),
                      flush=True)
            models[(mode, seed)] = model
    memory = KNNResidual(data, base_config)
    metrics = evaluate(memory, event, grid, motion, output / 'knn-predictions.npz')
    results['records'].append({'candidate': 'knn', 'seed': None, 'optimizer_updates': 0, **metrics})
    models[('knn', None)] = memory
    write_json(output / 'results.json', results)
    for record in results['records']:
        if record['optimizer_updates'] not in [0, 3000]:
            continue
        if not (record['event_probability_mean'] >= .5 or record['grid_all']['recall_at_half'] > 0):
            continue
        key = (record['candidate'], record['seed'])
        for planner_seed in [40000, 40001, 40002]:
            for start_kind in ['default', 'experienced_event']:
                nav = frozen_navigation(runtime, models[key], event, planner_seed, start_kind)
                path = f'{key[0]}-seed{key[1]}-planner{planner_seed}-{start_kind}-navigation.json'
                write_json(output / path, nav)
                summary = {k: v for k, v in nav.items() if k != 'frames'}
                results['navigation'].append({'candidate': key[0], 'seed': key[1],
                                              'trajectory_file': path, **summary})
                write_json(output / 'results.json', results)
                print(json.dumps({'navigation': key, **summary}), flush=True)
    results['dataset_unchanged'] = sha256(dataset) == dataset_hash
    results['protocol_unchanged'] = sha256(protocol) == results['protocol_sha256']
    results['completed'] = True
    write_json(output / 'results.json', results)
    if not results['dataset_unchanged'] or not results['protocol_unchanged']:
        raise AssertionError('Dataset or protocol changed during the study')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/006-event-learning')
    parser.add_argument('--dataset', default='results/003-event-recall/replay.npz')
    args = parser.parse_args()
    run(args.output, args.dataset)


if __name__ == '__main__':
    main()
