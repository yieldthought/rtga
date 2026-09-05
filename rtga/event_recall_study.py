"""Replay the unchanged seed-0 random run while observing rare-event recall.

The evaluator reconstructs one known real opening transition independently.
An optimizer post-step hook reads its prediction after every weight update;
the probe never enters replay or action selection. A read-only wrapper counts
the event in batches returned by the original replay sampler, without making
additional RNG calls. The original agent, model, and training calls are used.
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys

import numpy as np
import torch

from .agent import AgentConfig, RTGAAgent
from .envs import PuckConfig, PuckLab
from .experiments import environment_description, provenance, write_json
from .models import DynamicsEnsemble
from .planning import PlannerConfig


def load_runtime(source):
    """Import an isolated, hash-checked source snapshot without changing live code."""
    package = '_rtga_event_recall_reference'
    spec = importlib.util.spec_from_file_location(
        package, source / '__init__.py', submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return {name: importlib.import_module(f'{package}.{name}')
            for name in ['agent', 'models', 'planning', 'envs']}


def reconstruct_event(runtime, steps=2500):
    """Evaluator-only replay of the reference random action stream."""
    env = runtime['envs'].PuckLab(runtime['envs'].PuckConfig(variant='mechanism'), seed=3001)
    rng = np.random.default_rng(201)
    transitions, events = [], []
    for step in range(steps):
        state = env.observe()
        action = int(rng.integers(env.n_actions))
        target = env.step(action)
        item = {'step': step, 'state': state.tolist(), 'action': action,
                'next_state': target.tolist()}
        transitions.append(item)
        if state[4] == 0 and target[4] == 1:
            events.append(item)
    if len(events) != 1:
        raise AssertionError(f'Expected one opening event, found {len(events)}')
    return events[0], transitions


class RecallObserver:
    """Observe predictions and already-drawn replay batches without feedback."""

    def __init__(self, agent, event):
        self.agent = agent
        self.state = np.asarray(event['state'], dtype=np.float32)
        self.target = np.asarray(event['next_state'], dtype=np.float32)
        self.action = event['action']
        self.records = []
        self.batch_occurrences = np.zeros(agent.config.members, dtype=np.int64)
        self.total_occurrences = self.batch_occurrences.copy()
        self.updates_with_event = self.batch_occurrences.copy()
        self.original_sample = agent.replay.sample
        agent.replay.sample = self.sample
        self.hook = agent.model.optimizer.register_step_post_hook(self.after_update)
        self.record(0, 'initial')

    def sample(self, batch_size, members=1):
        batch = self.original_sample(batch_size, members)
        states, actions, targets = batch
        matches = ((states == self.state).all(axis=-1)
                   & (actions == self.action)
                   & (targets == self.target).all(axis=-1))
        self.batch_occurrences = matches.sum(axis=1)
        self.total_occurrences += self.batch_occurrences
        self.updates_with_event += self.batch_occurrences > 0
        return batch

    def after_update(self, optimizer, args, kwargs):
        # Core train_steps increments version immediately after optimizer.step.
        self.record(self.agent.model.version + 1, 'after_optimizer_step')

    def record(self, version, kind):
        model = self.agent.model
        means, variances = model.predict(self.state[None], np.array([self.action]))
        normalized = means[:, 0] / np.asarray(model.config.observation_scale)
        door_probability = means[:, 0, 4]
        self.records.append({
            'kind': kind, 'model_version': version,
            'observed_transitions': self.agent.transitions,
            'replay_size': len(self.agent.replay),
            'opening_probability_members': door_probability.tolist(),
            'opening_probability_mean': float(door_probability.mean()),
            'door_disagreement': float(door_probability.var()),
            'normalized_disagreement': float(normalized.var(axis=0).mean()),
            'member_next_state_means': means[:, 0].tolist(),
            'member_next_state_variances': variances[:, 0].tolist(),
            'batch_event_occurrences': self.batch_occurrences.tolist(),
            'cumulative_event_occurrences': self.total_occurrences.tolist(),
        })

    def close(self):
        self.hook.remove()
        self.agent.replay.sample = self.original_sample


def compare_checkpoints(actual, expected):
    """Compare all parameters, optimizer moments, configuration, and version."""
    mismatches = []
    tensor_count = 0
    max_tensor_error = 0.0

    def visit(a, b, path):
        nonlocal tensor_count, max_tensor_error
        if torch.is_tensor(a) and torch.is_tensor(b):
            tensor_count += 1
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                mismatches.append(path)
            if a.shape == b.shape and a.numel():
                max_tensor_error = max(max_tensor_error, float((a.double() - b.double()).abs().max()))
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                mismatches.append(path + '.keys')
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                mismatches.append(path + '.length')
            for index, (left, right) in enumerate(zip(a, b)):
                visit(left, right, f'{path}[{index}]')
        elif a != b:
            mismatches.append(path)

    visit(actual, expected, 'checkpoint')
    return {'exact_match': not mismatches, 'tensor_count': tensor_count,
            'max_tensor_absolute_error': max_tensor_error, 'mismatches': mismatches}


def recall_summary(observer, event):
    post = [row for row in observer.records
            if row['observed_transitions'] >= event['step'] + 1]
    values = np.array([row['opening_probability_members'] for row in post])
    means = values.mean(axis=1)
    peak = int(means.argmax())
    return {
        'post_event_optimizer_updates': len(post),
        'first_post_event_update': post[0],
        'peak_ensemble_probability': float(means[peak]),
        'peak_ensemble_probability_record': post[peak],
        'peak_member_probabilities': values.max(axis=0).tolist(),
        'member_updates_at_or_above_half': (values >= .5).sum(axis=0).tolist(),
        'ensemble_updates_at_or_above_half': int((means >= .5).sum()),
        'last_record': post[-1],
        'event_batch_occurrences_per_member': observer.total_occurrences.tolist(),
        'updates_containing_event_per_member': observer.updates_with_event.tolist(),
        'interpretation_limit': 'Recall concerns this one experienced state/action; '
                                'it does not establish a general switch rule.',
    }


def plot_recall(records, event, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [row for row in records if row['observed_transitions'] >= event['step'] + 1]
    versions = np.array([row['model_version'] for row in rows])
    probabilities = np.array([row['opening_probability_members'] for row in rows])
    occurrences = np.array([row['cumulative_event_occurrences'] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout='constrained')
    colors = ['#2563a6', '#c15b25', '#3e805c']
    for member, color in enumerate(colors):
        axes[0].plot(versions, probabilities[:, member], color=color, linewidth=1,
                     label=f'Member {member + 1}')
        axes[1].plot(versions, occurrences[:, member], color=color, linewidth=1)
    axes[0].axhline(.5, color='#888888', linestyle='--', linewidth=.8,
                    label='Planner binary threshold')
    axes[0].set(ylabel='Predicted opening probability', ylim=(0, .52),
                title='Exact experienced opening event: after every optimizer update')
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set(xlabel='Optimizer update / model version', ylabel='Cumulative event samples')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#eeeeee', linewidth=.7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output, reference, core_source=None):
    torch.set_num_threads(1)
    output, reference = Path(output), Path(reference)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((reference / 'results.json').read_text())
    if (original['seed'], original['mode'], original['variant'], original['steps_requested']) != (
            0, 'random', 'mechanism', 2500):
        raise ValueError('This diagnostic requires the original seed-0 random mechanism run')
    source = output / 'source'
    source.mkdir(exist_ok=True)
    core_source = Path(core_source) if core_source else Path(__file__).parent
    for name in ['__init__.py', 'agent.py', 'models.py', 'planning.py', 'envs.py',
                 'experiments.py', 'online_experiment.py', 'event_recall_study.py']:
        origin = Path(__file__) if name == 'event_recall_study.py' else core_source / name
        if origin.resolve() != (source / name).resolve():
            shutil.copy2(origin, source / name)
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ['agent.py', 'models.py', 'planning.py', 'envs.py']}
    source_match = {name: value == original['provenance']['source_sha256'][name]
                    for name, value in hashes.items()}
    if not all(source_match.values()):
        raise AssertionError(f'Core source differs from reference: {source_match}')
    runtime = load_runtime(source.resolve())
    config_data = original['config'].copy()
    config_data['planner'] = runtime['planning'].PlannerConfig(**config_data['planner'])
    config = runtime['agent'].AgentConfig(**config_data)
    event, expected_transitions = reconstruct_event(runtime)
    env = runtime['envs'].PuckLab(runtime['envs'].PuckConfig(variant='mechanism'), seed=3001)
    agent = runtime['agent'].RTGAAgent(env.observation_low, env.observation_high,
                                     env.n_actions, config, binary_dims=(4,))
    observer = RecallObserver(agent, event)
    frames = []
    saved_frames = {row['step']: row for row in json.loads(
        (reference / 'trace.json').read_text())['frames']}
    compared_fields = ['state', 'action', 'metrics', 'model_version',
                       'prediction_error', 'disagreement']
    trace_mismatches = []
    observation = env.observe()
    try:
        for step in range(2500):
            action = agent.act(observation, is_first=step == 0)
            target = env.step(action)
            expected = expected_transitions[step]
            if (action != expected['action']
                    or not np.array_equal(observation, expected['state'])
                    or not np.array_equal(target, expected['next_state'])):
                raise AssertionError(f'Independent random trajectory mismatch at step {step}')
            info = agent.last_info
            frame = {'step': step, 'state': observation.tolist(), 'action': action,
                     'next_state': target.tolist(), 'metrics': env.metrics(),
                     'model_version': agent.model.version,
                     'prediction_error': info['prediction_error_before_update'],
                     'disagreement': info['disagreement_before_update']}
            frames.append(frame)
            if step in saved_frames:
                for field in compared_fields:
                    if frame[field] != saved_frames[step][field]:
                        trace_mismatches.append({'step': step, 'field': field})
            observation = target
        agent.observe_final(observation)
    finally:
        observer.close()
    checkpoint_comparison = compare_checkpoints(agent.model.state_dict(),
        runtime['models'].DynamicsEnsemble.load(reference / 'model.pt').state_dict())
    validation = {
        'core_source_hashes_match_reference': source_match,
        'independent_random_trajectory_exact_match': True,
        'full_transitions_checked': len(frames),
        'reference_sampled_trace_frames_checked': len(saved_frames),
        'reference_trace_fields_checked': compared_fields,
        'reference_trace_mismatches': trace_mismatches,
        'checkpoint': checkpoint_comparison,
        'final_metrics_match': env.metrics() == original['final_metrics'],
        'training_update_count_match': agent.model.version == original['training_updates'],
    }
    validated = (not trace_mismatches and checkpoint_comparison['exact_match']
                 and validation['final_metrics_match'] and validation['training_update_count_match'])
    states, actions, targets = agent.replay.transitions()
    event_count = int((((states == observer.state).all(axis=1))
                       & (actions == observer.action)
                       & (targets == observer.target).all(axis=1)).sum())
    data = {
        'experiment': 'exact_event_recall', 'provenance': provenance(),
        'executed_source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source.glob('*.py')},
        'reference': str(reference), 'protocol': {
            'seed': 0, 'variant': 'mechanism', 'actual_steps': 2500,
            'config': asdict(config), 'probe_role': 'Evaluator-only; no probe injection',
            'logging': 'After every individual optimizer step, using a post-step hook',
            'exposure_counting': 'Inspect already-sampled batches; no additional RNG calls',
            'learned_event_criterion': 'Member next-door probability >= 0.5, matching planner threshold',
        },
        'event': event, 'opening_real_step_one_based': event['step'] + 1,
        'validation': validation, 'validated_original_prefix': validated,
        'replay_event_count': event_count,
        'actual_opening_transition_count': int(((states[:, 4] == 0) & (targets[:, 4] == 1)).sum()),
        'final_metrics': env.metrics(), 'summary': recall_summary(observer, event),
    }
    write_json(output / 'results.json', data)
    write_json(output / 'recall.json', {'event': event, 'records': observer.records})
    write_json(output / 'trajectory.json', {'schema_version': 1,
               'metadata': {'title': 'Unchanged random seed-0 replay', 'config': asdict(config)},
               'environment': environment_description(env), 'frames': frames})
    np.savez_compressed(output / 'replay.npz', states=states, actions=actions, next_states=targets)
    agent.model.save(output / 'model.pt')
    plot_recall(observer.records, event, output / 'event-recall.png')
    print(json.dumps({'validated_original_prefix': validated, 'validation': validation,
                      'summary': data['summary']}, indent=2), flush=True)
    if not validated:
        raise AssertionError('Original prefix did not match; inspect saved validation before interpreting')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/003-event-recall')
    parser.add_argument('--reference', default='runs/random-mechanism-dev0')
    parser.add_argument('--core-source', help='Directory containing the original rtga source files')
    args = parser.parse_args()
    run(args.output, args.reference, args.core_source)


if __name__ == '__main__':
    main()
