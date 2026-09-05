from dataclasses import replace

import numpy as np
import pytest

from rtga.agent import AgentConfig
from rtga.envs import PuckConfig, PuckLab
from rtga.models import DynamicsEnsemble
from rtga.online_experiment import noise_probe_transitions, run_online
from rtga.planning import PlannerConfig


def small_config(**overrides):
    config = AgentConfig(warmup=2, initial_fit=1, train_every=1, train_steps=1,
                         batch_size=8, hidden=8, members=3,
                         planner=PlannerConfig(population=4, horizon=2, generations=2))
    return replace(config, **overrides)


def test_all_inference_and_real_interactions_are_counted(tmp_path, monkeypatch):
    observed = {'inference': 0, 'environment': 0}
    original_predict = DynamicsEnsemble.predict_tensor
    original_step = PuckLab.step

    def count_predict(self, states, actions):
        observed['inference'] += self.config.members * len(states)
        return original_predict(self, states, actions)

    def count_step(self, action):
        observed['environment'] += 1
        return original_step(self, action)

    monkeypatch.setattr(DynamicsEnsemble, 'predict_tensor', count_predict)
    monkeypatch.setattr(PuckLab, 'step', count_step)
    data = run_online(tmp_path, mode='curiosity', variant='noise', steps=5,
                      agent_config=small_config(), checkpoint_every=2,
                      goal_evaluation_steps=3)
    assert data['planning_model_transitions'] == 3 * (3 * 3 * 4 * 2 * 2)
    assert data['model_transitions'] == data['planning_model_transitions']
    assert data['surprise_model_transitions'] == 2 * 3
    assert data['probe_model_transitions'] == 3 * 72 * 3
    assert data['noise_probe_model_transitions'] == 3 * 32 * 3
    assert data['frozen_goal_model_transitions'] == 3 * 3 * (3 * 3 * 4 * 2 * 2)
    assert data['total_model_inference_transitions'] == observed['inference'] == 2670
    assert data['total_environment_interactions'] == observed['environment'] == 5 + 72 + 32 + 9
    assert data['training_updates'] == data['final_model_version'] == 3
    assert data['training_member_samples'] == 3 * 8 * 3
    assert data['evaluation_seconds'] == pytest.approx(
        data['probe_generation_seconds'] + data['checkpoint_evaluation_seconds']
        + data['noise_probe_evaluation_seconds'] + data['frozen_goal_evaluation_seconds'])
    assert data['run_wall_seconds'] >= data['evaluation_seconds']
    assert data['noise_probe_interactions'] == 32
    assert data['checkpoints'][-1]['noise_probe']['observed_sensor_variance'] > 0


def test_last_transition_is_recorded_without_changing_fit_semantics(tmp_path):
    data = run_online(tmp_path, mode='random', steps=3,
                      agent_config=small_config(warmup=3, initial_fit=2),
                      checkpoint_every=2, evaluate_goals=False)
    assert [checkpoint['consumed_transitions'] for checkpoint in data['checkpoints']] == [1, 2]
    assert all(checkpoint['pending_transition_count'] == 1 for checkpoint in data['checkpoints'])
    assert data['consumed_transitions'] == data['replay_size'] == 3
    assert data['pending_transition_count'] == 0
    assert data['final_model_version'] == data['training_updates'] == 0
    assert data['model_last_fit_consumed_transitions'] is None
    replay = np.load(tmp_path / 'actual_transitions.npz')
    assert len(replay['actions']) == 3
    np.testing.assert_array_equal(replay['next_states'][:-1], replay['states'][1:])


def test_frozen_evaluation_does_not_change_collection_or_fitting(tmp_path):
    outputs = []
    for evaluate_goals in (False, True):
        directory = tmp_path / str(evaluate_goals)
        outputs.append(run_online(directory, seed=7, mode='curiosity', variant='noise', steps=5,
                                  agent_config=small_config(), checkpoint_every=3,
                                  evaluate_goals=evaluate_goals, goal_evaluation_steps=2))
    left = np.load(tmp_path / 'False' / 'actual_transitions.npz')
    right = np.load(tmp_path / 'True' / 'actual_transitions.npz')
    for name in ('states', 'actions', 'next_states'):
        np.testing.assert_array_equal(left[name], right[name])
    assert len(left['actions']) == len(right['actions']) == 5
    first = DynamicsEnsemble.load(tmp_path / 'False' / 'model.pt')
    second = DynamicsEnsemble.load(tmp_path / 'True' / 'model.pt')
    for a, b in zip(first.predict(left['states'], left['actions']),
                    second.predict(right['states'], right['actions'])):
        np.testing.assert_array_equal(a, b)
    assert outputs[0]['final_model_version'] == outputs[1]['final_model_version']
    assert outputs[1]['total_environment_interactions'] - outputs[0]['total_environment_interactions'] == 6


def test_noise_probes_repeat_inputs_but_advance_outcome_rng():
    probes = noise_probe_transitions(PuckConfig(variant='noise'), seed=5, repeats=32)
    states, actions, targets = probes
    np.testing.assert_array_equal(states, np.broadcast_to(states[0], states.shape))
    assert np.all(actions == 5)
    assert np.all((targets[:, 5] >= -1) & (targets[:, 5] <= 1))
    assert len(np.unique(targets[:, 5])) == 32
    for expected, actual in zip(probes, noise_probe_transitions(PuckConfig(variant='noise'), 5, 32)):
        np.testing.assert_array_equal(expected, actual)
    assert noise_probe_transitions(PuckConfig(variant='open'), seed=5) is None


def test_checkpoint_probe_frequency_does_not_change_replay_or_model(tmp_path):
    outputs = []
    for frequency in (1, 5):
        outputs.append(run_online(tmp_path / str(frequency), seed=11, mode='reactive',
                                  variant='noise', steps=5, agent_config=small_config(),
                                  checkpoint_every=frequency, evaluate_goals=False))
    frequent = np.load(tmp_path / '1' / 'actual_transitions.npz')
    sparse = np.load(tmp_path / '5' / 'actual_transitions.npz')
    for name in ('states', 'actions', 'next_states'):
        np.testing.assert_array_equal(frequent[name], sparse[name])
    first = DynamicsEnsemble.load(tmp_path / '1' / 'model.pt')
    second = DynamicsEnsemble.load(tmp_path / '5' / 'model.pt')
    for a, b in zip(first.predict(frequent['states'], frequent['actions']),
                    second.predict(sparse['states'], sparse['actions'])):
        np.testing.assert_array_equal(a, b)
    assert outputs[0]['planning_model_transitions'] == outputs[1]['planning_model_transitions'] == 3 * 3 * 6
    assert outputs[0]['probe_model_transitions'] == 5 * outputs[1]['probe_model_transitions']


@pytest.mark.parametrize('kwargs', [{'steps': 0}, {'steps': -1}, {'checkpoint_every': 0},
                                   {'checkpoint_every': -2}, {'goal_evaluation_steps': 0},
                                   {'noise_probe_repeats': 0}])
def test_invalid_run_lengths_fail_before_collecting(tmp_path, kwargs):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        run_online(output, **kwargs)
    assert not output.exists()
