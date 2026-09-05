from copy import deepcopy

import numpy as np
import pytest

from rtga.envs import ExactPuckModel, PuckConfig, PuckLab


def set_state(env, values):
    snapshot = env.snapshot()
    snapshot["state"] = np.array(values, dtype=float)
    env.restore(snapshot)


@pytest.mark.parametrize("variant", ["open", "rooms", "mechanism", "noise"])
def test_all_observations_respect_bounds(variant):
    env = PuckLab(PuckConfig(variant=variant), seed=12)
    rng = np.random.default_rng(7)
    for action in rng.integers(0, 6, size=1200):
        observation = env.step(int(action))
        assert np.all(observation >= env.observation_low)
        assert np.all(observation <= env.observation_high)
        if variant != "open":
            clearance = env.config.wall_half_width + env.config.radius
            if abs(observation[0] - env.config.wall_x) < clearance - 1e-12:
                assert observation[4] == 1
                low, high = env.config.doorway_y
                assert low + env.config.radius <= observation[1] <= high - env.config.radius
    observation[:] = 99
    assert np.all(env.observe() <= env.observation_high)


@pytest.mark.parametrize("action,axis,bound", [(1, 0, "low"), (2, 0, "high"), (3, 1, "high"), (4, 1, "low")])
def test_outer_collision_stops_normal_velocity(action, axis, bound):
    env = PuckLab()
    for _ in range(150):
        state = env.step(action)
    expected = env.config.radius if bound == "low" else 1 - env.config.radius
    assert state[axis] == pytest.approx(expected)
    assert state[axis + 2] == 0


@pytest.mark.parametrize("variant", ["open", "rooms", "mechanism"])
def test_batched_rollout_matches_real_trajectories_without_mutating_env(variant):
    env = PuckLab(PuckConfig(variant=variant), seed=1)
    before = env.snapshot()
    plans = np.random.default_rng(8).integers(0, 6, size=(7, 100))
    plans_before = plans.copy()
    model = ExactPuckModel(env)
    predicted = model.rollout(env.observe(), plans)
    for i, plan in enumerate(plans):
        replay = env.clone()
        actual = np.array([replay.step(int(action)) for action in plan])
        np.testing.assert_array_equal(actual, predicted[i])
    np.testing.assert_array_equal(env.observe(), before["state"])
    assert env.snapshot()["rng_state"] == before["rng_state"]
    assert env.metrics()["steps"] == 0
    np.testing.assert_array_equal(plans, plans_before)
    assert model.model_calls == plans.size


def test_rooms_doorway_passes_but_wall_blocks_and_cannot_tunnel():
    env = PuckLab(PuckConfig(variant="rooms"))
    for _ in range(65):
        state = env.step(2)
    assert state[0] > env.config.wall_x
    set_state(env, [0.2, 0.2, 0, 0, 1, 0])
    for _ in range(100):
        state = env.step(2)
    assert state[0] == pytest.approx(0.465)
    assert state[2] == 0
    fast = PuckLab(PuckConfig(variant="rooms", dt=4.0))
    set_state(fast, [0.2, 0.2, 0, 0, 1, 0])
    assert fast.step(2)[0] == pytest.approx(0.465)


def test_closed_mechanism_door_requires_nearby_interaction():
    env = PuckLab(PuckConfig(variant="mechanism"))
    assert env.step(5)[4] == 0  # Default start is outside switch radius.
    for _ in range(80):
        state = env.step(2)
    assert state[0] < env.config.wall_x
    set_state(env, [*env.config.switch, 0, 0, 0, 0])
    assert env.step(0)[4] == 0
    assert env.step(5)[4] == 1
    assert env.step(5)[4] == 1  # The mechanism latches; it does not toggle.
    for _ in range(65):
        state = env.step(2)
    assert state[0] > env.config.wall_x


def test_noise_trigger_snapshot_and_clone_preserve_rng():
    env = PuckLab(PuckConfig(variant="noise"), seed=45)
    assert env.step(5)[5] == 0
    set_state(env, [*env.config.noise_source, 0, 0, 0, 0])
    initial = env.snapshot()
    first = env.step(5)
    assert first[5] != 0
    assert env.step(0)[5] == first[5]
    expected = [env.step(5)[5] for _ in range(5)]
    env.restore(initial)
    np.testing.assert_array_equal(env.step(5), first)
    env.step(0)
    clone = env.clone()
    assert [env.step(5)[5] for _ in range(5)] == expected
    assert [clone.step(5)[5] for _ in range(5)] == expected
    clone.step(5)
    assert clone.metrics()["steps"] == env.metrics()["steps"] + 1
    assert env.metrics()["noise_interactions"] == 6


def test_exact_noise_model_uses_mean_and_leaves_rng_unchanged():
    env = PuckLab(PuckConfig(variant="noise"), seed=3)
    set_state(env, [*env.config.noise_source, 0, 0, 0, 0.4])
    before = deepcopy(env.snapshot())
    model = ExactPuckModel(env)
    state = env.observe()[None, :]
    assert model.predict(state, np.array([0]))[0, 5] == 0.4
    assert model.predict(state, np.array([5]))[0, 5] == 0.0
    np.testing.assert_array_equal(env.observe(), before["state"])
    assert env.snapshot()["rng_state"] == before["rng_state"]


def test_metrics_are_separate_from_observation_and_reset_counters():
    env = PuckLab()
    assert env.observe().shape == (6,)
    for _ in range(20):
        env.step(2)
    assert env.metrics()["steps"] == 20
    assert env.metrics()["visited_cells"] > 1
    env.reset()
    assert env.metrics()["steps"] == 0
    assert env.metrics()["visited_cells"] == 1


@pytest.mark.parametrize("action", [-1, 6, 1.2])
def test_invalid_action_rejected(action):
    with pytest.raises(ValueError):
        PuckLab().step(action)


def test_invalid_config_and_empty_rollouts():
    with pytest.raises(ValueError):
        PuckConfig(variant="unknown")
    with pytest.raises(ValueError):
        PuckConfig(variant="mechanism", start=(0.5, 0.5))
    env = PuckLab()
    assert ExactPuckModel(env).rollout(env.observe(), np.zeros((3, 0), dtype=int)).shape == (3, 0, 6)
