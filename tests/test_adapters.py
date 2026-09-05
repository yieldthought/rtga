from dataclasses import fields

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rtga.adapters import ObservationActionAdapter, Transition


class ToyEnv(gym.Env):
    def __init__(self, reward=1.0, truncated=False, observation_space=None, observation=None,
                 action_space=None, autoreset_key=None):
        self.observation_space = observation_space or spaces.Box(-10, 10, (2,), np.float32)
        self.action_space = action_space or spaces.Discrete(3)
        self._fixed_observation = observation
        self.reward = reward
        self.truncate = truncated
        self.autoreset_key = autoreset_key
        self.steps = self.resets = 0
        self.last_action = None

    def _obs(self):
        return self._fixed_observation if self._fixed_observation is not None else np.array([self.steps, 0], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.resets += 1
        return self._obs(), {"secret": self.reward, "seed": seed}

    def step(self, action):
        self.last_action = action
        self.steps += 1
        done = self.steps == 2
        info = {"secret": self.reward, "nested": {"value": self.reward}}
        if self.autoreset_key:
            info[self.autoreset_key] = self._obs()
        return self._obs(), self.reward, done and not self.truncate, done and self.truncate, info


def test_reward_and_info_cannot_change_agent_facing_transition():
    a = ObservationActionAdapter(ToyEnv(reward=-1000))
    b = ObservationActionAdapter(ToyEnv(reward=9999))
    np.testing.assert_array_equal(a.reset(seed=4), b.reset(seed=4))
    assert {f.name for f in fields(Transition)} == {"observation", "terminated", "truncated"}
    for action in [0, 1]:
        left, right = a.step(action), b.step(action)
        np.testing.assert_array_equal(left.observation, right.observation)
        assert left.terminated == right.terminated and left.truncated == right.truncated
        assert not hasattr(left, "reward") and not hasattr(left, "info")
    assert a.evaluator_metrics()["reward"] == -1000
    metrics = a.evaluator_metrics()
    metrics["info"]["nested"]["value"] = "modified"
    assert a.evaluator_metrics()["info"]["nested"]["value"] == -1000


@pytest.mark.parametrize("truncated", [False, True])
def test_final_observation_returned_before_explicit_reset(truncated):
    env = ToyEnv(truncated=truncated)
    adapter = ObservationActionAdapter(env)
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(0)
    adapter.reset()
    adapter.step(0)
    final = adapter.step(0)
    np.testing.assert_array_equal(final.observation, [2, 0])
    assert final.truncated == truncated
    assert final.terminated == (not truncated)
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(0)
    assert env.steps == 2 and env.resets == 1
    np.testing.assert_array_equal(adapter.reset(), [0, 0])
    assert env.resets == 2
    assert adapter.evaluator_metrics()["reward"] is None
    np.testing.assert_array_equal(final.observation, [2, 0])


def test_numeric_dict_tuple_and_multidiscrete_use_declared_flattening():
    space = spaces.Dict({"position": spaces.Box(-1, 1, (2,), np.float32),
                         "other": spaces.Tuple((spaces.Discrete(3), spaces.MultiDiscrete([2, 4])))})
    obs = {"position": np.array([0.2, 0.3], np.float32), "other": (2, np.array([1, 3]))}
    adapter = ObservationActionAdapter(ToyEnv(observation_space=space, observation=obs))
    np.testing.assert_array_equal(adapter.reset(), spaces.flatten(space, obs))
    assert adapter.state_dim == spaces.flatdim(space)
    np.testing.assert_array_equal(adapter.space_low, spaces.flatten_space(space).low)
    np.testing.assert_array_equal(adapter.space_high, spaces.flatten_space(space).high)


def test_image_stack_preserves_shape_dtype_and_independence():
    pixels = np.zeros((4, 24, 32, 3), np.uint8)
    space = spaces.Box(0, 255, pixels.shape, np.uint8)
    env = ToyEnv(observation_space=space, observation=pixels)
    adapter = ObservationActionAdapter(env)
    obs = adapter.reset()
    assert adapter.observation_mode == "raw" and adapter.state_dim is None
    assert obs.shape == pixels.shape and obs.dtype == np.uint8
    obs[:] = 255
    assert not pixels.any()
    flat = ObservationActionAdapter(env, observation_mode="flatten")
    assert flat.reset().shape == (pixels.size,)


def test_nested_images_require_explicit_flatten_choice():
    space = spaces.Dict({"image": spaces.Box(0, 255, (24, 32, 3), np.uint8)})
    env = ToyEnv(observation_space=space, observation={"image": np.zeros((24, 32, 3), np.uint8)})
    with pytest.raises(ValueError, match="explicit"):
        ObservationActionAdapter(env)
    assert ObservationActionAdapter(env, observation_mode="flatten").reset().size == 24 * 32 * 3


def test_discrete_start_offset_and_multibinary_mapping_are_explicit():
    env = ToyEnv(action_space=spaces.Discrete(3, start=5))
    adapter = ObservationActionAdapter(env)
    adapter.reset()
    adapter.step(1)
    assert env.last_action == 6
    assert adapter.action_mapping == (5, 6, 7)
    binary = ObservationActionAdapter(ToyEnv(action_space=spaces.MultiBinary((2, 2))))
    assert binary.n_actions == 16
    np.testing.assert_array_equal(binary.decode_action(5), [[1, 0], [1, 0]])
    mapping = binary.action_mapping
    mapping[0][:] = 1
    np.testing.assert_array_equal(binary.decode_action(0), np.zeros((2, 2)))


def test_large_button_spaces_require_selected_actions_and_invalid_mapping_rejected():
    env = ToyEnv(action_space=spaces.MultiBinary(12))
    with pytest.raises(ValueError, match="explicit"):
        ObservationActionAdapter(env)
    adapter = ObservationActionAdapter(env, action_mapping=[np.zeros(12, np.int8), np.ones(12, np.int8)])
    assert adapter.n_actions == 2
    with pytest.raises(ValueError, match="belong"):
        ObservationActionAdapter(env, action_mapping=[np.ones(12, np.int8) * 2])


@pytest.mark.parametrize("action", [-1, 3, 0.5, True])
def test_invalid_action_never_reaches_environment(action):
    env = ToyEnv()
    adapter = ObservationActionAdapter(env)
    adapter.reset()
    with pytest.raises(ValueError):
        adapter.step(action)
    assert env.steps == 0


def test_autoreset_and_vector_wrappers_rejected():
    with pytest.raises(ValueError, match="Autoreset"):
        ObservationActionAdapter(gym.wrappers.Autoreset(ToyEnv()))
    vector = gym.vector.SyncVectorEnv([ToyEnv])
    try:
        with pytest.raises(ValueError, match="Vector"):
            ObservationActionAdapter(vector)
    finally:
        vector.close()


@pytest.mark.parametrize("key", ["final_observation", "final_obs", "terminal_observation", "final_info"])
def test_legacy_autoreset_signals_fail_instead_of_returning_reset_as_next_state(key):
    adapter = ObservationActionAdapter(ToyEnv(autoreset_key=key))
    adapter.reset()
    with pytest.raises(ValueError, match="Autoreset"):
        adapter.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(0)


def test_cartpole_runs_but_unbounded_normalization_is_not_fabricated():
    adapter = ObservationActionAdapter(gym.make("CartPole-v1"))
    try:
        observation = adapter.reset(seed=3)
        assert observation.shape == (4,)
        assert adapter.requires_scaling
        assert np.isinf(adapter.space_high).any()
        transition = adapter.step(0)
        assert transition.observation.shape == (4,)
    finally:
        adapter.close()


def test_pendulum_requires_explicit_discretization():
    env = gym.make("Pendulum-v1")
    try:
        with pytest.raises(ValueError, match="continuous"):
            ObservationActionAdapter(env)
        mapping = [np.array([torque], np.float32) for torque in [-2, 0, 2]]
        adapter = ObservationActionAdapter(env, action_mapping=mapping)
        assert adapter.reset(seed=1).shape == (3,)
        assert adapter.step(2).observation.shape == (3,)
    finally:
        env.close()
