"""Gymnasium observations/actions with a separate evaluator-only metric channel.

This adapter does not infer rewards, normalize observations, clip unbounded
spaces, or reset automatically. Pass only observations and episode boundaries to
an agent. ``evaluator_metrics()`` is for the experiment runner, never the agent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    terminated: bool
    truncated: bool


def _has_multidimensional_box(space: spaces.Space) -> bool:
    if isinstance(space, spaces.Box):
        return len(space.shape) >= 2
    if isinstance(space, spaces.Dict):
        return any(_has_multidimensional_box(s) for s in space.spaces.values())
    if isinstance(space, spaces.Tuple):
        return any(_has_multidimensional_box(s) for s in space.spaces)
    return False


def _numeric_space(space: spaces.Space) -> bool:
    if isinstance(space, (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    if isinstance(space, spaces.Dict):
        return all(_numeric_space(s) for s in space.spaces.values())
    if isinstance(space, spaces.Tuple):
        return all(_numeric_space(s) for s in space.spaces)
    return False


class ObservationActionAdapter:
    """A single, manually reset Gymnasium environment with indexed actions.

    ``auto`` preserves multidimensional Box observations (including image stacks)
    and flattens numeric vector/Dict/Tuple/discrete observations. Nested image
    spaces require an encoder or an explicit ``flatten`` choice. ``raw`` accepts
    Box observations and preserves shape and dtype. Flattening uses Gymnasium's
    one-hot encoding for Discrete/MultiDiscrete components.

    Discrete actions retain their declared ``start`` offset. MultiBinary spaces
    up to eight bits enumerate indices little-endian: bit zero controls the first
    flattened button. Larger spaces require an explicit finite action mapping.
    Continuous actions are unsupported unless the caller explicitly supplies a
    finite mapping of valid actions, thereby choosing a discretization.

    Bounds remain exactly as declared, including infinities. ``requires_scaling``
    flags unbounded coordinates; callers must choose suitable learned or explicit
    normalization before passing them to a model requiring finite bounds. Raw
    images likewise need a pixel encoder before the current vector RTGAAgent.
    """

    def __init__(self, env: gym.Env, observation_mode: str = "auto",
                 action_mapping: Sequence[Any] | None = None):
        if isinstance(env, gym.vector.VectorEnv):
            raise ValueError("Vector environments are unsupported; wrap one manually reset environment")
        current = env
        while True:
            if isinstance(current, gym.wrappers.Autoreset) or type(current).__name__ == "AutoResetWrapper":
                raise ValueError("Autoreset wrappers are unsupported; reset explicitly after consuming the final observation")
            if not isinstance(current, gym.Wrapper):
                break
            current = current.env
        if observation_mode not in {"auto", "flatten", "raw"}:
            raise ValueError("observation_mode must be auto, flatten, or raw")
        self._env = env
        self.original_observation_space = env.observation_space
        self.original_action_space = env.action_space
        if not _numeric_space(env.observation_space):
            raise ValueError("Only numeric Box/Discrete/MultiDiscrete/MultiBinary/Dict/Tuple observations are supported")
        if observation_mode == "auto":
            if isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) >= 2:
                observation_mode = "raw"
            elif _has_multidimensional_box(env.observation_space):
                raise ValueError("Nested multidimensional observations require an encoder or explicit observation_mode='flatten'")
            else:
                observation_mode = "flatten"
        self.observation_mode = observation_mode
        if observation_mode == "raw":
            if not isinstance(env.observation_space, spaces.Box):
                raise ValueError("Raw observations require a Box space")
            self.observation_space = deepcopy(env.observation_space)
        else:
            self.observation_space = spaces.flatten_space(env.observation_space)
        self.space_low = self.observation_space.low.copy()
        self.space_high = self.observation_space.high.copy()
        self.observation_low = self.space_low
        self.observation_high = self.space_high
        self.observation_shape = self.observation_space.shape
        self.observation_dtype = self.observation_space.dtype
        self.state_dim = self.observation_shape[0] if len(self.observation_shape) == 1 else None
        self.requires_scaling = not bool(np.all(np.isfinite(self.space_low)) and np.all(np.isfinite(self.space_high)))
        self._action_mapping = self._make_action_mapping(action_mapping)
        self.n_actions = len(self._action_mapping)
        self.action_space = spaces.Discrete(self.n_actions)
        self._needs_reset = True
        self._metrics: dict[str, Any] = {"reward": None, "info": {}}

    def _make_action_mapping(self, mapping: Sequence[Any] | None) -> tuple[Any, ...]:
        space = self.original_action_space
        if mapping is None:
            if isinstance(space, spaces.Discrete):
                mapping = tuple(range(int(space.start), int(space.start + space.n)))
            elif isinstance(space, spaces.MultiBinary):
                bits = int(np.prod(space.shape))
                if bits > 8:
                    raise ValueError("MultiBinary spaces above eight bits require an explicit action_mapping")
                mapping = tuple(
                    ((i >> np.arange(bits)) & 1).astype(space.dtype).reshape(space.shape)
                    for i in range(2**bits)
                )
            else:
                raise ValueError("Non-discrete actions require an explicit finite action_mapping; continuous planning is unsupported")
        if len(mapping) == 0:
            raise ValueError("action_mapping cannot be empty")
        if any(not space.contains(action) for action in mapping):
            raise ValueError("Every mapped action must belong to the original action_space")
        return tuple(deepcopy(action) for action in mapping)

    @property
    def action_mapping(self) -> tuple[Any, ...]:
        """A defensive copy declaring exactly what each integer action means."""
        return deepcopy(self._action_mapping)

    def decode_action(self, action: int) -> Any:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer index")
        if not 0 <= action < self.n_actions:
            raise ValueError("Action index outside adapter action_space")
        return deepcopy(self._action_mapping[int(action)])

    def _observation(self, observation: Any) -> np.ndarray:
        if not self.original_observation_space.contains(observation):
            raise ValueError("Environment observation is outside its declared space")
        if self.observation_mode == "flatten":
            observation = spaces.flatten(self.original_observation_space, observation)
        result = np.array(observation, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError("Observed numeric values must be finite")
        return result

    @staticmethod
    def _reject_autoreset_info(info: dict[str, Any]) -> None:
        if any(key in info for key in ("final_observation", "final_obs", "terminal_observation", "final_info")):
            raise ValueError("Autoreset/final-observation info is unsupported; use an environment with explicit resets")

    def reset(self, seed: int | None = None, *, options: dict[str, Any] | None = None) -> np.ndarray:
        self._needs_reset = True
        observation, info = self._env.reset(seed=seed, options=options)
        self._reject_autoreset_info(info)
        result = self._observation(observation)
        self._metrics = {"reward": None, "info": deepcopy(info)}
        self._needs_reset = False
        return result

    def step(self, action: int) -> Transition:
        if self._needs_reset:
            raise RuntimeError("Call reset before stepping, including after termination or truncation")
        decoded = self.decode_action(action)
        # If a step or conversion raises, require reset rather than assume that
        # the underlying environment did not advance.
        self._needs_reset = True
        observation, reward, terminated, truncated, info = self._env.step(decoded)
        self._reject_autoreset_info(info)
        result = self._observation(observation)
        self._metrics = {"reward": float(reward), "info": deepcopy(info)}
        self._needs_reset = bool(terminated or truncated)
        return Transition(result, bool(terminated), bool(truncated))

    def evaluator_metrics(self) -> dict[str, Any]:
        """Last reset/step's external reward and info, for evaluation only."""
        return deepcopy(self._metrics)

    def close(self) -> None:
        self._env.close()
        self._needs_reset = True
