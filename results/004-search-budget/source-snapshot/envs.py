"""Small observable puck worlds and a batched diagnostic simulator.

Positions use a unit square with positive y pointing up. A puck has axis-aligned
clearance ``radius`` around walls; collision response removes normal velocity.
The six observation coordinates are the entire physical state. Coverage and
interaction counters are available separately for evaluation, never as reward.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PuckConfig:
    variant: str = "open"
    dt: float = 0.05
    acceleration: float = 1.2
    drag: float = 0.88
    max_speed: float = 0.5
    radius: float = 0.015
    start: tuple[float, float] = (0.20, 0.50)
    wall_x: float = 0.50
    wall_half_width: float = 0.02
    doorway_y: tuple[float, float] = (0.38, 0.62)
    switch: tuple[float, float] = (0.30, 0.50)
    noise_source: tuple[float, float] = (0.20, 0.80)
    interaction_radius: float = 0.09
    coverage_bins: int = 16

    def __post_init__(self) -> None:
        if self.variant not in {"open", "rooms", "mechanism", "noise"}:
            raise ValueError("variant must be open, rooms, mechanism, or noise")
        if not (self.dt > 0 and self.acceleration >= 0 and self.max_speed > 0):
            raise ValueError("dt/max_speed must be positive; acceleration nonnegative")
        if not (0 <= self.drag <= 1 and 0 < self.radius < 0.25):
            raise ValueError("drag must be in [0,1] and radius in (0,.25)")
        if not (self.radius < self.wall_x - self.wall_half_width
                < self.wall_x + self.wall_half_width < 1 - self.radius):
            raise ValueError("wall must have positive width and fit inside the world")
        low, high = self.doorway_y
        if not (0 < low < high < 1 and high - low > 2 * self.radius):
            raise ValueError("doorway must fit the puck and lie inside the world")
        for name in ("start", "switch", "noise_source"):
            point = np.asarray(getattr(self, name), dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError(f"{name} must be a finite 2D point")
            if np.any(point < self.radius) or np.any(point > 1 - self.radius):
                raise ValueError(f"{name} must lie inside the world with puck clearance")
        if not (self.interaction_radius > 0 and self.coverage_bins >= 1
                and isinstance(self.coverage_bins, int)):
            raise ValueError("interaction_radius must be positive; coverage_bins an integer")
        if self.variant != "open":
            x, y = self.start
            in_wall = abs(x - self.wall_x) < self.wall_half_width + self.radius
            in_gap = low + self.radius <= y <= high - self.radius
            if in_wall and (self.variant != "rooms" or not in_gap):
                raise ValueError("start intersects the wall or closed door")


ACTION_NAMES = ("noop", "left", "right", "up", "down", "interact")
ACTION_VECTORS = np.array(
    [[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=np.float64
)
ACTION_VECTORS.setflags(write=False)


def _bounds(config: PuckConfig) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([0, 0, -config.max_speed, -config.max_speed, 0, -1], dtype=float),
        np.array([1, 1, config.max_speed, config.max_speed, 1, 1], dtype=float),
    )


def _actions(actions: np.ndarray, batch: int) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.shape != (batch,) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions must be an integer array with shape (batch,)")
    if np.any(actions < 0) or np.any(actions >= len(ACTION_NAMES)):
        raise ValueError("action index outside [0,5]")
    return actions


def _states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6 or not np.all(np.isfinite(states)):
        raise ValueError("states must be a finite array with shape (batch,6)")
    return states


def _transitions(
    config: PuckConfig, states: np.ndarray, actions: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized physics; a missing RNG uses zero for newly sampled sensor noise."""
    states = _states(states)
    actions = _actions(actions, len(states))
    result = states.copy()
    interact = actions == 5
    near_switch = np.linalg.norm(states[:, :2] - config.switch, axis=1) <= config.interaction_radius
    near_noise = np.linalg.norm(states[:, :2] - config.noise_source, axis=1) <= config.interaction_radius
    switch_hits = interact & near_switch & (config.variant in {"mechanism", "noise"})
    noise_hits = interact & near_noise & (config.variant == "noise")
    if config.variant in {"open", "rooms"}:
        result[:, 4] = 1.0
    else:
        result[switch_hits, 4] = 1.0
    if rng is not None:
        result[noise_hits, 5] = rng.uniform(-1, 1, size=int(noise_hits.sum()))
    else:
        result[noise_hits, 5] = 0.0

    velocity = config.drag * states[:, 2:4] + config.acceleration * config.dt * ACTION_VECTORS[actions]
    velocity = np.clip(velocity, -config.max_speed, config.max_speed)
    old_position = states[:, :2]
    position = old_position + config.dt * velocity
    outer_hit = (position < config.radius) | (position > 1 - config.radius)
    position = np.clip(position, config.radius, 1 - config.radius)
    velocity[outer_hit] = 0.0

    if config.variant != "open":
        x0 = config.wall_x - config.wall_half_width - config.radius
        x1 = config.wall_x + config.wall_half_width + config.radius
        lower, upper = config.doorway_y
        # Split the wall into two permanent segments and the observable door.
        rectangles = (
            (x0, x1, -config.radius, lower + config.radius, np.ones(len(states), bool)),
            (x0, x1, upper - config.radius, 1 + config.radius, np.ones(len(states), bool)),
            (x0, x1, lower - config.radius, upper + config.radius, result[:, 4] < 0.5),
        )
        # Swept axis-separated collision checks avoid tunnelling even at large dt.
        for left, right, bottom, top, enabled in rectangles:
            within_y = enabled & (old_position[:, 1] > bottom) & (old_position[:, 1] < top)
            hit_left = within_y & (old_position[:, 0] <= left) & (position[:, 0] > left)
            hit_right = within_y & (old_position[:, 0] >= right) & (position[:, 0] < right)
            position[hit_left, 0] = left
            position[hit_right, 0] = right
            velocity[hit_left | hit_right, 0] = 0.0
        for left, right, bottom, top, enabled in rectangles:
            within_x = enabled & (position[:, 0] > left) & (position[:, 0] < right)
            hit_bottom = within_x & (old_position[:, 1] <= bottom) & (position[:, 1] > bottom)
            hit_top = within_x & (old_position[:, 1] >= top) & (position[:, 1] < top)
            position[hit_bottom, 1] = bottom
            position[hit_top, 1] = top
            velocity[hit_bottom | hit_top, 1] = 0.0
    result[:, :2] = position
    result[:, 2:4] = velocity
    return result, switch_hits, noise_hits


class PuckLab:
    """Continuing environment; ``step`` returns observation only, without reward."""

    state_dim = 6
    n_actions = 6
    action_names = ACTION_NAMES
    action_vectors = ACTION_VECTORS

    def __init__(self, config: PuckConfig | None = None, seed: int = 0):
        self.config = config or PuckConfig()
        self.observation_low, self.observation_high = _bounds(self.config)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        door_open = float(self.config.variant in {"open", "rooms"})
        self._state = np.array([*self.config.start, 0, 0, door_open, 0], dtype=float)
        self._steps = 0
        self._switch_interactions = 0
        self._noise_interactions = 0
        self._visited: set[tuple[int, int]] = set()
        self._record_cell()
        return self.observe()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    def step(self, action: int) -> np.ndarray:
        values, switches, noises = _transitions(
            self.config, self._state[None, :], np.asarray([action]), self.rng
        )
        self._state = values[0]
        self._steps += 1
        self._switch_interactions += int(switches[0])
        self._noise_interactions += int(noises[0])
        self._record_cell()
        return self.observe()

    def _record_cell(self) -> None:
        cell = np.minimum((self._state[:2] * self.config.coverage_bins).astype(int), self.config.coverage_bins - 1)
        self._visited.add((int(cell[0]), int(cell[1])))

    def metrics(self) -> dict[str, int | float | bool]:
        """Evaluator-only diagnostics. Coverage uses all square grid cells as denominator."""
        return {
            "steps": self._steps,
            "visited_cells": len(self._visited),
            "coverage": len(self._visited) / self.config.coverage_bins**2,
            "door_open": bool(self._state[4] >= 0.5),
            "switch_interactions": self._switch_interactions,
            "noise_interactions": self._noise_interactions,
        }

    def snapshot(self) -> dict[str, Any]:
        """Independent state, RNG, and evaluator state for exact replay/forking."""
        return {
            "config": self.config,
            "state": self.observe(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "steps": self._steps,
            "switch_interactions": self._switch_interactions,
            "noise_interactions": self._noise_interactions,
            "visited": self._visited.copy(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot["config"] != self.config:
            raise ValueError("snapshot configuration differs from environment")
        state = _states(np.asarray(snapshot["state"])[None, :])[0]
        if np.any(state < self.observation_low) or np.any(state > self.observation_high):
            raise ValueError("snapshot state lies outside declared observation bounds")
        self._state = state.copy()
        self.rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self._steps = int(snapshot["steps"])
        self._switch_interactions = int(snapshot["switch_interactions"])
        self._noise_interactions = int(snapshot["noise_interactions"])
        self._visited = set(snapshot["visited"])

    def clone(self) -> PuckLab:
        result = PuckLab(self.config)
        result.restore(self.snapshot())
        return result


class ExactPuckModel:
    """Diagnostic batched model; never calls or mutates an environment.

    This is exact for deterministic physical coordinates. On a triggered noise
    sample it predicts its conditional mean (zero), not a stochastic draw; this
    is not an exact predictive distribution for the ``noise`` variant.
    """

    state_dim = 6
    n_actions = 6

    def __init__(self, env: PuckLab | PuckConfig | None = None):
        self.config = env.config if isinstance(env, PuckLab) else (env or PuckConfig())
        self.observation_low, self.observation_high = _bounds(self.config)
        self.model_calls = 0  # Number of individual hypothetical transitions.

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        result, _, _ = _transitions(self.config, states, actions)
        self.model_calls += len(result)
        return result

    def rollout(self, initial: np.ndarray, plans: np.ndarray) -> np.ndarray:
        initial = np.asarray(initial, dtype=np.float64)
        plans = np.asarray(plans)
        if initial.shape != (6,) or plans.ndim != 2:
            raise ValueError("initial must have shape (6,) and plans shape (population,horizon)")
        _states(initial[None, :])
        _actions(plans.reshape(-1), plans.size)
        population, horizon = plans.shape
        states = np.broadcast_to(initial, (population, 6)).copy()
        result = np.empty((population, horizon, 6), dtype=np.float64)
        for t in range(horizon):
            states = self.predict(states, plans[:, t])
            result[:, t] = states
        return result
