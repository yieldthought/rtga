"""Learn dynamics from actual transitions, without access to an environment.

The ensemble dimension is batched throughout inference and fitting. Each member
has its own parameters and receives an independently bootstrapped replay batch.
Continuous heads predict Gaussian state changes; binary heads predict next-state
Bernoulli probabilities. Predictions are in the original observation units.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EnsembleConfig:
    state_dim: int
    n_actions: int
    members: int = 3
    hidden: int = 64
    learning_rate: float = 0.002
    seed: int = 0
    binary_dims: tuple[int, ...] = ()
    observation_scale: tuple[float, ...] | None = None
    observation_low: tuple[float, ...] | None = None
    observation_high: tuple[float, ...] | None = None
    min_logvar: float = -8.0
    max_logvar: float = 1.0
    binary_loss_weight: float = 10.0

    def __post_init__(self) -> None:
        if min(self.state_dim, self.n_actions, self.members, self.hidden) < 1:
            raise ValueError("Model dimensions and member count must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not self.min_logvar < self.max_logvar:
            raise ValueError("min_logvar must be less than max_logvar")
        if not math.isfinite(self.binary_loss_weight) or self.binary_loss_weight <= 0:
            raise ValueError("binary_loss_weight must be positive and finite")
        binary_dims = tuple(int(dim) for dim in self.binary_dims)
        if len(set(binary_dims)) != len(binary_dims) or any(
            dim < 0 or dim >= self.state_dim for dim in binary_dims
        ):
            raise ValueError("binary_dims must contain distinct valid dimensions")
        object.__setattr__(self, "binary_dims", binary_dims)
        for name in ("observation_scale", "observation_low", "observation_high"):
            values = getattr(self, name)
            if values is None:
                continue
            values = tuple(float(value) for value in values)
            if len(values) != self.state_dim or np.isnan(values).any():
                raise ValueError(f"{name} must have state_dim values without NaNs")
            if name == "observation_scale" and (not np.isfinite(values).all() or min(values) <= 0):
                raise ValueError("observation_scale must be finite and strictly positive")
            object.__setattr__(self, name, values)
        if self.observation_low is not None and self.observation_high is not None:
            if np.any(np.asarray(self.observation_low) > self.observation_high):
                raise ValueError("Observation lower bounds exceed upper bounds")


class ReplayBuffer:
    """A bounded collection of observed transitions with its own sampling RNG."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 0) -> None:
        if capacity < 1 or state_dim < 1:
            raise ValueError("capacity and state_dim must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def append(
        self, state: Sequence[float], action: int, next_state: Sequence[float]
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("A transition must contain two flat state vectors")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError("Replay states must be finite")
        if not isinstance(action, (int, np.integer)) or action < 0:
            raise ValueError("action must be a nonnegative integer")
        self.states[self._cursor] = state
        self.actions[self._cursor] = action
        self.next_states[self._cursor] = next_state
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, members: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample with replacement independently for every ensemble member.

        Shapes are ``[members, batch_size, state_dim]``,
        ``[members, batch_size]``, and ``[members, batch_size, state_dim]``.
        Returned arrays are copies; fitting cannot modify the stored experience.
        """
        if not self._size:
            raise ValueError("Cannot sample an empty replay buffer")
        if batch_size < 1 or members < 1:
            raise ValueError("batch_size and members must be positive")
        indices = self.rng.integers(self._size, size=(members, batch_size))
        return self.states[indices], self.actions[indices], self.next_states[indices]

    def transitions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of all retained transitions, oldest first."""
        start = self._cursor if self._size == self.capacity else 0
        indices = (start + np.arange(self._size)) % self.capacity
        return self.states[indices], self.actions[indices], self.next_states[indices]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "states": self.states[: self._size].copy(),
            "actions": self.actions[: self._size].copy(),
            "next_states": self.next_states[: self._size].copy(),
            "cursor": self._cursor,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity or state["state_dim"] != self.state_dim:
            raise ValueError("Replay checkpoint dimensions do not match")
        size = len(state["actions"])
        if not 0 <= size <= self.capacity:
            raise ValueError("Invalid replay checkpoint size")
        self.states[:size] = state["states"]
        self.actions[:size] = state["actions"]
        self.next_states[:size] = state["next_states"]
        self._size = size
        self._cursor = state["cursor"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class _EnsembleLinear(nn.Module):
    def __init__(self, members: int, inputs: int, outputs: int, generator: torch.Generator):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(members, 1, outputs))
        bound = math.sqrt(6.0 / (inputs + outputs))
        with torch.no_grad():
            self.weight.uniform_(-bound, bound, generator=generator)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.bmm(inputs, self.weight) + self.bias


class _EnsembleNetwork(nn.Module):
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        # A private generator avoids changing the runner's/global torch RNG.
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.input = _EnsembleLinear(
            config.members, config.state_dim + config.n_actions, config.hidden, generator
        )
        self.hidden = _EnsembleLinear(
            config.members, config.hidden, config.hidden, generator
        )
        self.output = _EnsembleLinear(
            config.members, config.hidden, config.state_dim * 2, generator
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(F.silu(self.hidden(F.silu(self.input(inputs)))))


class DynamicsEnsemble:
    """Small independently parameterized probabilistic forward models.

    ``predict_tensor`` is the planner API: shared inputs ``[B,D]`` produce
    per-member next-state means and variances ``[K,B,D]`` without gradients.
    Thread counts are deliberately left to the application.
    """

    def __init__(self, config: EnsembleConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.network = _EnsembleNetwork(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.version = 0
        self._scale = torch.tensor(
            config.observation_scale or (1.0,) * config.state_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._binary = torch.zeros(config.state_dim, dtype=torch.bool, device=self.device)
        self._binary[list(config.binary_dims)] = True
        self._low = self._optional_tensor(config.observation_low)
        self._high = self._optional_tensor(config.observation_high)

    def _optional_tensor(self, values: tuple[float, ...] | None) -> Tensor | None:
        return None if values is None else torch.tensor(values, device=self.device)

    def _inputs(self, states, actions) -> tuple[Tensor, Tensor]:
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if states.ndim != 2 or states.shape[-1] != self.config.state_dim:
            raise ValueError("states must have shape [batch_size, state_dim]")
        if actions.shape != states.shape[:-1]:
            raise ValueError("actions must have shape [batch_size]")
        return states, actions

    def _forward(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        # Training uses independently sampled [K,B,D]; inference expands [B,D].
        if states.ndim == 2:
            states = states.unsqueeze(0).expand(self.config.members, -1, -1)
            actions = actions.unsqueeze(0).expand(self.config.members, -1)
        one_hot = F.one_hot(actions, self.config.n_actions).to(states.dtype)
        output = self.network(torch.cat((states / self._scale, one_hot), dim=-1))
        mean, raw_logvar = output.chunk(2, dim=-1)
        logvar = self.config.min_logvar + (
            self.config.max_logvar - self.config.min_logvar
        ) * torch.sigmoid(raw_logvar)
        return mean, logvar

    @torch.no_grad()
    def predict_tensor(self, states, actions) -> tuple[Tensor, Tensor]:
        states, actions = self._inputs(states, actions)
        normalized_delta, logvar = self._forward(states, actions)
        means = states.unsqueeze(0) + normalized_delta * self._scale
        variances = logvar.exp() * self._scale.square()
        if self.config.binary_dims:
            probabilities = torch.sigmoid(normalized_delta[..., self._binary])
            means[..., self._binary] = probabilities
            variances[..., self._binary] = probabilities * (1.0 - probabilities)
        if self._low is not None:
            means = torch.maximum(means, self._low)
        if self._high is not None:
            means = torch.minimum(means, self._high)
        return means, variances

    def predict(self, states, actions) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_tensor(states, actions)
        return means.cpu().numpy(), variances.cpu().numpy()

    def train_steps(
        self, replay: ReplayBuffer, steps: int, batch_size: int = 128
    ) -> dict[str, float | int]:
        if steps < 0 or batch_size < 1:
            raise ValueError("steps must be nonnegative and batch_size positive")
        if replay.state_dim != self.config.state_dim:
            raise ValueError("Replay state dimension does not match the model")
        if not len(replay) or steps == 0:
            return {"loss": 0.0, "steps": 0, "samples": 0, "version": self.version}
        losses = []
        self.network.train()
        for _ in range(steps):
            state_batch, action_batch, next_batch = replay.sample(batch_size, self.config.members)
            states = torch.as_tensor(state_batch, device=self.device)
            actions = torch.as_tensor(action_batch, device=self.device)
            next_states = torch.as_tensor(next_batch, device=self.device)
            mean, logvar = self._forward(states, actions)
            target = (next_states - states) / self._scale
            element_loss = 0.5 * ((mean - target).square() * (-logvar).exp() + logvar)
            if self.config.binary_dims:
                binary_targets = next_states[..., self._binary]
                if torch.any((binary_targets < 0) | (binary_targets > 1)):
                    raise ValueError("Binary target dimensions must be in [0, 1]")
                # Deterministic continuous dimensions can acquire very large
                # inverse-variance gradients. Explicit weighting keeps discrete
                # events learnable in the shared trunk; it does not change the
                # predictive likelihood reported by evaluate().
                element_loss[..., self._binary] = self.config.binary_loss_weight * F.binary_cross_entropy_with_logits(
                    mean[..., self._binary], binary_targets, reduction="none"
                )
            member_loss = element_loss.mean(dim=(1, 2))
            self.optimizer.zero_grad(set_to_none=True)
            member_loss.sum().backward()
            self._clip_member_gradients(10.0)
            self.optimizer.step()
            self.version += 1
            losses.append(member_loss.detach().mean())
        self.network.eval()
        return {
            "loss": float(torch.stack(losses).mean().cpu()),
            "steps": steps,
            "samples": steps * batch_size * self.config.members,
            "version": self.version,
        }

    def _clip_member_gradients(self, max_norm: float) -> None:
        # Clip each member separately, preserving independent optimization.
        with torch.no_grad():
            params = [param for param in self.network.parameters() if param.grad is not None]
            squared_norm = sum(
                param.grad.reshape(self.config.members, -1).square().sum(dim=1)
                for param in params
            )
            factors = (max_norm / (squared_norm.sqrt() + 1e-8)).clamp(max=1.0)
            for param in params:
                param.grad.mul_(factors.reshape(-1, *([1] * (param.ndim - 1))))

    @torch.no_grad()
    def evaluate(self, states, actions, next_states) -> dict[str, float]:
        """Score supplied real held-out transitions; never fit or store them.

        ``nll`` is the mean negative log density of the equally weighted member
        mixture, divided by state dimension. Discrete dimensions use Bernoulli
        log probabilities; continuous dimensions use observation-unit Gaussians.
        """
        states, actions = self._inputs(states, actions)
        targets = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        if targets.shape != states.shape or len(states) == 0:
            raise ValueError("Held-out targets must match a nonempty state batch")
        means, variances = self.predict_tensor(states, actions)
        ensemble_mean = means.mean(dim=0)
        member_nll = 0.5 * (
            math.log(2.0 * math.pi)
            + variances.clamp_min(1e-12).log()
            + (targets.unsqueeze(0) - means).square() / variances.clamp_min(1e-12)
        )
        if self.config.binary_dims:
            probabilities = means[..., self._binary].clamp(1e-7, 1.0 - 1e-7)
            binary_targets = targets[..., self._binary].unsqueeze(0).expand_as(probabilities)
            member_nll[..., self._binary] = F.binary_cross_entropy(
                probabilities, binary_targets, reduction="none"
            )
        mixture_nll = -torch.logsumexp(-member_nll.sum(dim=-1), dim=0) + math.log(
            self.config.members
        )
        result = {
            "mse": float((ensemble_mean - targets).square().mean().cpu()),
            "normalized_mse": float(((ensemble_mean - targets) / self._scale).square().mean().cpu()),
            "nll": float((mixture_nll / self.config.state_dim).mean().cpu()),
            "epistemic": float((means.var(dim=0, unbiased=False) / self._scale.square()).mean().cpu()),
            "aleatoric": float((variances.mean(dim=0) / self._scale.square()).mean().cpu()),
        }
        if self.config.binary_dims:
            result["binary_brier"] = float(
                (ensemble_mean[..., self._binary] - targets[..., self._binary]).square().mean().cpu()
            )
        return result

    def state_dict(self) -> dict:
        """Return an independent snapshot, including the fitting state."""
        return copy.deepcopy({
            "format_version": 1,
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "version": self.version,
        })

    def load_state_dict(self, state: dict) -> None:
        if state.get("format_version") != 1 or EnsembleConfig(**state["config"]) != self.config:
            raise ValueError("Model checkpoint configuration does not match")
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.version = int(state["version"])
        self.network.eval()

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> DynamicsEnsemble:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(EnsembleConfig(**checkpoint["config"]), device=device)
        model.load_state_dict(checkpoint)
        return model
