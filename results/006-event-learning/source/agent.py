"""Observation/action agent: online learning and decision-time population search.

This module imports no environment and has no access to simulator state, rewards,
coverage, or completion metrics. Goals are an explicit diagnostic-only option.
"""

import copy
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np
import torch

from .models import DynamicsEnsemble, EnsembleConfig, ReplayBuffer
from .planning import EvolutionPlanner, PlannerConfig, Evaluation


@dataclass
class AgentConfig:
    warmup: int = 256
    initial_fit: int = 200
    train_every: int = 8
    train_steps: int = 8
    batch_size: int = 128
    replay_capacity: int = 100_000
    members: int = 3
    hidden: int = 64
    seed: int = 0
    mode: str = 'curiosity'
    warmup_repeat: int = 1
    discount: float = .98
    risk_penalty: float = 0.0
    planner: PlannerConfig = field(default_factory=lambda: PlannerConfig(
        population=48, horizon=12, generations=3))


class LearnedEvaluator:
    """Coherent member trajectories with same-input local disagreement.

    Every candidate has K carrier trajectories. At each carrier state all K
    members predict the same action, for K² member transitions per plan step.
    Means (thresholded for binary coordinates) advance each carrier. This first
    estimator integrates epistemic alternatives, not stochastic particles.
    """

    def __init__(self, model, observation, mode='curiosity', goal=None,
                 discount=.98, risk_penalty=0.0):
        self.model, self.mode, self.discount = model, mode, discount
        self.observation = torch.as_tensor(observation, dtype=torch.float32, device=model.device)
        self.goal = None if goal is None else torch.as_tensor(goal, dtype=torch.float32, device=model.device)
        if mode not in {'curiosity', 'goal'} or (mode == 'goal' and goal is None):
            raise ValueError('Goal mode requires an explicit goal; otherwise use curiosity')
        self.risk_penalty = risk_penalty
        self.version = model.version

    @torch.no_grad()
    def __call__(self, plans):
        if self.model.version != self.version:
            raise RuntimeError('World model changed during one planning decision')
        model, device = self.model, self.model.device
        k, n, h = model.config.members, *plans.shape
        actions = torch.as_tensor(plans, dtype=torch.long, device=device)
        states = self.observation.expand(k * n, -1).clone()
        carrier = torch.arange(k, device=device).repeat_interleave(n)
        index = torch.arange(k * n, device=device)
        returns = torch.zeros(k * n, device=device)
        d_total = torch.zeros_like(returns)
        paths = []
        scale = torch.as_tensor(model.config.observation_scale or
                                (1.,) * model.config.state_dim, device=device)
        for t in range(h):
            mean, _variance = model.predict_tensor(states, actions[:, t].repeat(k))
            local_d = (mean / scale).var(dim=0, unbiased=False).mean(dim=-1)
            next_states = mean[carrier, index].clone()
            if model.config.binary_dims:
                dims = list(model.config.binary_dims)
                next_states[:, dims] = (next_states[:, dims] >= .5).float()
            weight = self.discount ** t
            d_total += weight * local_d
            if self.mode == 'curiosity':
                returns += weight * local_d
            else:
                returns -= weight * torch.linalg.vector_norm(next_states[:, :len(self.goal)] - self.goal, dim=-1)
            states = next_states
            paths.append(states.reshape(k, n, -1).mean(dim=0))
        returns = returns.reshape(k, n)
        scores = returns.mean(dim=0) - self.risk_penalty * returns.std(dim=0, unbiased=False)
        if self.model.version != self.version:
            raise RuntimeError('World model changed during evaluation')
        return Evaluation(scores.cpu().numpy(), torch.stack(paths, dim=1).cpu().numpy(),
                          model_transitions=k * k * n * h,
                          disagreement=d_total.reshape(k, n).mean(dim=0).cpu().numpy())


class RTGAAgent:
    def __init__(self, observation_low, observation_high, n_actions,
                 config=None, binary_dims=(), goal=None):
        self.config = config or AgentConfig()
        c = self.config
        if c.mode not in {'curiosity', 'goal', 'random', 'reactive'}:
            raise ValueError('Unknown agent mode')
        if c.train_every < 1 or c.warmup < 1 or c.warmup_repeat < 1:
            raise ValueError('Training periods, warmup, and repeats must be positive')
        self.n_actions, self.goal = n_actions, goal
        low, high = np.asarray(observation_low), np.asarray(observation_high)
        if low.ndim != 1 or high.shape != low.shape:
            raise ValueError('RTGAAgent currently needs flat vectors; image observations need an encoder')
        if c.mode == 'goal' and goal is None:
            raise ValueError('Goal mode requires an explicit diagnostic goal')
        magnitude = np.maximum(np.abs(low), np.abs(high))
        # Unbounded numeric Gymnasium dimensions start in their native units.
        # This is an explicit scale fallback, not an invented observation bound.
        scale = np.maximum(np.where(np.isfinite(magnitude), magnitude, 1.), 1e-3)
        self.model = DynamicsEnsemble(EnsembleConfig(
            len(low), n_actions, members=c.members, hidden=c.hidden, seed=c.seed,
            observation_scale=tuple(scale), observation_low=tuple(low),
            observation_high=tuple(high), binary_dims=tuple(binary_dims)))
        self.replay = ReplayBuffer(c.replay_capacity, len(low), seed=c.seed + 101)
        self.planner = EvolutionPlanner(n_actions, c.planner)
        self.rng = np.random.default_rng(c.seed + 201)
        self.previous_state = self.previous_action = None
        self.transitions = 0
        self.last_decision = None
        self.last_info = {}
        self._warmup_action = 0

    def state_dict(self):
        """Snapshot decision state; the environment must be saved separately.

        Diagnostic timings and the last rendered decision are not resumed.
        Model parameters, optimizer, replay, genomes and all private RNGs are.
        """
        return copy.deepcopy({
            'format_version': 1, 'config': asdict(self.config),
            'n_actions': self.n_actions,
            'goal': None if self.goal is None else np.asarray(self.goal).tolist(),
            'model': self.model.state_dict(), 'replay': self.replay.state_dict(),
            'planner': {'population': self.planner.population, 'ages': self.planner.ages,
                        'rng': self.planner.rng.bit_generator.state},
            'rng': self.rng.bit_generator.state,
            'previous_state': self.previous_state, 'previous_action': self.previous_action,
            'transitions': self.transitions, 'warmup_action': self._warmup_action,
        })

    def load_state_dict(self, state):
        if (state.get('format_version') != 1 or state['config'] != asdict(self.config)
                or state['n_actions'] != self.n_actions
                or state['goal'] != (None if self.goal is None else np.asarray(self.goal).tolist())):
            raise ValueError('Agent checkpoint configuration does not match')
        state = copy.deepcopy(state)
        self.model.load_state_dict(state['model'])
        self.replay.load_state_dict(state['replay'])
        self.planner.population = state['planner']['population']
        self.planner.ages = state['planner']['ages']
        self.planner.rng.bit_generator.state = state['planner']['rng']
        self.planner.last_evaluated_population = None
        self.rng.bit_generator.state = state['rng']
        self.previous_state, self.previous_action = state['previous_state'], state['previous_action']
        self.transitions, self._warmup_action = state['transitions'], state['warmup_action']
        self.last_decision, self.last_info = None, {}

    def save(self, path):
        # Encode arrays as plain tensor records so loading needs no pickle
        # globals beyond PyTorch's restricted weights-only loader.
        def encode(value):
            if isinstance(value, np.ndarray):
                return {'__rtga_array__': torch.from_numpy(value.copy())}
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(encode(item) for item in value)
            return value
        torch.save(encode(self.state_dict()), path)

    @classmethod
    def load(cls, path):
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'__rtga_array__'}:
                    return value['__rtga_array__'].numpy().copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(decode(item) for item in value)
            return value
        state = decode(torch.load(path, map_location='cpu', weights_only=True))
        config = dict(state['config'])
        config['planner'] = PlannerConfig(**config['planner'])
        model_config = state['model']['config']
        agent = cls(model_config['observation_low'], model_config['observation_high'],
                    state['n_actions'], AgentConfig(**config),
                    binary_dims=model_config['binary_dims'], goal=state['goal'])
        agent.load_state_dict(state)
        return agent

    def observe_final(self, observation):
        """Record final observation before resetting; do not invent a reset transition."""
        if self.previous_state is not None:
            self.replay.append(self.previous_state, self.previous_action, observation)
            self.transitions += 1
        self.previous_state = self.previous_action = None

    def act(self, observation, is_first=False):
        start = perf_counter()
        c = self.config
        observation = np.asarray(observation, dtype=np.float32)
        if is_first:
            self.previous_state = self.previous_action = None
            self.planner.reset()
        prediction_error = disagreement = None
        if self.previous_state is not None:
            if self.model.version:
                mu, _ = self.model.predict(self.previous_state[None], np.array([self.previous_action]))
                prediction_error = float(np.mean((mu.mean(axis=0)[0] - observation) ** 2))
                disagreement = float(mu.var(axis=0).mean())
            self.replay.append(self.previous_state, self.previous_action, observation)
            self.transitions += 1
        train_start = perf_counter()
        training = {'steps': 0}
        if self.transitions == c.warmup:
            training = self.model.train_steps(self.replay, c.initial_fit, c.batch_size)
        elif self.transitions > c.warmup and (self.transitions - c.warmup) % c.train_every == 0:
            training = self.model.train_steps(self.replay, c.train_steps, c.batch_size)
        train_ms = (perf_counter() - train_start) * 1000
        self.last_decision = None
        if self.transitions < c.warmup or c.mode == 'random':
            if self.transitions % c.warmup_repeat == 0:
                self._warmup_action = int(self.rng.integers(self.n_actions))
            action = self._warmup_action
        elif c.mode == 'reactive':
            states = np.repeat(observation[None], self.n_actions, axis=0)
            mu, _ = self.model.predict(states, np.arange(self.n_actions))
            scale = np.array(self.model.config.observation_scale)
            score = (mu / scale).var(axis=0).mean(axis=-1)
            action = int(np.argmax(score))
        else:
            evaluator = LearnedEvaluator(self.model, observation,
                                         mode='goal' if c.mode == 'goal' else 'curiosity',
                                         goal=self.goal, discount=c.discount,
                                         risk_penalty=c.risk_penalty)
            self.last_decision = self.planner.plan(evaluator)
            action = self.last_decision.action
        self.previous_state, self.previous_action = observation.copy(), action
        self.last_info = {'prediction_error_before_update': prediction_error,
                          'disagreement_before_update': disagreement,
                          'model_version': self.model.version, 'train_ms': train_ms,
                          'training_steps': training['steps'], 'transition_count': self.transitions,
                          'action_latency_ms': (perf_counter() - start) * 1000}
        return action
