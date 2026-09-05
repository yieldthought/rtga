"""Population search independent of the simulator, reward, and environment."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass
class PlannerConfig:
    population: int = 96
    horizon: int = 32
    generations: int = 6
    method: str = "persistent"
    elite_fraction: float = .10
    mutation_rate: float | None = None
    crossover: bool = True
    segment_mutation: float = .15
    immigrants: float = 0.0
    initial_repeat: int = 1
    seed: int = 0


@dataclass
class Evaluation:
    scores: np.ndarray
    paths: np.ndarray | None = None
    model_transitions: int = 0
    disagreement: np.ndarray | None = None


@dataclass
class Decision:
    action: int
    plan: np.ndarray
    score: float
    plans: np.ndarray
    scores: np.ndarray
    paths: np.ndarray | None
    selected_path: np.ndarray | None
    gene_ages: np.ndarray
    model_transitions: int
    candidate_evaluations: int
    latency_ms: float
    improvements: list[float]
    disagreement: float | None


class EvolutionPlanner:
    """Keep action genomes between decisions, never their stale scores/states.

    Each decision evaluates population * generations candidates. Reused elites
    are charged again. A call to plan advances the saved population by exactly
    one primitive step; the caller must execute the returned first action.
    """

    def __init__(self, n_actions: int, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        c = self.config
        if c.method not in {"persistent", "fresh", "random", "cem"}:
            raise ValueError(f"Unknown search method: {c.method}")
        if min(c.population, c.horizon, c.generations, n_actions) < 1:
            raise ValueError("Search sizes must be positive")
        if not 0 <= c.immigrants <= 1 or not 0 < c.elite_fraction <= 1:
            raise ValueError("Invalid population fraction")
        if c.mutation_rate is not None and not 0 <= c.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0,1] or None")
        if not 0 <= c.segment_mutation <= 1:
            raise ValueError("segment_mutation must be in [0,1]")
        if c.initial_repeat < 1:
            raise ValueError("initial_repeat must be positive")
        self.n_actions = n_actions
        self.rng = np.random.default_rng(c.seed)
        self.population = None
        self.ages = None
        self.last_evaluated_population = None

    def reset(self):
        self.population = self.ages = None

    def _random(self, count):
        c = self.config
        chunks = (c.horizon + c.initial_repeat - 1) // c.initial_repeat
        return np.repeat(self.rng.integers(self.n_actions, size=(count, chunks)),
                         c.initial_repeat, axis=1)[:, :c.horizon].copy()

    def _offspring(self, population, ages, scores):
        c = self.config
        n, h = population.shape
        elite_count = max(1, int(n * c.elite_fraction))
        ranking = np.argsort(scores)[::-1]
        tournaments = self.rng.integers(n, size=(2, n, 3))
        winners = np.take_along_axis(tournaments,
                                    scores[tournaments].argmax(axis=2)[..., None], axis=2)[..., 0]
        child = population[winners[0]].copy()
        child_age = ages[winners[0]].copy()
        if c.crossover and h > 1:
            cuts = self.rng.integers(1, h, size=n)
            right = np.arange(h)[None, :] >= cuts[:, None]
            child = np.where(right, population[winners[1]], child)
            child_age = np.where(right, ages[winners[1]], child_age)
        rate = 1 / h if c.mutation_rate is None else c.mutation_rate
        mutated = self.rng.random((n, h)) < rate
        point_values = self.rng.integers(self.n_actions, size=(n, h))
        child[mutated] = point_values[mutated]
        for row in np.flatnonzero(self.rng.random(n) < c.segment_mutation):
            start = int(self.rng.integers(h))
            length = int(self.rng.integers(1, min(8, h - start) + 1))
            mutated[row, start:start + length] = True
            child[row, start:start + length] = self.rng.integers(self.n_actions)
        child_age[mutated] = 0
        child[:elite_count] = population[ranking[:elite_count]]
        child_age[:elite_count] = ages[ranking[:elite_count]]
        immigrants = min(n - elite_count, int(n * c.immigrants))
        if immigrants:
            child[-immigrants:] = self._random(immigrants)
            child_age[-immigrants:] = 0
        return child, child_age

    def plan(self, evaluate: Callable[[np.ndarray], Evaluation]) -> Decision:
        start = perf_counter()
        c = self.config
        if self.population is None or c.method in {"fresh", "random", "cem"}:
            population = self._random(c.population)
            ages = np.zeros_like(population)
        else:
            population, ages = self.population.copy(), self.ages.copy()
        probs = np.full((c.horizon, self.n_actions), 1 / self.n_actions)
        best_score, best_plan, best_path, best_disagreement = -np.inf, None, None, None
        best_ages = None
        model_transitions, improvements = 0, []
        for generation in range(c.generations):
            result = evaluate(population)
            scores = np.asarray(result.scores, dtype=float)
            if scores.shape != (c.population,) or not np.all(np.isfinite(scores)):
                raise ValueError("Evaluator must return one finite score per plan")
            model_transitions += result.model_transitions
            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score = float(scores[idx])
                best_plan = population[idx].copy()
                best_ages = ages[idx].copy()
                best_path = None if result.paths is None else result.paths[idx].copy()
                best_disagreement = (None if result.disagreement is None
                                     else float(result.disagreement[idx]))
            improvements.append(best_score)
            if generation + 1 == c.generations:
                break
            if c.method == "random":
                population, ages = self._random(c.population), np.zeros_like(ages)
            elif c.method == "cem":
                elites = population[np.argsort(scores)[-max(1, int(c.population * c.elite_fraction)):]]
                counts = (elites[..., None] == np.arange(self.n_actions)).mean(axis=0)
                probs = .1 / self.n_actions + .9 * counts
                u = self.rng.random((c.population, c.horizon, 1))
                population = (u > probs.cumsum(axis=1)[None]).sum(axis=2).clip(max=self.n_actions - 1)
                population[0] = best_plan
                ages = np.zeros_like(population)
            else:
                population, ages = self._offspring(population, ages, scores)
        # The returned best may come from an earlier generation (including a
        # tied elite). Archive it explicitly in the final living population.
        # This reuses its evaluation from the SAME decision/model snapshot.
        matches = np.flatnonzero(np.all(population == best_plan, axis=1))
        selected = int(matches[0]) if len(matches) else int(np.argmin(scores))
        population[selected] = best_plan
        ages[selected] = best_ages
        scores[selected] = best_score
        if result.paths is not None:
            result.paths[selected] = best_path
        order = np.concatenate([[selected], np.argsort(scores)[::-1]])
        order = np.concatenate([order[:1], order[1:][order[1:] != selected]])
        self.last_evaluated_population = population.copy()
        # Save only genomes. Paths/scores belong to this observation and model.
        self.population = np.concatenate([population[:, 1:], self.rng.integers(
            self.n_actions, size=(c.population, 1))], axis=1)
        self.ages = np.concatenate([ages[:, 1:] + 1, np.zeros((c.population, 1), dtype=int)], axis=1)
        return Decision(int(best_plan[0]), best_plan, best_score,
                        population[order].copy(), scores[order].copy(),
                        None if result.paths is None else result.paths[order].copy(),
                        best_path, ages[order].copy(), model_transitions,
                        c.population * c.generations, (perf_counter() - start) * 1000,
                        improvements, best_disagreement)
