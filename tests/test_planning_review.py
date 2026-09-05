"""Independent integration checks for the planner/simulator boundary."""

import numpy as np
import pytest

from rtga.envs import ExactPuckModel, PuckConfig, PuckLab
from rtga.planning import Evaluation, EvolutionPlanner, PlannerConfig


@pytest.mark.parametrize("method", ["persistent", "fresh", "random", "cem"])
def test_selected_oracle_path_matches_executed_plan_and_transition_budget(method):
    env = PuckLab(PuckConfig(variant="rooms"), seed=2)
    config = PlannerConfig(population=16, horizon=12, generations=3, method=method, seed=4)
    planner = EvolutionPlanner(env.n_actions, config)
    model = ExactPuckModel(env)
    target = np.array([0.8, 0.5])

    for _ in range(3):
        root = env.observe()
        before = model.model_calls

        def evaluate(plans):
            paths = model.rollout(root, plans)
            scores = -np.linalg.norm(paths[..., :2] - target, axis=-1).sum(axis=1)
            return Evaluation(scores, paths=paths, model_transitions=plans.size)

        decision = planner.plan(evaluate)
        # The reported selected trajectory must belong to the action sequence
        # that will actually be used, even if it came from an earlier generation.
        selected_world = env.clone()
        actual_selected = np.array([selected_world.step(int(a)) for a in decision.plan])
        np.testing.assert_array_equal(decision.selected_path, actual_selected)
        assert decision.model_transitions == model.model_calls - before == 16 * 12 * 3
        assert decision.candidate_evaluations == 16 * 3
        np.testing.assert_array_equal(env.observe(), root)
        np.testing.assert_array_equal(env.step(decision.action), decision.selected_path[0])


def test_elapsed_gene_age_follows_shift_and_not_number_of_generations():
    config = PlannerConfig(population=8, horizon=6, generations=4,
                           elite_fraction=1.0, mutation_rate=0.0,
                           segment_mutation=0.0, crossover=False, seed=2)
    planner = EvolutionPlanner(6, config)

    def evaluate(plans):
        return Evaluation(plans.sum(axis=1))

    first = planner.plan(evaluate)
    np.testing.assert_array_equal(first.gene_ages, np.zeros((8, 6), dtype=int))
    second = planner.plan(evaluate)
    np.testing.assert_array_equal(second.gene_ages[:, :-1], np.ones((8, 5), dtype=int))
    np.testing.assert_array_equal(second.gene_ages[:, -1], np.zeros(8, dtype=int))


@pytest.mark.parametrize("method", ["persistent", "fresh", "cem"])
def test_actual_elite_scores_do_not_regress_for_deterministic_objective(method):
    planner = EvolutionPlanner(6, PlannerConfig(population=24, horizon=12,
                               generations=6, method=method, seed=5))
    generation_best = []

    def evaluate(plans):
        scores = (plans == 2).sum(axis=1).astype(float)
        generation_best.append(scores.max())
        return Evaluation(scores)

    planner.plan(evaluate)
    assert np.all(np.diff(generation_best) >= 0)


@pytest.mark.parametrize("method", ["persistent", "fresh", "random", "cem"])
def test_tied_historical_winner_is_archived_with_its_path_and_gene_ages(method):
    planner = EvolutionPlanner(6, PlannerConfig(population=32, horizon=16,
                               generations=4, method=method, seed=0))
    # Make age preservation observable for the persistent case rather than
    # merely checking that a newly generated population has zero ages.
    planner.population = np.random.default_rng(20).integers(0, 6, size=(32, 16))
    planner.ages = np.broadcast_to(np.arange(16) + 11, (32, 16)).copy()
    first_plan, first_path = None, None
    expected_ages = planner.ages[0].copy() if method == "persistent" else np.zeros(16, int)

    def evaluate(plans):
        nonlocal first_plan, first_path
        paths = np.cumsum(plans, axis=1)[..., None].astype(float)
        if first_plan is None:
            first_plan, first_path = plans[0].copy(), paths[0].copy()
        return Evaluation(np.zeros(len(plans)), paths=paths, model_transitions=plans.size)

    decision = planner.plan(evaluate)
    np.testing.assert_array_equal(decision.plan, first_plan)
    np.testing.assert_array_equal(decision.plans[0], first_plan)
    np.testing.assert_array_equal(decision.selected_path, first_path)
    np.testing.assert_array_equal(decision.paths[0], first_path)
    np.testing.assert_array_equal(decision.gene_ages[0], expected_ages)
    assert decision.action == first_plan[0]
    assert decision.scores[0] == decision.score == 0
    archived = np.flatnonzero(np.all(planner.last_evaluated_population == first_plan, axis=1))
    assert len(archived) >= 1
    np.testing.assert_array_equal(planner.population[archived[0], :-1], first_plan[1:])
    np.testing.assert_array_equal(planner.ages[archived[0], :-1], expected_ages[1:] + 1)
    assert planner.ages[archived[0], -1] == 0
    assert decision.model_transitions == 32 * 16 * 4


@pytest.mark.parametrize("field", ["mutation_rate", "segment_mutation"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_mutation_probabilities_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        EvolutionPlanner(6, PlannerConfig(**{field: value}))
