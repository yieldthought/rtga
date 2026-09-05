import numpy as np

from rtga.planning import EvolutionPlanner, PlannerConfig, Evaluation


def test_shift_keeps_genomes_and_reevaluates_from_new_root():
    planner = EvolutionPlanner(4, PlannerConfig(population=12, horizon=8, generations=2, seed=4))
    calls = []
    def score(plans):
        calls.append(plans.copy())
        return Evaluation(-np.sum(plans, axis=1), model_transitions=plans.size)
    first = planner.plan(score)
    np.testing.assert_array_equal(planner.population[:, :-1], calls[-1][:, 1:])
    shifted = planner.population.copy()
    calls.clear()
    second = planner.plan(score)
    np.testing.assert_array_equal(calls[0], shifted)
    assert first.action == first.plan[0]
    assert second.model_transitions == 12 * 8 * 2
    assert second.candidate_evaluations == 24


def test_methods_have_equal_charged_budget_and_improve_simple_objective():
    for method in ['persistent', 'fresh', 'random', 'cem']:
        planner = EvolutionPlanner(3, PlannerConfig(population=32, horizon=8, generations=5, method=method))
        out = planner.plan(lambda p: Evaluation(np.sum(p == 2, axis=1), model_transitions=p.size))
        assert out.model_transitions == 32 * 8 * 5
        assert out.candidate_evaluations == 160
        assert out.score >= out.improvements[0]
        assert np.all(out.plan < 3)


def test_reproducible_without_global_rng():
    def run():
        p = EvolutionPlanner(3, PlannerConfig(population=24, horizon=12, generations=4, seed=7))
        return [p.plan(lambda x: Evaluation(x.sum(axis=1))).plan for _ in range(3)]
    np.testing.assert_array_equal(run(), run())
