# Real-time genetic algorithms

A research project exploring persistent populations of future action plans, online world models, and curiosity driven by disagreement between those models.

At each real step: evolve possible futures, execute the best plan's first action, shift the population, append proposals, and evaluate again from the new observation.

The proposed starting point is a small visible 2D environment with three learned models and primitive actions. The key experiment is whether planning for useful uncertainty produces knowledge that supports previously unseen goals.

Read the [research and experiment plan](RESEARCH_PLAN.md) for primary literature, architecture, comparison experiments, visual diagnostics, and the progression toward learned action hierarchies and pixel environments.

The implementation now includes a vectorized puck environment, persistent evolutionary search, categorical CEM and random-search comparisons, bootstrapped online dynamics ensembles, and a standalone trace viewer.

Read the [research journal](PROGRESS.md) for the current experiments and mini-papers. Curiosity-only Mario completion remains the target; it has not been demonstrated here.

## Run the first experiment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m rtga.cli oracle --output runs/oracle --seeds 10
```

The output directory contains complete per-seed measurements and self-contained HTML viewers of the first seed's actual planning traces. This initial diagnostic supplies an exact simulator and a goal objective; it is not the curiosity-only experiment.

## Observation/action integration

`RTGAAgent` learns online from flat observations and can save/resume its full decision state. `ObservationActionAdapter` isolates Gymnasium rewards and diagnostics from the agent. The optional [Mario adapter](docs/MARIO.md) accepts an explicit local ROM and delivers RGB frames; a learned visual representation remains required before the vector agent can control it.
