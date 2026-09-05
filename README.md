# Real-time genetic algorithms

A research project exploring persistent populations of future action plans, online world models, and curiosity driven by disagreement between those models.

At each real step: evolve possible futures, execute the best plan's first action, shift the population, append proposals, and evaluate again from the new observation.

The proposed starting point is a small visible 2D environment with three learned models and primitive actions. The key experiment is whether planning for useful uncertainty produces knowledge that supports previously unseen goals.

Read the [research and experiment plan](RESEARCH_PLAN.md) for primary literature, architecture, comparison experiments, visual diagnostics, and the progression toward learned action hierarchies and pixel environments.

Status: research and planning complete; implementation has not started. No benchmark results are claimed.
