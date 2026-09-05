# RTGA research journal

The target is an agent that learns from observations and actions, explores through intrinsic motivation, and gains reusable control. Curiosity-only Mario completion is the stretch outcome. It has not been demonstrated here.

## Current work · 5 September 2026

Building the first experimental system: a batched puck simulator, persistent evolutionary planner, three online dynamics models, and a viewer of actual planning traces.

| Checkpoint | Status | Required evidence |
|---|---|---|
| Research plan and repository | In progress | Sources, experiment gates, reproducible package |
| Exact-model planner | In progress | Paired-seed navigation comparisons and verified traces |
| Online learned models | In progress | Predictive improvement and actual control |
| Planned curiosity | Pending | Discovery beyond one-step reach and withheld-goal competence |
| Noise and surprise | Pending | Useful uncertainty and measured recovery |
| Action memory / hierarchy | Pending | Benefit at equal compute |
| Platformer / Mario | Pending | Curiosity-only learning with no score or progress leakage |

## Papers

Papers will appear here once measurements exist. Each will include the question, method, complete results, limitations, and exact reproduction commands. Development results and confirmatory comparisons will be labelled separately.

## Research rules

- Preserve negative results and record why a direction changes.
- Keep environment reward and evaluator information out of curiosity agents.
- Count experience from the first action, including warmup.
- Re-evaluate shifted plans from the actual new observation.
- Commit runnable checkpoints and their supporting results.
- Use independent reviews for consequential claims and confusing failures.

