# RTGA research journal

The target is an agent that learns from observations and actions, explores through intrinsic motivation, and gains reusable control. Curiosity-only Mario completion is the stretch outcome. It has not been demonstrated here.

## Current work · 5 September 2026

The first system is running: batched puck simulation, persistent evolutionary search, three online dynamics models, and actual planning traces. The exact-model comparison and a model-calibration study are complete. Initial curiosity runs expose failures in learning and using the switch mechanism. An exact replay shows that the experienced opening is repeatedly sampled but never learned above the planner’s decision threshold. Independent search-budget and event-learning experiments are separating planner traps from dynamics failures.

| Checkpoint | Status | Required evidence |
|---|---|---|
| Research plan and repository | Complete | Public repository and source checkpoints |
| Exact-model planner | Complete | 240 paired-seed navigation runs; paper 001 |
| Online learned models | Running | 24 calibration fits; first online goal reached after 340 actions |
| Planned curiosity | Running | Initial planned/reactive/random comparison; mechanism learning unresolved |
| Noise and surprise | Pending | Useful uncertainty and measured recovery |
| Action memory / hierarchy | Pending | Benefit at equal compute |
| Platformer / Mario | Pending | Curiosity-only learning with no score or progress leakage |

## Papers

- [001 · Persistent plans under an exact simulator](papers/001-persistent-planning.md) · [PDF](papers/001-persistent-planning.pdf). Persistence improves average progress and precision, but does not dominate two-room success. Thirty held-out paired seeds per environment; complete outcomes and intervals.
- [002 · Learning the puck dynamics](papers/002-model-calibration.md) · [PDF](papers/002-model-calibration.pdf). Twenty-four development fits. Rare contact events account for most predictive error; disagreement is an imperfect ranking of error.

- [003 · Recall of a rare opening event](papers/003-event-recall.md). Exact replay of the original 2,500 actions and every optimizer update. Repeated training exposure does not produce a usable opening prediction.

- [004 · Search budget and wall traps](papers/004-search-budget.md). Fixed development and confirmation comparisons: individual rescues do not increase held-out success; longer horizons trade away population width and cost more wall time.

- [006 · Remembering is not generalizing](papers/006-event-learning.md). Reweighting generic large changes fits the recorded opening but produces false openings outside the switch. Nearest-neighbour memory also separates point recall from reusable knowledge.

Each paper includes the question, method, complete results, limitations, and reproduction commands. Development results and confirmatory comparisons are labelled separately.

## Current observations, not yet confirmed across seeds

- With a supplied goal and learned dynamics, the first open-world run reached the target at action 340, including 256 random warmup actions. This is task-directed control, not curiosity.
- In the first 2,500-action mechanism run, planned curiosity covered 121 grid cells but never opened the door. Reactive curiosity covered 79 cells and opened it at action 858; random exploration covered 88 and opened it at action 327.
- All three resulting frozen models reached the two left-room goals but failed the right-room goal from a fresh start. Exploration and reusable mechanism knowledge are therefore separate problems in this prototype.
- A matched failure replay is preserved for oracle seed 108: persistence stalls at the wall while fresh evolution reaches the goal.

## Artifact inspection

Papers 001 and 002 have each been rendered and visually checked as two A4 pages. The interactive trace viewer passes serialization and JavaScript interaction checks. Its browser screenshot inspection remains unverified because browser URL policy rejected the local file; no workaround was attempted for that viewer.

## Research rules

- Preserve negative results and record why a direction changes.
- Keep environment reward and evaluator information out of curiosity agents.
- Count experience from the first action, including warmup.
- Re-evaluate shifted plans from the actual new observation.
- Commit runnable checkpoints and their supporting results.
- Use independent reviews for consequential claims and confusing failures.

## Reproducible continuation

The observation/action agent can now save and restore its models, optimizer, replay, shifted genomes, and private random generators. Tests resume before fitting, during learning, and after the replay ring wraps; subsequent actions and model parameters match exactly. The caller saves environment state separately. Unbounded Gymnasium vector coordinates use a finite native-unit scale fallback; image encoding is still a separate milestone.

## Next fixed comparison

Study 005 compares planned curiosity, reactive disagreement, and random actions on five new paired seeds in each of the mechanism and noise worlds. Each run receives 2,500 real transitions and the same fitting schedule; planning compute is counted separately. All resulting frozen models are tested on the same three new goals. No setting is selected from this suite’s outcomes.

The optional Mario adapter is implemented and its metadata/mock checks pass. It accepts a supplied local compatible NES ROM, keeps integration files in ignored project storage, counts emulator frames, and exposes RGB observations without score or diagnostics. Actual emulator execution and visual world-model learning remain unverified. See [Mario setup](docs/MARIO.md).
