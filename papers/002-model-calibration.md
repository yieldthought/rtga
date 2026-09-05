# Small learned models predict ordinary motion before they predict collisions

Development study 002 · 5 September 2026 · Three seeds per condition; exploratory evidence, not a benchmark claim.

A three-member probabilistic ensemble learns useful short-range dynamics from 500–2,000 random-action transitions. Its remaining errors concentrate on collisions, and its disagreement is an inconsistent ranking of real prediction error. More spatial coverage helps expose the world but does not, by itself, establish a useful long-horizon simulator.

## Protocol

We fit fresh ensembles in the existing `open` and `rooms` PuckLab variants. Every training example is an actual observation/action/next-observation transition collected by uniform random actions. Training starts at `(0.20, 0.50)` with zero velocity. There is no goal, reward, task-trained initialization, reset, or hypothetical simulator query. The agent acts every primitive frame; action hold 8 means repeating a randomly selected primitive action for eight successive actual frames. It is a temporally correlated behavior policy, not a faster macro simulator.

The matrix is two environments × action holds `{1,8}` × dataset sizes `{500,2000}` × seeds `{0,1,2}`: **24 independently fitted ensembles**. The smaller dataset is a prefix of the larger logged trajectory. Both conditions receive the same fitting budget. Each model starts fresh and uses three separately parameterized, two-layer 64-unit networks; independent bootstrap batches of 128 transitions per member; Adam at 0.002; input/delta scales `[1,1,.5,.5,1,1]`; Gaussian log-variance bounds `[-8,1]`; and a Bernoulli head for coordinate 4, with training loss weight 10. Checkpoints are taken before fitting and after 100 and 500 optimizer batches. The study did not adjust these choices against the probe results.

Evaluation uses separate actual trajectories from 16 fixed starting positions, each with action holds 1 and 8: **32 trajectories of 48 transitions per environment/seed**. The positions differ from the training start; they are not guaranteed to lie outside all regions visited during collection. Nearest training-position distances are recorded. Probe trajectories and action sequences are reused across paired conditions, never inserted into replay, and never used for fitting.

One-step physical MSE averages squared error over observed `(x,y,vx,vy)` in their declared units. Position and velocity errors are also retained separately. Eight- and 24-step metrics are endpoint MSEs from coherent conditional-mean rollouts: each branch retains one ensemble member throughout. At the 20 Hz environment rate these horizons are 0.4 and 1.2 seconds. To measure epistemic disagreement, all members receive the **same** state/action; uncertainty is not inferred from the spread of already-diverged trajectory endpoints. Reported Spearman correlations compare local disagreement against normalized physical squared error on actual one-step probes. They test error ranking, not calibrated probabilities or information gain.

## Results after 500 fitting batches

Each table entry is a mean over three development seeds. Full individual values, ranges, and per-transition arrays are retained in the artifacts. Adjacent probe transitions are correlated and are not additional independent seeds.

| World | Action hold | Real transitions | Spatial coverage | 1-step MSE | 8-step MSE | 24-step MSE | Disagreement/error Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| open | 1 | 500 | 9.8% | 0.000146 | 0.00128 | 0.0134 | 0.191 |
| open | 1 | 2,000 | 27.1% | 0.000143 | 0.00117 | 0.00979 | 0.234 |
| open | 8 | 500 | 19.8% | 0.000179 | 0.00173 | 0.00882 | 0.229 |
| open | 8 | 2,000 | 60.0% | 0.000158 | 0.00152 | 0.00964 | −0.022 |
| rooms | 1 | 500 | 9.5% | 0.000215 | 0.00105 | 0.0112 | 0.357 |
| rooms | 1 | 2,000 | 24.2% | 0.000209 | 0.000973 | 0.00770 | 0.268 |
| rooms | 8 | 500 | 18.0% | 0.000267 | 0.00252 | 0.0139 | 0.291 |
| rooms | 8 | 2,000 | 54.8% | 0.000230 | 0.00155 | 0.00793 | −0.001 |

Coverage is the fraction of all 256 square grid cells visited during training, including cells that may overlap walls in `rooms`. It is a descriptive evaluator metric, not a policy objective.

**Ordinary motion becomes predictable quickly.** Averaged over all data policies and sizes, physical one-step MSE decreases from 0.00333 to 0.000156 in `open`, and from 0.00340 to 0.000230 in `rooms`. A post-hoc constant-state baseline predicts the current observation indefinitely. Every fitted ensemble beats that baseline at one and eight steps; 20 of 24 beat it at 24 steps. The longer predictions remain imperfect: per-coordinate position RMSE at step 24 ranges from 0.070 to 0.193 across runs in a unit-square world.

**Residual error concentrates on contact-like events.** As a post-hoc diagnostic, flag an actual transition if a velocity coordinate drops from magnitude above 0.025 to below `1e-7`. This uses observed transitions, not the environment's collision implementation; it is a collision proxy rather than an oracle label. Such events account for only 0.74% of `open` probes and 1.06% of `rooms` probes, yet contribute means of 69% and 74% of total one-step physical squared error respectively. Their per-event MSE is roughly 300–350 times that of the remaining transitions. A small average loss therefore conceals the events that can invalidate a plan near a wall.

**Disagreement has a useful tail in some runs but weak global consistency.** Final Spearman correlations span −0.167 to 0.496, with five of 24 negative. Group-mean error among the top 10% most-disagreed transitions is 1.14–4.19 times the overall mean error. Tail enrichment and weak rank correlation can coexist. Increasing data coverage does not consistently improve either statistic. In particular, high coverage from held actions is insufficient evidence that agreement means accuracy. This study does not establish that disagreement predicts reducible error or actual learning progress; that needs a separate before/after information-acquisition experiment.

**Binary Brier scores are small but uninformative about mechanisms.** They range from `9.49e-7` to `2.76e-5`. Both requested variants keep the door flag at 1 throughout. No switch transition is present, and the noise sensor is constant. These results validate prediction of a constant flag only; they do not demonstrate mechanism learning, noisy-TV avoidance, or stochastic calibration.

## Fitting budget and implications for the next experiment

On the available Apple Silicon CPU with PyTorch restricted to one thread, median cumulative fitting time is **0.068 seconds for 100 batches and 0.335 seconds for 500 batches**. These timings exclude collection and evaluation. Going from 100 to 500 batches improves one-step MSE in 20 of 24 runs, but improves 24-step endpoint MSE in only 16. In `open`, mean 24-step error increases slightly, from 0.01005 to 0.01042; in `rooms` it decreases from 0.01342 to 0.01019. We did not select the better checkpoint per run.

For the first online agent, a practical starting budget is 100 initial fitting batches after its counted random warmup, followed by a small bounded update count such as 1–4 batches per real action. Keep 500-batch checkpoints as a comparison, not an assumed improvement. At the observed throughput, 1–4 updates cost roughly 0.7–2.7 milliseconds; this is a throughput extrapolation, not a measured end-to-end control latency. Test short planning horizons around eight primitive steps alongside longer ones, rather than declaring eight optimal from this prediction study.

The next questions should be whether more actual contact experience and generic replay prioritization of high real prediction errors improve held-out contact predictions; whether such improvements translate into closed-loop task control; and whether conditional disagreement identifies transitions that yield subsequent learning progress. Keep unseen layouts and active exploration separate from this fixed-layout, passive-data test. Do not replace the learned model's contact errors with hidden exact-wall constraints and call the resulting controller learned.

The architecture, variance floor, shared feature trunk, fitting schedule, and amount of contact data are all possible contributors to the remaining errors. This study identifies their concentration; it does not isolate those causes. Comparisons use fixed optimizer updates rather than fitting every dataset to convergence, and only three seeds. No claim about closed-loop mastery follows from these prediction results.

## Reproduction and artifacts

From the repository root:

```bash
.venv/bin/python -m rtga.model_study --output results/002-model-calibration
.venv/bin/python results/002-model-calibration/analyze.py
```

The source is [model_study.py](../rtga/model_study.py). Complete metrics, model settings, source SHA-256 hashes, runtime versions, and executed command are in [results.json](../results/002-model-calibration/results.json). [summary.json](../results/002-model-calibration/summary.json) contains per-seed aggregates; [summary.md](../results/002-model-calibration/summary.md) is the generated table. Actual trajectories are preserved in `results/002-model-calibration/datasets/`, all 24 fitted checkpoints in `checkpoints/`, and the exact source versions used in `source/`.

The constant-state baseline and contact-proxy breakdown were added after the first unchanged-configuration pass to explain its residual errors. Re-running produced identical prediction metrics. No model, data-policy, or fitting hyperparameter was tuned during that diagnostic addition.
