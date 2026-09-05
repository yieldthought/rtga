# What the ensemble's disagreement currently means

Review of the checked implementation and studies 002, 003 and 006, 5 September 2026. This is an analysis and proposed experiment; no model, replay, objective or production default was changed.

**The current model is an independently initialized deep ensemble trained with independent minibatches from one shared dataset. It is not a persistent bootstrap of that dataset.** This distinction is real, but it does not establish that the ensemble is ineffective or that fixed bootstrap weights will improve it. The observed failure is narrower: members can agree on an incorrect rare transition, and increasing that transition's replay frequency can improve point recall while producing incorrect generalization. Test the uncertainty estimator on frozen data before spending another online exploration run on it.

## Checked behavior

| Component | Actual behavior | Consequence |
|---|---|---|
| Replay | [`ReplayBuffer.sample`](../rtga/models.py) draws a fresh `K × B` array of uniform indices with replacement on every optimizer update. There are no stored member masks, counts or weights. | Every member optimizes the same expected empirical loss. Members differ in initial parameters and stochastic optimization history, not in a persistent empirical dataset. |
| Networks and optimizer | Three separately parameterized two-hidden-layer, width-64 SiLU networks by default. Batched layers keep parameters separate. Per-member losses are summed; gradient clipping is per member. | The batched implementation does not share learned parameters across members. The shared feature trunk is between output coordinates **within** a member. |
| Continuous outputs | Gaussian normalized state-change means and diagonal variances; log variance bounded to `[-8,1]`. | A predicted variance can absorb stochasticity, unresolved dynamics, approximation error or incomplete fitting. Naming it `aleatoric` does not validate that interpretation. |
| Binary outputs | A next-value Bernoulli probability replaces the residual prediction for declared binary coordinates. Its variance is `p(1-p)`. BCE has weight 10 relative to each continuous coordinate. | For a deterministic switch, `p≈0.5` can reflect ignorance or a smoothed decision boundary. It need not mean that the physical switch is stochastic. |
| Prediction bounds | Means are clipped to declared observation bounds; variances are unchanged. The component Gaussian is not truncated to those bounds. | Compare both clipping rates and predictive calibration. Clipping can conceal raw mean disagreement near a boundary, and Gaussian tails may lie outside the observation space. |
| Curiosity | [`LearnedEvaluator`](../rtga/agent.py) computes normalized variance of **means**, on the same input state/action for all members, averaged over coordinates. Predicted variances are returned but ignored. | Correctly avoids confusing already-diverged paths with local disagreement, but does not compare full predictive distributions or directly estimate information gain. |
| Imagined trajectories | Each candidate has one carrier per member. The carrier keeps that member's identity, advances its conditional mean, and thresholds binary probabilities at 0.5. All members evaluate every carrier, costing `K²NH` member transitions per population evaluation. | These are coherent model alternatives, not samples of stochastic outcomes. A door with probability 0.49 is always closed on that carrier. Tail risk from stochastic outcomes is not integrated. |
| Online fitting | Default: 256 real warmup transitions, 200 initial updates, then eight updates per eight real transitions, batch 128/member. Prediction error is measured before fitting on the newly observed transition. | Repeated replay of one event is optimization exposure, not repeated independent evidence about that event. |

The source's phrase “independently bootstrapped replay batch” describes batch resampling accurately, but can mislead readers into assuming fixed member datasets. Prefer “independent replay minibatches” when describing this implementation. The current trace diagnostic `disagreement_before_update` is in raw observation units; the planner's curiosity is normalized by observation scale. Their numbers are not interchangeable.

For fixed replay size `n`, after `U` batches of size `B`, one member's event-draw count has mean `UB/n`; its chance of never drawing that event is `(1−1/n)^(UB)`. Increasing updates therefore makes the members' aggregate sampling proportions converge, even though nonlinear optimization can retain different solutions. In a conventional size-`n` bootstrap, counts are sampled once and reused. A single observation is omitted by a member with probability `(1−1/n)^n ≈ 0.368`. With three independently bootstrapped members, approximately 5.0% of ensembles omit it from **all** members, and 25.3% include it in all members. These are sampling calculations, not measured outcomes of a new experiment. Bootstrap diversity can thus come at the cost of withholding our only positive event.

## What the primary papers support

**PETS:** Each member is trained on an `N`-draw, with-replacement bootstrap dataset. Each network predicts a distribution. Its trajectory sampling propagates samples from those distributions; the fixed-member variant keeps a particle's model identity over time while still sampling outcomes. Our fixed carriers preserve identity but omit that second source of variation. PETS motivates separating within-model outcome variance from between-model differences; it does not make our mean-only carriers or three members a validated substitute. [Chua et al., 2018, §§4–5](https://arxiv.org/html/1805.12114)

**Deep ensembles:** Fixed bootstrap datasets are not a prerequisite for useful neural predictive uncertainty. Lakshminarayanan et al. train each network on the whole dataset with independent initialization and sampling. Bagging worsened their experimental performance. That is a direct reason to compare our existing method fairly, instead of treating persistent bootstrap as a correction guaranteed by the literature. Their results concern their tested predictive tasks, not this switch or curiosity objective. [Lakshminarayanan et al., 2017, §2.4](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)

**Randomized prior functions:** A member predicts `trainable_function + fixed_random_function`; training updates only the trainable part against a perturbed dataset. This preserves a distinct prior contribution beyond initialization alone. Exact Bayesian equivalence is established for a particular Gaussian linear setting; the nonlinear evidence does not imply calibrated posteriors for our mixed Gaussian/Bernoulli model. It is a plausible ablation for shared extrapolation errors, with an extra prior-scale choice and compute cost. [Osband et al., 2018, §3 and Algorithm 1](https://arxiv.org/html/1806.03335)

**MAX:** The local objective compares next-state **distributions**, using Jensen–Shannon divergence for its discrete formulation and a Jensen–Rényi approximation for Gaussian predictions. Its continuous implementation also tempers sensitivity to predicted variances; it does not simply trust all learned variances. Our same-input scoring follows the relevant local comparison, but variance of means is a different statistic. The paper supports a distribution-aware comparison as an experiment, not automatic noisy-TV immunity. [Shyam et al., 2019, §§2.2–2.4](https://proceedings.mlr.press/v97/shyam19a/shyam19a.pdf)

## Evidence from this repository

These studies did not compare bootstrap schemes or randomized priors. None identifies their absence as the cause of an error.

| Study | Observed evidence | What it leaves unresolved |
|---|---|---|
| [002: passive dynamics](../papers/002-model-calibration.md) | Across 24 fitted ensembles, final same-input disagreement/error Spearman ranges from −0.167 to 0.496; five correlations are negative. Contact-like transitions concentrate much of the residual motion error. | Whether disagreement predicts **reducible** error, and whether another ensemble construction improves it. Door and noise coordinates are constant in these data. |
| [003: exact event replay](../papers/003-event-recall.md) | The single naturally observed opening is drawn 236, 257 and 243 times by the three members. No member ever reaches probability 0.5 on that event. Final ensemble probability is 1.402%, with member probability variance `1.5742e−5`. | Relative effects of representation, loss balance, event prevalence and optimizer dynamics. This is failure to acquire reliable point recall, not established forgetting after mastery. |
| [006: frozen event data](../results/006-event-learning/PROTOCOL.md) | At 3,000 updates, uniform replay gives mean event probability 2.703%, zero opening recall on the controlled grid. Generic high-change replay gives 90.482% point probability, 39.533% opening recall, but 40.889% false positives on outside-switch interactions. | Whether a different uncertainty estimator would identify or repair the generalization errors. This sampler changes the fitting distribution; it is not a bootstrap ablation. |

The 006 figures above average neural seeds 0, 1 and 2; they are not percentages over independent environments. All candidates use the same 2,500-transition trajectory containing one opening event. High-change replay puts half each batch on the 25 largest normalized changes and half on uniform data. At 3,000 updates, the opening event has been drawn 7,592–7,893 times per member, versus 133–176 for uniform replay. Repetition improves its influence without supplying additional switch locations. Continuous normalized motion MSE also rises from `8.06e−6` to `4.39e−5`. Wrong-action false positives on the specified inside-switch grid remain zero; the observed defect is primarily spatial/velocity generalization, not indiscriminate action confusion.

Study 006 was running when this review was assigned. At inspection, its saved result reports `completed=true`, 13 model/checkpoint records and 24 frozen navigation trials, with unchanged dataset and protocol. High-change models succeed in 1/18 navigation trials, including 0/9 from the default start. The single kNN memory candidate recalls the stored event at 99.979%, but succeeds in 0/6 navigation trials. Uniform models did not meet the prespecified navigation gate; their zero trials must not be reported as zero successes out of attempted trials. These outcomes constrain claims about useful learned dynamics; they are not exploration results or an uncertainty comparison.

## Limits of the present objective

For normalized coordinate `j`, the implemented local score is

`D(s,a) = mean_j Var_k[ μ_k,j(s,a) / scale_j ]`.

For the ensemble's own predictive mixture, the total variance identity is

`Var(Y | s,a) = mean_k σ²_k(s,a) + Var_k μ_k(s,a)`.

This identity is exact for the mixture being represented. Calling its terms physical stochasticity and epistemic uncertainty is an additional modeling claim that needs evidence. All members can share a misspecified mean or inflate their predicted variance around an unresolved deterministic discontinuity.

Several concrete limitations follow:

1. **Same mean, different distribution.** Members predicting `N(0,0.01)` and `N(0,1)` yield zero mean disagreement despite disagreeing about outcomes. Conversely, a fixed mean separation receives the same score whether noise standard deviation is 0.01 or 10. The observation can be much less informative in the second case.
2. **Bernoulli probabilities have a useful exact comparator.** For a single binary coordinate, the member-index information statistic is `h(mean_k p_k) − mean_k h(p_k)`, where `h(p)=−p log p−(1−p)log(1−p)`. Near common probability `p`, it is approximately `Var(p_k)/[2p(1−p)]`; probability variance alone is not its scale. This quantity describes the fitted ensemble's disagreement, not guaranteed information about the true system. Three identical wrong probabilities give zero for either score.
3. **Representation and scale matter.** Averaging coordinate variances gives equal declared-scale weights, not equal information value. Duplicating a feature or adding many predictable coordinates changes the average. Bernoulli saturation and clipping can reduce measured disagreement while errors persist.
4. **Rollout fidelity matters separately.** Mean propagation through a nonlinear model does not generally equal the mean of propagated stochastic states. Thresholding a binary coordinate discards intermediate event probabilities. A mean-minus-standard-deviation penalty over the current carriers measures variation between these model paths; it is not a complete risk measure over outcomes.
5. **The score is not cumulative information gain.** The model is frozen throughout each imagined sequence. A plan can repeatedly collect the same local disagreement without accounting for how the first observation would change the model. Replanning after real observations helps operationally but does not make the imagined sum a posterior-update calculation.

Changing the estimator to distribution divergence should be evaluated on the **same frozen predictions** first. That separates a scoring change from training changes. For the mixed output, joint mixture divergence cannot generally be replaced by the sum of per-coordinate divergences: member identity couples coordinates. A Monte Carlo joint mixture estimate and a clearly labeled coordinatewise diagnostic are different estimands.

## Smallest decisive next ablation

Run one offline comparison with three arms. Keep three members, architecture, initialization seeds, optimizer, loss, scaling, bounds and all evaluation inputs fixed. Do not run curiosity planning during this comparison.

| Arm | Only intended difference |
|---|---|
| A — current | Fresh uniform independent minibatches, exactly as today. |
| B — fixed bootstrap weights | Draw independent `Poisson(1)` integer weights once for every raw transition/member. Sample each member's batches proportionally to its retained weights. Never redraw them at optimizer steps or checkpoints. This is a Poisson bootstrap approximation, not exactly an `N`-draw multinomial bootstrap. |
| C — B plus prior | Reuse B's weights. Add a separately initialized, permanently frozen two-layer width-64 prior to the mean/logit output before the existing likelihood transformation. Keep variance heads trainable without a prior. |

The smallest clean data contrast is the existing [003 replay](../results/003-event-recall/replay.npz), once with its sole opening row omitted and once with all rows present. Use stable transition identifiers so all retained rows have identical B/C weights in both data versions. This is an offline leave-one-event diagnostic; the omitted-event dataset still contains later open-door experience and is **not** the history available before the opening.

Concrete run specification for a new study-local runner, **not implemented by this review**:

- Dataset hash: `0a1d3cfc75c10485b3c718ae28ce4265079a22d125ca126daf9b20db9a91b7f2`; opening row index 326. Preserve the file. Both data versions are frozen before fitting.
- Outer seeds `70000..70019`, three arms, two data versions: 120 fresh fits. Each outer seed generates an ensemble, not three independent experimental replicates. Reuse trainable initialization across paired arms; use separate recorded streams for initialization, weight generation, minibatches and priors.
- Batch 128/member; Adam 0.002; snapshots at updates 0, 1,000 and 3,000, continuing the same fit. Each fitted condition receives 1,152,000 member-sample presentations. Report wall time separately because a prior adds work.
- For a first falsifiable prior choice, set its multiplier to `0.1` in normalized continuous-delta units and `1.0` in binary-logit units before fitting. These are provisional choices, not established optimum scales. Use independent prior initialization; never adjust it using probe labels. Record initial output spread and clipping rates so saturation is visible.
- Reuse the saved 006 grid and motion probes without training on them. Report the exact event separately from other grid points. Report all three member probabilities, raw mean and variance arrays, event weights and realized sampling exposure. Include every seed, including ensembles whose members all omit the event. Such outcomes are part of the method.
- Save a protocol before execution, source hashes, models, RNG states and raw probe predictions. No best-checkpoint or best-seed selection. Keep the existing high-change results as contextual evidence; do not add prioritization to B or C in this test.

Primary evaluation should combine **accuracy and uncertainty**, rather than rewarding spread:

- On the switch grid: positive and negative Brier/NLL separately; inside-switch recall; outside-switch and wrong-action false positives; equal-count reliability bins with bin counts. Global Brier alone can hide the rare positive class. The grid's prevalence is an evaluation choice, so do not interpret its reliability as calibration under an online policy's visitation distribution.
- On motion probes: position and velocity RMSE separately, continuous mixture NLL, marginal 50/80/95% predictive-interval coverage **and interval width**, plus results conditioned on contact-like events and distance from training inputs. Use mixture quantiles rather than silently replacing the mixture with a Gaussian.
- For ranking: same-input disagreement versus held-out error, area under a selective-risk curve, and low-disagreement/high-error counts. A method that merely raises all uncertainty should not improve rank-based scores. Report per-coordinate scores as well as the current average.
- For information sensitivity: compare the full-data and omitted-event fits on the exact point and the rest of the grid. Does adding the one distinct event improve held-out predictions, and does the prior disagreement identify the affected region? One surprising event can legitimately increase disagreement by exposing an alternative explanation; do not require a decrease after every observation. This tests a controlled data perturbation, not a complete online information-gain estimate.
- Pair differences by outer seed and show all 20 values plus paired bootstrap intervals. These intervals describe optimizer/ensemble randomness conditional on **one dataset**, not variation over environments. Label secondary metrics and avoid promoting an isolated significant result among many comparisons.

The decision is whether B improves error ranking/calibration over A without sacrificing event and motion predictions, and whether C adds useful uncertainty on omitted or poorly covered inputs that becomes less necessary when evidence is supplied. If it only increases disagreement, or improves stored-point recall with worse spatial false positives, it has not earned an online default. A negative result would be useful: it directs effort toward the function class, event representation and data diversity instead of ensemble bookkeeping. This first comparison will not settle all possible prior scales or bootstrap units.

## What is required before claiming separation from stochastic noise

The deterministic studies above cannot establish noisy-TV avoidance. A unit test in [`test_models.py`](../tests/test_models.py) does fit repeated Gaussian outcomes at one input and checks positive predicted variance with smaller mean disagreement. That verifies a basic numerical capability, not spatial or long-horizon calibration.

A subsequent frozen-data noise panel should collect **independent outcomes at exactly the same observed state/action** under independent environment RNG draws, with additional held-out input locations and independent test outcomes. In PuckLab, interacting near the noise source refreshes its last coordinate from `Uniform(−1,1)`; other actions preserve the current value. The observed current noise value must be included in the conditioning state. Matching only position would mix different conditional distributions.

Use declared unique-outcome counts such as 8, 32 and 128 at each queried input, with paired datasets across arms. Log every collection call; these controlled resets/queries are evaluator diagnostics, not an agent's sample-efficiency result. As genuinely new outcomes accumulate, assess whether conditional-mean error and between-member disagreement shrink while predicted within-member variance approaches the observed conditional variance. At a triggering input the evaluator knows the target mean is zero and variance is `1/3`; off-trigger the next sensor value is deterministic. A Gaussian can match those first two moments without matching uniform tails, so retain coverage and density-misspecification diagnostics.

Use independent test outcomes to evaluate predictive NLL, interval coverage and widths; estimate error of the **conditional mean** from repeated outcomes, rather than equating a single noisy residual with model ignorance. A stochastic source can initially offer real information about its mean, variance or triggering conditions; the desired property is reduced attraction after those become predictable, not zero curiosity from the outset. Keep replay duplicates separate from new measurements: ten presentations of the same stored draw add no evidence. The same separation is needed to measure learning progress after acquisition rather than after extra optimization alone.

Fixed per-transition bootstrap weights assume the replay rows are suitable resampling units. Adjacent transitions from one trajectory are correlated; persistent row weights do not solve that statistical problem. With multiple independent episodes, an episode/block bootstrap becomes a separate defensible comparison. Record this limitation now, rather than calling one continuing trajectory an independent sample of every state.

## Provenance of this review

Inspection timestamp: `2026-09-05T10:17:58Z`. Repository HEAD then: `83a3aba69ecae77266ce365258e4a62b22960adf`; workspace contained uncommitted 005/006 artifacts. Checked file SHA-256 values:

| File | SHA-256 |
|---|---|
| `rtga/models.py` | `dd329c391f93a317f185a895fb8369b0e39370c1b77e5ad7f67f5dbcfa6b298f` |
| `rtga/agent.py` | `54d8870e9ad39010e44204be1503e9392ca3ac4b418bc1324af9e454b8babe80` |
| `rtga/event_learning_study.py` | `f5d96351db4bdb8eca2dc70f0bb844f9cecc6829ebd7a12201f2e6b53b7c7b94` |
| `results/002-model-calibration/results.json` | `315d5f488782b8c3cbe713cdce2f6a1e3f9fcc1d0e0ec1cdc63fdbf7fbaf825a` |
| `results/003-event-recall/results.json` | `52f52dfdd53e7badd373b9c82fb39f85eea0b4b25829f3ea91061e9934d915cb` |
| `results/006-event-learning/results.json` | `4e5d97c4eaed8c4ea2e6f6dea403e498a30cef7d742708a4726b22bfc9cd8d36` |

The archived 002/003 numerical model source has hash `1d31a88ba68eb37eafd804c1d159d79d59aa57c6628240ef96ae6ee3b34cbbbb`. Its difference from the reviewed model is configuration validation for infinite observation bounds; replay sampling, training loss and predictive equations are unchanged. Study 006 uses the reviewed model hash. Result aggregates above were checked against the saved raw records; no new fitting or environment experiment was performed for this document.
