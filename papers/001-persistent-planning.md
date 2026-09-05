# 001 · Persistent plans under an exact simulator

Confirmation run · clean working tree · generated 2026-09-05 10:02 UTC

## Abstract

We compared persistent action-plan populations with fresh evolution, random shooting and categorical CEM using exact dynamics in a unit-square puck task. Each run contained 80 actions; the protocol declared 30 paired seeds per environment. In open, observed successes were Persistent RTGA 30/30; Fresh evolution 30/30; Random shooting 30/30; Categorical CEM 30/30. In rooms, observed successes were Persistent RTGA 26/30; Fresh evolution 28/30; Random shooting 26/30; Categorical CEM 27/30. These tests isolate search behavior with a supplied objective. They do not test learned dynamics, curiosity or transfer.

## Methods

The agent selects among six primitive actions, simulates 64 plans of 24 steps for 4 generations, executes the best first action, then replans. Fitness is negative discounted mean future distance (discount 0.99); success means observed distance ≤ 0.06 at least once. Open has no internal wall; rooms has a permanent open doorway. Initial positions and goals are matched by seed.

Persistent RTGA shifts all genomes and appends a random action. Fresh evolution uses the same genetic operators but initializes anew at each decision. Random shooting samples independent populations; categorical CEM refits per-position action probabilities with a 10% uniform floor and keeps the incumbent within each decision. CEM is initialized afresh between decisions, so this is not an iCEM comparison.

## Results

Persistence did not dominate the observed success counts: Fresh evolution reached more goals than persistence in rooms (28/30 versus 26/30); Categorical CEM reached more goals than persistence in rooms (27/30 versus 26/30). Improved distance control and reliable goal reachability are separate endpoints.

| Variant / method | Success | First hit† | Mean dist. | Final dist. | Latency p50 / p95, ms |
| --- | --- | --- | --- | --- | --- |
| open / Persistent RTGA | 30/30 | 42.5 | 0.226 | 0.001 | 3.50 / 3.91 |
| open / Fresh evolution | 30/30 | 49.5 | 0.270 | 0.006 | 3.51 / 3.97 |
| open / Random shooting | 30/30 | 53.5 | 0.293 | 0.008 | 3.24 / 3.66 |
| open / Categorical CEM | 30/30 | 43.5 | 0.230 | 0.005 | 3.38 / 3.80 |
| rooms / Persistent RTGA | 26/30 | 45.0 | 0.268 | 0.052 | 8.39 / 8.91 |
| rooms / Fresh evolution | 28/30 | 51.5 | 0.296 | 0.031 | 8.39 / 8.90 |
| rooms / Random shooting | 26/30 | 58.5 | 0.332 | 0.055 | 8.13 / 8.65 |
| rooms / Categorical CEM | 27/30 | 45.0 | 0.265 | 0.044 | 8.26 / 8.77 |

† Median first-success step among successful runs only; not used for paired inference. Failed-run first-hit times remain censored at the interaction limit. Mean distance averages all executed steps, then equally weights seeds; final distance equally weights seeds. Success counts retain every completed seed.

Recorded per-decision model-transition counts: 6,144. The suite contains 117,964,800 hypothetical transitions. Latency percentiles pool recorded decisions within each method/variant, rather than averaging run percentiles. Statistical intervals use seeds, not decisions, as independent units.

## Paired effects

Every effect is the mean paired difference, persistent minus comparator, within one variant. The 95% percentile intervals resample complete seed pairs 20,000 times (NumPy generator seed 20260905). Success effects are percentage points. Distance effects use world units. Restricted first-hit time is min(T, 80); failures contribute 80 actions. Negative distance/time differences favor persistence. Intervals are unadjusted across comparisons and endpoints; degenerate success intervals do not prove equal success probabilities.

| Variant / comparator | Δ success, pp | Δ restricted hit | Δ mean distance | Δ final distance |
| --- | --- | --- | --- | --- |
| open / Fresh evolution | +0.0 [+0.0, +0.0] | -7.4 [-8.5, -6.3] | -0.044 [-0.051, -0.038] | -0.005 [-0.007, -0.004] |
| open / Random shooting | +0.0 [+0.0, +0.0] | -11.1 [-12.3, -9.9] | -0.067 [-0.076, -0.059] | -0.007 [-0.008, -0.005] |
| open / Categorical CEM | +0.0 [+0.0, +0.0] | -0.5 [-0.8, -0.1] | -0.003 [-0.005, -0.002] | -0.004 [-0.005, -0.003] |
| rooms / Fresh evolution | -6.7 [-16.7, +0.0] | -5.2 [-7.7, -2.0] | -0.028 [-0.042, -0.010] | +0.021 [-0.005, +0.058] |
| rooms / Random shooting | +0.0 [-10.0, +10.0] | -10.9 [-13.3, -8.5] | -0.064 [-0.079, -0.049] | -0.003 [-0.035, +0.032] |
| rooms / Categorical CEM | -3.3 [-10.0, +0.0] | +0.3 [-0.7, +1.9] | +0.003 [-0.003, +0.013] | +0.008 [-0.005, +0.035] |

## Limitations

This is one simple environment family, one selected compute setting and known deterministic dynamics. The goal-directed distance objective provides dense guidance, and rooms does not require discovering a switch. Reaching the goal once is not the same as remaining there; mean and final distance expose that difference. The CEM baseline lacks cross-decision warm starts. Planning-time measurements exclude real environment steps, trace writing and training; they are not hard real-time guarantees. No learned-model or curiosity claim follows.

## Reproduction and provenance

- Results: [../results/001-oracle/results.json](../results/001-oracle/results.json)
- Result SHA-256: `a44d9f99c4c21dd7f53019823fe551014b2cd9b71cc4f9a45c21dc71ffa4fddc`
- Recorded run commit: `41bd35db6b875376d5f9a3557c4f4029c965042c` (clean working tree). The run began from a clean recorded commit. The result file also contains per-source SHA-256 hashes.
- Run timestamp: `2026-09-05T09:57:08.297674+00:00`
- Seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129
- Runtime: Python 3.12.11; NumPy 2.5.2; macOS-26.6.2-arm64-arm-64bit
- Command below: recorded invocation.

```sh
python -m rtga.cli oracle --output results/001-oracle --seeds 30 --seed-start 100 --steps 80 --population 64 --horizon 24 --generations 4
```

Report generation:

```sh
python -m rtga.oracle_report --input results/001-oracle/results.json --output papers/001-persistent-planning --stage confirmation --bootstrap-samples 20000 --bootstrap-seed 20260905 --run-command 'python -m rtga.cli oracle --output results/001-oracle --seeds 30 --seed-start 100 --steps 80 --population 64 --horizon 24 --generations 4'
```

### Recorded source hashes

```json
{
  "envs.py": "9883f95d36f3929cb61ca83eb0372550f0a082d8365fec5818f38121c8455956",
  "viewer.py": "91ca26171e2f2de4fbaa0f707803d811e8e567ffdbbb26ac92bac8a959f168d6",
  "models.py": "1d31a88ba68eb37eafd804c1d159d79d59aa57c6628240ef96ae6ee3b34cbbbb",
  "__init__.py": "aaf2010db85eef0acabcaf4dd3eeeeae037891e3b9a7d1c2b00dd900ac1b0c36",
  "experiments.py": "9ef5975bd5471b141b8ce1b2d36ac84aa4eec00c255979ff6b06f8581abd7a63",
  "cli.py": "f3ff13862cb15d0715c4279d63099bdb4a9523e90e3a2dac571aa2bf13c60d69",
  "planning.py": "201f301fa74b340f72ce97d542cede37d389cc0ab79a3adf89c71c64ce52b36a"
}
```

### Exact completed seed outcomes

| Variant | Method | Successful seed IDs | Failed seed IDs |
| --- | --- | --- | --- |
| open | Persistent RTGA | 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129 | none |
| open | Fresh evolution | 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129 | none |
| open | Random shooting | 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129 | none |
| open | Categorical CEM | 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129 | none |
| rooms | Persistent RTGA | 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 123, 124, 125, 127, 128, 129 | 108, 116, 122, 126 |
| rooms | Fresh evolution | 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 129 | 116, 126 |
| rooms | Random shooting | 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 123, 124, 125, 127, 128, 129 | 103, 116, 122, 126 |
| rooms | Categorical CEM | 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 123, 124, 125, 127, 128, 129 | 116, 122, 126 |

## Related methods

[RHEA with a shift buffer, Gaina et al. 2017](https://rdgain.github.io/assets/pdf/papers/gaina2017rhhybrids.pdf) is the closest published planning mechanism. [iCEM, Pinneri et al.](https://proceedings.mlr.press/v155/pinneri21a.html) is a stronger continuous-control reference with elite memory and temporal sampling structure, not implemented in this comparison.
