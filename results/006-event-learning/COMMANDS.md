# Offline event-learning study: commands and checks

The numerical study was executed after saving `PROTOCOL.md`, with no intervening
changes to its candidate rules, metrics or navigation gate:

```bash
.venv/bin/python -m rtga.event_learning_study \
  --output results/006-event-learning \
  --dataset results/003-event-recall/replay.npz \
  > results/006-event-learning/run.log
```

The current core sources are copied into `source/` before fitting and imported
under an isolated package name. Their hashes, the unchanged input dataset hash,
the protocol hash, and the exact interpreter arguments are in `results.json`.
No production model or replay code is changed. `OfflineReplay` and `KNNResidual`
are local diagnostic implementations in `rtga/event_learning_study.py`.

Six neural fits (three fixed seeds × two sampling distributions) are evaluated
after 1,000 and 3,000 updates. All 2,500 actual transitions are available from the
start, so this experiment must not be interpreted as an online-learning sample
efficiency comparison. Each member receives 128,000 draws at the first checkpoint
and 384,000 cumulatively at the second. No evaluation case is fed to training.

The retained controls comprise 6,084 real local interaction probes and 144 real
motion probes. Each of the four gate-passing final candidates (three high-change
fits and kNN) receives six frozen navigation trials, yielding 24 full real
trajectories. The gate was fixed before fitting. Models remain frozen during
all evaluations.

Spatial maps and the tabular summary were generated after fitting with:

```bash
MPLCONFIGDIR=/tmp/rtga-event-learning-mpl \
XDG_CACHE_HOME=/tmp/rtga-event-learning-cache \
.venv/bin/python results/006-event-learning/analyze.py \
  > results/006-event-learning/analysis.log
```

The average spatial map and the per-seed maps retain the complete evaluator
position grid. Red crosses indicate outside-region interaction predictions at
or above the existing 0.5 planner threshold. The main figure was inspected with
the local image viewer and adjusted to remove overlapping labels. Plotting does
not refit or select models.

Artifact checks verify that all twelve checkpoints exist, all source hashes
match the snapshots, the protocol and dataset hashes are unchanged, per-member
draw counts match the declared budgets, and all 24 trajectories contain 100 real
steps. No new collection condition or reset implementation is part of this study.
