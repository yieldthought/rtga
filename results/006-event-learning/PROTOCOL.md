# Declared before fitting: offline rare-event learning diagnostic

This study uses all 2,500 unchanged real transitions in
`results/003-event-recall/replay.npz` from the beginning of fitting. It is an
offline representation/data diagnostic, not exploration success and not an
additional sample-count baseline for study 003. The source file is hashed and
left unchanged. Evaluation transitions never enter training or model selection.

The candidates are fixed before any fit:

1. Current three-member Gaussian/Bernoulli neural ensemble, uniform replay.
2. The same architecture, loss, initialization seeds, optimizer and batch size;
   each batch draws 64 transitions uniformly from all data and 64 uniformly
   from the highest-change 1% (25 transitions). Change is
   `max(abs(next_state - state) / observation_scale)` across all coordinates.
   Stable descending sorting resolves ties. No door labels, coordinates,
   action identity, or event-specific priorities enter this sampling rule.
3. A deterministic action-conditioned five-nearest-neighbour memory model.
   Distance is squared Euclidean distance after the same observation scaling.
   Weights are proportional to `1 / (squared_distance + 1e-6)`. Continuous
   outputs interpolate observed residuals; the binary output interpolates
   observed next values. Outputs respect the same observation bounds.
   This is an empirical-memory diagnostic, not a proposed final architecture.

Neural seeds are 0, 1 and 2; each seed creates three ensemble members. Sampling
seeds are model seed + 101. Batch size is 128 per member, Adam learning rate
0.002, hidden width 64, binary loss weight 10, and the existing member gradient
clipping and variance heads are unchanged. Evaluate at 1,000 and 3,000 total
optimizer updates, continuing the same fit between checkpoints. No tuning or
selection on the evaluation probes. kNN has no fitting seed or optimizer budget.

Primary measurements:

- Probability at the exact experienced opening state/action, member values
  and ensemble mean; 0.5 is the planner's existing binary threshold.
- Evaluator-only real one-step transitions on a 13-by-13 position grid around
  the switch, x=0.18..0.42 and y=0.38..0.62 in steps of 0.02, door closed, all
  six actions, and six velocity settings: zero, each cardinal velocity of
  magnitude 0.12, and the exact experienced opening velocity. Report opening
  recall, outside-switch interaction false positives, wrong-action false
  positives, positive/negative Brier scores, and probability by velocity.
  These distinguish a stored point from a conditional interaction rule.
- Continuous motion error on a separate fixed probe set spanning both rooms,
  both door states, all actions, and velocities drawn from [-0.2, 0.2] using
  evaluator RNG seed 60601. Position error and velocity error are separate;
  binary errors are excluded from continuous-motion metrics.

If a candidate reaches mean opening probability >=0.5 at the experienced
event or has nonzero opening recall on the controlled grid, run frozen
goal-directed navigation at the 3,000-update checkpoint (or the single kNN
candidate): 100 steps from the default start, and 100 from the exact opening
state, toward (0.8,0.5). Use the existing persistent planner (population 48,
horizon 12, generations 3), planner seeds 40000,40001,40002. Record full real
trajectories, door opening, goal attainment within radius 0.06, final distance,
and model-call count. These are task-directed diagnostics with frozen models;
they do not establish curiosity-driven exploration.

Source snapshots, actual training sample counts and opening-event exposure
(counted after sampling, without influencing it), full probe predictions,
all checkpoints and exact commands will be retained. Gradient effects are
not identified merely by counting batch inclusion. A single opening event
cannot determine the true interaction region; limits are part of the result.
