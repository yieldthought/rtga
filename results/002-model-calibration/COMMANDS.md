# Commands executed

Run from `/Users/moconnor/Projects/rtga`:

```bash
.venv/bin/python -m rtga.model_study --output results/002-model-calibration
.venv/bin/python results/002-model-calibration/analyze.py
```

The study was run twice. The second run adds descriptive constant-state and
observed nonzero-to-zero velocity-event metrics, with unchanged model settings,
training examples, probe trajectories, and fitting budgets. All repeated model
prediction metrics are identical. Runtime values in results.json are from the
second run. No checkpoint selection or evaluation-driven hyperparameter search
was performed.

The result provenance records Python/library versions and SHA-256 hashes of
the exact model, environment, and study source. Matching source copies are
preserved in source/. The independent real datasets and final checkpoints are
preserved, with their relative paths listed per run in results.json.
