# Exact event recall: commands and provenance

Executed from the repository root with `.venv/bin/python`.

The first attempt stopped at the source-hash guard because the live agent and
model files had acquired input-validation changes during preparation. No
experiment ran under the changed core. Original source files were then copied
from commit `5f6b89e` to `reference-source/` using `git show` (read only):

```python
from pathlib import Path
import subprocess

output = Path('results/003-event-recall/reference-source')
output.mkdir(parents=True, exist_ok=True)
for name in ['__init__.py', 'agent.py', 'models.py', 'planning.py', 'envs.py',
             'experiments.py', 'online_experiment.py']:
    data = subprocess.check_output(['git', 'show', f'5f6b89e:rtga/{name}'])
    (output / name).write_bytes(data)
```

All four core files (`agent.py`, `models.py`, `planning.py`, `envs.py`) match
the SHA256 hashes recorded in the original random run. `online_experiment.py`
is retained for context and is not executed by this diagnostic; its committed
version has a provenance-only change relative to the original run.

Successful experiment command:

```bash
.venv/bin/python -m rtga.event_recall_study \
  --output results/003-event-recall \
  --reference runs/random-mechanism-dev0 \
  --core-source results/003-event-recall/reference-source \
  > results/003-event-recall/run.log
```

`source/` is the executed source snapshot, including the diagnostic harness.
`results.json.executed_source_sha256` records its hashes separately from the
live-worktree provenance. The reference package is imported under an isolated
module name. The diagnostic uses the original agent and original `train_steps`
calls; a post-optimizer hook only evaluates the known event. The replay wrapper
counts occurrences in batches already returned by the original sampler, with
no extra sampling, data insertion, or modification.

Validation completed successfully:

- All 2,500 transitions match an independent reconstruction of the random
  action stream and simulator trajectory exactly.
- All 625 original trace records match for state, action, metrics, model
  version, prediction error, and disagreement. Timing fields are excluded.
- Every checkpoint field, including all 24 parameter/optimizer tensors,
  matches the original saved model exactly; maximum tensor error is zero.
- Final metrics and the 2,440-update count match the original run.
- The retained replay contains exactly one closed-to-open transition and
  exactly one copy of the probed event.

The generated `event-recall.png` was inspected with the local image viewer.
No continuation beyond the validated original 2,500 steps was run.
