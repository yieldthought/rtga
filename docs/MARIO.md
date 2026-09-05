# Optional Mario environment

This prepares an RGB observation/action boundary for Mario. It does **not** provide a trained visual world model or demonstrate curiosity-driven Mario control. The current RTGAAgent consumes flat vectors; these images require a pixel encoder and recurrent visual dynamics before that agent can use them meaningfully.

Stable-Retro 1.0.1 supports the project's Python 3.12/macOS ARM64 runtime and native Gymnasium. Its installed `SuperMarioBros-Nes-v0` integration includes metadata and start states. The Mario ROM is not included in Stable-Retro; supply an explicitly selected local `.nes` file you are authorized to use. This module never searches for or downloads ROMs. [Stable-Retro package](https://pypi.org/project/stable-retro/), [getting started](https://stable-retro.farama.org/getting_started/).

Install the optional dependency with `.venv/bin/python -m pip install -e ".[mario]"`.

## Read-only checks

```bash
.venv/bin/python -m rtga.mario --check
.venv/bin/python -m rtga.mario --check --rom /explicit/path/to/game.nes
```

The first command verifies installed integration metadata without a ROM. The second additionally checks the supplied ROM's SHA1 against that metadata. For NES files, Stable-Retro hashes the bytes after the 16-byte iNES header. The installed 1.0.1 integration expects body SHA1 `facee9c577a5262dbe33ac4930bb0b58c8c037f7`. Checks create no files and do not construct an emulator. A hash match identifies a compatible ROM version; it is not an emulator smoke test.

## Constructing the adapter

```python
from rtga.mario import MarioConfig, create_mario

adapter = create_mario(
    "/explicit/path/to/game.nes",
    MarioConfig(state="Level1-1", frame_repeat=4, resize=(84, 84), grayscale=True),
)
observation = adapter.reset(seed=0)  # uint8 [84, 84, 1]; no reward or info
transition = adapter.step(2)       # index 2 = right
rgb = adapter.rgb_observation()    # unresized observed RGB retained separately
evaluation = adapter.evaluator_metrics()  # experiment runner only
adapter.close()
```

Construction validates the explicit ROM before copying anything. It copies integration metadata/start states and that ROM to `runs/mario/integrations/<body-sha1>/SuperMarioBros-Nes-v0/`, which is under the project's ignored `runs/` tree. It does not call Stable-Retro's import helper, modify the installed package, or write user-global data. A configured integration root must resolve within this project. Existing differing files are rejected rather than overwritten. Do not publish this local ROM directory.

The loader registers that directory as a custom integration and verifies that it resolves to the intended local ROM before constructing the environment. If another custom registration shadows it, the loader fails explicitly. The default is the ordinary `Level1-1` state, not the integration's alternate 99-lives state. Available states are printed by the check command. [Integration format](https://stable-retro.farama.org/integration/).

## Observation and action semantics

Default configuration preserves native RGB and advances one emulator frame per action. Optional nearest-neighbor resizing uses `(height, width)`; grayscale computes rounded luminance with coefficients 0.299/0.587/0.114 and retains a channel dimension. No frame stacking, max-pooling, RAM features, score extraction, or inferred velocity is silently added. `rgb_observation()` returns a copy of the most recently observed original RGB frame.

The twelve declared button choices are noop, left, right, jump, left+jump, right+jump, left+run, right+run, left+run+jump, right+run+jump, down, and fire/run-button alone. Both directions remain available; menu buttons and opposing directions are excluded. `adapter.action_names` and `adapter.action_mapping` expose the exact mapping. Stable-Retro is configured for raw button inputs, and the adapter restricts them explicitly. [Python API and action spaces](https://stable-retro.farama.org/python/).

Frame repetition stops immediately at either termination or truncation. The final observation is delivered before any reset. `evaluator_metrics()["frame_counts"]` reports `step_frames`, `agent_decisions`, `last_step_frames`, and `reset_calls`. Counts are cumulative over this adapter's lifetime. `step_frames` counts actual calls to Stable-Retro's one-frame step; constructor/reset frames are excluded and reset calls are reported separately. Count repeated frames when comparing interaction budgets. If discounting is per emulator frame, an action lasting `d` frames advances discount by `gamma**d`.

Only `Transition(observation, terminated, truncated)` crosses the agent-facing boundary. External reward is summed over repeated frames and retained with the last info dictionary exclusively in `evaluator_metrics()`. Strict curiosity code must receive neither. On a boundary, feed the final observation to the agent's final-observation handler, reset explicitly, and mark the next initial observation with `is_first=True`; do not train a transition from death/end-state to reset-state. Episode reset conventions still influence the experiment and must be reported.

## Verification status

The package import, installed Mario metadata, and button schema have been inspected. Tests cover ROM hash checks using synthetic non-game bytes, project-local copying, write-free checks, refusal to overwrite edited integration files, button mapping, RGB preprocessing, repeated-frame counts, final observation handling, and reward/info separation using mock environments. No supplied Mario ROM was available when these tests were written, so **actual Mario reset/step behavior remains untested**.
