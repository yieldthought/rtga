"""Optional Mario setup from an explicitly supplied local NES ROM.

No ROM search or download is performed. Integration files and the validated ROM
are copied only below this project's ``runs/`` directory by default. Agent-facing
outputs use ObservationActionAdapter's reward-free contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .adapters import ObservationActionAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME = "SuperMarioBros-Nes-v0"
BUTTON_ACTIONS = (
    (), ("LEFT",), ("RIGHT",), ("A",), ("LEFT", "A"), ("RIGHT", "A"),
    ("LEFT", "B"), ("RIGHT", "B"), ("LEFT", "A", "B"),
    ("RIGHT", "A", "B"), ("DOWN",), ("B",),
)
ACTION_NAMES = ("noop", "left", "right", "jump", "left_jump", "right_jump",
                "left_run", "right_run", "left_run_jump", "right_run_jump", "down", "fire")


@dataclass(frozen=True)
class MarioConfig:
    state: str = "Level1-1"
    frame_repeat: int = 1
    resize: tuple[int, int] | None = None  # Height, width; nearest-neighbor.
    grayscale: bool = False
    integration_root: str | Path | None = None

    def __post_init__(self):
        if not isinstance(self.frame_repeat, int) or self.frame_repeat < 1:
            raise ValueError("frame_repeat must be a positive integer")
        if Path(self.state).name != self.state or self.state.endswith(".state"):
            raise ValueError("state must be a plain integration state name without .state")
        if self.resize is not None and (len(self.resize) != 2 or any(
                not isinstance(x, int) or x < 1 for x in self.resize)):
            raise ValueError("resize must be positive integer (height, width)")


def _retro():
    try:
        return importlib.import_module("stable_retro")
    except ImportError as exc:
        raise RuntimeError("Optional Mario integration requires stable-retro==1.0.1") from exc


def _integration(retro, config: MarioConfig) -> tuple[Path, dict[str, Any]]:
    path = retro.data.get_file_path(GAME, "rom.sha", retro.data.Integrations.STABLE)
    if path is None:
        raise RuntimeError(f"Installed stable-retro has no integration metadata for {GAME}")
    directory = Path(path).parent
    expected = [line.strip().lower() for line in Path(path).read_text().splitlines() if line.strip()]
    if not expected or any(len(digest) != 40 or any(c not in "0123456789abcdef" for c in digest) for digest in expected):
        raise ValueError("Installed integration has invalid SHA1 metadata")
    states = sorted(file.stem for file in directory.glob("*.state") if not file.name.startswith("_"))
    if config.state not in states:
        raise ValueError(f"State {config.state!r} is not available in the installed integration")
    for name in ("data.json", "scenario.json", "metadata.json"):
        if not (directory / name).is_file():
            raise ValueError(f"Installed integration is missing {name}")
        json.loads((directory / name).read_text())
    return directory, {"game": GAME, "state": config.state, "available_states": states,
                       "expected_body_sha1": expected}


def _read_rom(rom_path: str | Path) -> tuple[bytes, str]:
    path = Path(rom_path).expanduser()
    if path.suffix.lower() != ".nes" or not path.is_file():
        raise ValueError("Supply an explicit existing .nes file; directories and archives are not searched")
    with path.open("rb") as handle:
        data = handle.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024 or len(data) <= 16 or data[:4] != b"NES\x1a":
        raise ValueError("ROM must contain a valid iNES header and fit within 32 MiB")
    # Match installed stable_retro.data.groom_rom: NES SHA1 excludes its header.
    return data, hashlib.sha1(data[16:]).hexdigest()


def check_mario(rom_path: str | Path | None = None, config: MarioConfig | None = None) -> dict[str, Any]:
    """Read-only package/integration check; optional supplied ROM hash validation."""
    config = config or MarioConfig()
    _directory, result = _integration(_retro(), config)
    result["stable_retro_version"] = metadata.version("stable-retro")
    result["rom_supplied"] = rom_path is not None
    result["rom_matches"] = None
    if rom_path is not None:
        _data, digest = _read_rom(rom_path)
        result.update(rom_body_sha1=digest, rom_matches=digest in result["expected_body_sha1"])
    result["emulator_rollout_tested"] = False
    return result


def _project_local(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("integration_root must remain inside this project")
    return path


def _write_same_or_new(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink integration file: {path.name}")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing project-local integration file differs: {path.name}")
    else:
        with path.open("xb") as handle:
            handle.write(content)


def prepare_mario(rom_path: str | Path, config: MarioConfig | None = None) -> Path:
    """Validate the supplied ROM, then create an isolated project-local dataset."""
    config = config or MarioConfig()
    source, description = _integration(_retro(), config)
    data, digest = _read_rom(rom_path)
    if digest not in description["expected_body_sha1"]:
        raise ValueError(f"ROM body SHA1 {digest} does not match installed {GAME} integration")
    base = Path(config.integration_root) if config.integration_root is not None else PROJECT_ROOT / "runs/mario/integrations"
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    destination = _project_local(base / digest / GAME)
    destination.mkdir(parents=True, exist_ok=True)
    for file in sorted(source.iterdir()):
        if file.is_file() and file.suffix in {".json", ".state", ".lua", ".sha"}:
            _write_same_or_new(destination / file.name, file.read_bytes())
    _write_same_or_new(destination / "rom.nes", data)
    return destination.parent


def button_mapping(buttons: list[str | None]) -> tuple[np.ndarray, ...]:
    """Explicit balanced actions: no START/SELECT or simultaneous opposite directions."""
    needed = {button for action in BUTTON_ACTIONS for button in action}
    if not needed.issubset(buttons):
        raise ValueError("Emulator does not expose the required NES buttons")
    return tuple(np.array([int(button in action) for button in buttons], np.int8) for action in BUTTON_ACTIONS)


class MarioFrames(gym.Wrapper):
    """Repeat button actions, stop at a boundary, and preserve the latest true RGB.

    Counters count calls to the underlying one-frame ``step``. Initialization and
    reset may also advance emulator frames; they are not included in step_frames.
    """

    def __init__(self, env: gym.Env, config: MarioConfig):
        super().__init__(env)
        self.config = config
        source = env.observation_space
        if not isinstance(source, spaces.Box) or len(source.shape) != 3 or source.shape[-1] != 3 or source.dtype != np.uint8:
            raise ValueError("MarioFrames requires uint8 RGB observations")
        height, width = config.resize or source.shape[:2]
        shape = (height, width, 1 if config.grayscale else 3)
        self.observation_space = spaces.Box(0, 255, shape, np.uint8)
        self._rgb = None
        self._counts = {"step_frames": 0, "agent_decisions": 0, "last_step_frames": 0, "reset_calls": 0}

    def _convert(self, observation):
        self._rgb = np.asarray(observation, np.uint8).copy()
        result = self._rgb
        if self.config.resize is not None:
            height, width = self.config.resize
            rows = np.minimum(((np.arange(height) + .5) * result.shape[0] / height).astype(int), result.shape[0] - 1)
            cols = np.minimum(((np.arange(width) + .5) * result.shape[1] / width).astype(int), result.shape[1] - 1)
            result = result[rows[:, None], cols[None, :]]
        if self.config.grayscale:
            result = np.rint(result.astype(float) @ np.array([.299, .587, .114])).astype(np.uint8)[..., None]
        return result.copy()

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._counts["reset_calls"] += 1
        self._counts["last_step_frames"] = 0
        return self._convert(observation), info

    def step(self, action):
        reward_total = 0.0
        frames = 0
        for _ in range(self.config.frame_repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            reward_total += float(reward)
            frames += 1
            if terminated or truncated:
                break
        self._counts["step_frames"] += frames
        self._counts["agent_decisions"] += 1
        self._counts["last_step_frames"] = frames
        return self._convert(observation), reward_total, terminated, truncated, info

    def rgb_observation(self) -> np.ndarray:
        if self._rgb is None:
            raise RuntimeError("Reset before requesting RGB observation")
        return self._rgb.copy()

    def frame_counts(self) -> dict[str, int]:
        return self._counts.copy()


class MarioAdapter(ObservationActionAdapter):
    def __init__(self, env: gym.Env, config: MarioConfig):
        mapping = button_mapping(env.unwrapped.buttons)
        self._frames = MarioFrames(env, config)
        super().__init__(self._frames, observation_mode="raw", action_mapping=mapping)
        self.action_names = ACTION_NAMES
        self.config = config

    def rgb_observation(self) -> np.ndarray:
        """Unresized, ungrayscaled observed pixels; never RAM or evaluator info."""
        return self._frames.rgb_observation()

    def evaluator_metrics(self) -> dict[str, Any]:
        result = super().evaluator_metrics()
        result["frame_counts"] = self._frames.frame_counts()
        return result


def create_mario(rom_path: str | Path, config: MarioConfig | None = None) -> MarioAdapter:
    """Create RGB Mario after explicit local ROM validation; no pretrained model."""
    config = config or MarioConfig()
    root = prepare_mario(rom_path, config)
    retro = _retro()
    custom = retro.data.Integrations.CUSTOM_ONLY
    if str(root) not in custom.paths:
        retro.data.Integrations.add_custom_path(str(root))
    resolved = retro.data.get_romfile_path(GAME, custom)
    if Path(resolved).resolve() != (root / GAME / "rom.nes").resolve():
        raise RuntimeError("Another registered custom integration shadows the project-local Mario ROM")
    env = retro.make(GAME, state=config.state, inttype=custom,
                     use_restricted_actions=retro.Actions.ALL,
                     obs_type=retro.Observations.IMAGE, render_mode="rgb_array")
    try:
        return MarioAdapter(env, config)
    except Exception:
        env.close()
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Mario package and optional local ROM check")
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--rom", type=Path, help="Explicit local .nes file; never searched or downloaded")
    parser.add_argument("--state", default="Level1-1")
    args = parser.parse_args(argv)
    try:
        result = check_mario(args.rom, MarioConfig(state=args.state))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Mario check failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 2 if result["rom_matches"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
