import hashlib
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rtga import mario


class PixelEnv(gym.Env):
    buttons = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

    def __init__(self, truncated=False, end_at=2):
        self.observation_space = spaces.Box(0, 255, (12, 16, 3), np.uint8)
        self.action_space = spaces.MultiBinary(9)
        self.steps = 0
        self.end_at = end_at
        self.truncated = truncated
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros((12, 16, 3), np.uint8), {"secret": 500}

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        done = self.steps == self.end_at
        return (np.full((12, 16, 3), self.steps, np.uint8), 10,
                done and not self.truncated, done and self.truncated, {"secret": 500})

    def close(self):
        self.closed = True


@pytest.mark.parametrize("truncated", [False, True])
def test_repeat_stops_at_final_observation_and_counts_actual_frames(truncated):
    env = PixelEnv(truncated=truncated)
    adapter = mario.MarioAdapter(env, mario.MarioConfig(frame_repeat=4))
    assert adapter.reset().shape == (12, 16, 3)
    result = adapter.step(2)
    assert env.steps == 2
    assert result.terminated == (not truncated) and result.truncated == truncated
    assert (result.observation == 2).all()
    assert not hasattr(result, "reward") and not hasattr(result, "info")
    metrics = adapter.evaluator_metrics()
    assert metrics["reward"] == 20
    assert metrics["frame_counts"] == {"step_frames": 2, "agent_decisions": 1, "last_step_frames": 2, "reset_calls": 1}
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step(0)
    adapter.reset()
    assert adapter.evaluator_metrics()["frame_counts"]["step_frames"] == 2


def test_preprocessing_retains_actual_rgb_and_declares_pixel_boundary():
    adapter = mario.MarioAdapter(PixelEnv(), mario.MarioConfig(resize=(6, 8), grayscale=True))
    observation = adapter.reset()
    assert observation.shape == (6, 8, 1) and observation.dtype == np.uint8
    assert adapter.state_dim is None
    rgb = adapter.rgb_observation()
    assert rgb.shape == (12, 16, 3)
    rgb[:] = 255
    assert not adapter.rgb_observation().any()
    assert adapter.observation_space.contains(adapter.step(3).observation)


def test_mapping_is_balanced_and_excludes_menu_and_opposite_buttons():
    buttons = PixelEnv.buttons
    mapping = mario.button_mapping(buttons)
    assert len(mapping) == len(mario.ACTION_NAMES) == 12
    assert "left_run_jump" in mario.ACTION_NAMES and "right_run_jump" in mario.ACTION_NAMES
    for action in mapping:
        assert action.shape == (9,)
        assert not (action[buttons.index("LEFT")] and action[buttons.index("RIGHT")])
        assert not action[buttons.index("START")] and not action[buttons.index("SELECT")]
    with pytest.raises(ValueError, match="buttons"):
        mario.button_mapping(["LEFT"])


@pytest.fixture
def fake_integration(tmp_path, monkeypatch):
    package = tmp_path / "package" / mario.GAME
    package.mkdir(parents=True)
    body = b"test fixture, not a game ROM"
    content = b"NES\x1a" + b"\x00" * 12 + body
    digest = hashlib.sha1(body).hexdigest()
    (package / "rom.sha").write_text(digest + "\n")
    for name in ["data.json", "scenario.json", "metadata.json"]:
        (package / name).write_text("{}")
    (package / "Level1-1.state").write_bytes(b"fake state metadata")
    rom = tmp_path / "supplied.nes"
    rom.write_bytes(content)
    project = tmp_path / "project"
    project.mkdir()
    integration = SimpleNamespace(STABLE="stable")
    retro = SimpleNamespace(data=SimpleNamespace(Integrations=integration,
                            get_file_path=lambda game, file, mode: str(package / file)))
    monkeypatch.setattr(mario, "_retro", lambda: retro)
    monkeypatch.setattr(mario, "PROJECT_ROOT", project)
    monkeypatch.setattr(mario.metadata, "version", lambda name: "test")
    return rom, package, project, digest


def test_hash_check_writes_nothing_then_prepare_stays_in_project(fake_integration):
    rom, package, project, digest = fake_integration
    report = mario.check_mario(rom)
    assert report["rom_matches"] and report["rom_body_sha1"] == digest
    assert list(project.iterdir()) == []
    assert not (package / "rom.nes").exists()
    root = mario.prepare_mario(rom)
    assert root.is_relative_to(project)
    assert (root / mario.GAME / "rom.nes").read_bytes() == rom.read_bytes()
    assert not (package / "rom.nes").exists()
    assert mario.prepare_mario(rom) == root


def test_mismatched_rom_and_outside_destination_never_written(fake_integration, tmp_path):
    rom, _package, project, _digest = fake_integration
    with pytest.raises(ValueError, match="inside this project"):
        mario.prepare_mario(rom, mario.MarioConfig(integration_root=tmp_path / "outside"))
    rom.write_bytes(rom.read_bytes() + b"different")
    assert mario.check_mario(rom)["rom_matches"] is False
    with pytest.raises(ValueError, match="does not match"):
        mario.prepare_mario(rom)
    assert list(project.iterdir()) == []


def test_existing_changed_integration_is_not_overwritten(fake_integration):
    rom, _package, _project, _digest = fake_integration
    root = mario.prepare_mario(rom)
    state = root / mario.GAME / "Level1-1.state"
    state.write_bytes(b"edited")
    with pytest.raises(ValueError, match="differs"):
        mario.prepare_mario(rom)
    assert state.read_bytes() == b"edited"


def test_metadata_check_without_rom_never_constructs_emulator(fake_integration):
    _rom, _package, project, _digest = fake_integration
    report = mario.check_mario()
    assert report["rom_supplied"] is False and report["rom_matches"] is None
    assert report["emulator_rollout_tested"] is False
    assert list(project.iterdir()) == []


def test_cli_check_only(fake_integration, capsys):
    rom, _package, project, _digest = fake_integration
    assert mario.main(["--check", "--rom", str(rom)]) == 0
    assert '"rom_matches": true' in capsys.readouterr().out
    assert list(project.iterdir()) == []


def test_loader_uses_only_verified_custom_dataset(fake_integration, monkeypatch):
    rom, package, _project, _digest = fake_integration
    paths = []
    custom = SimpleNamespace(paths=paths)
    integrations = SimpleNamespace(STABLE="stable", CUSTOM_ONLY=custom,
                                   add_custom_path=paths.append)
    made = []

    def make(game, **kwargs):
        made.append((game, kwargs))
        return PixelEnv()

    retro = SimpleNamespace(
        data=SimpleNamespace(Integrations=integrations,
                             get_file_path=lambda game, file, mode: str(package / file),
                             get_romfile_path=lambda game, mode: str(mario.Path(paths[0]) / game / "rom.nes")),
        Actions=SimpleNamespace(ALL="all"), Observations=SimpleNamespace(IMAGE="image"), make=make,
    )
    monkeypatch.setattr(mario, "_retro", lambda: retro)
    adapter = mario.create_mario(rom)
    assert adapter.reset().shape == (12, 16, 3)
    assert adapter.step(0).observation.shape == (12, 16, 3)
    game, kwargs = made[0]
    assert game == mario.GAME
    assert kwargs == {"state": "Level1-1", "inttype": custom, "use_restricted_actions": "all",
                      "obs_type": "image", "render_mode": "rgb_array"}
    assert not (package / "rom.nes").exists()
    adapter.close()
