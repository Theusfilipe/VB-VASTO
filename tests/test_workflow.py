import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalConfig, RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.actions import Action
from adaptive_ai.environments.rabbit_survival.run_config import read_config, write_config

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_config_fails_at_load(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"perception_radius": 1.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be integers"):
        read_config(config_path)
    config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        read_config(config_path)


def test_rgb_array_is_optional_and_tracks_state() -> None:
    pytest.importorskip("matplotlib")
    env = RabbitSurvivalEnv(render_mode="rgb_array")
    assert env.render() is None
    try:
        env.reset(seed=0)
        first = env.render()
        env.step(Action.SLEEP)
        second = env.render()
        assert first is not None and second is not None
        assert first.shape == second.shape and first.shape[-1] == 3
        assert first.dtype == np.uint8 and not np.array_equal(first, second)
    finally:
        env.close()


def test_train_evaluate_and_resume(tmp_path: Path) -> None:
    sb3 = pytest.importorskip("stable_baselines3")
    pytest.importorskip("matplotlib")

    def run(script: str, *args: str | Path) -> str:
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            text=True, capture_output=True, timeout=60, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    config = RabbitSurvivalConfig(max_episode_steps=1, perception_radius=2)
    source = tmp_path / "custom.json"
    write_config(source, config)
    trained = tmp_path / "trained"
    run("train.py", "--steps", "512", "--output", trained,
        "--checkpoint-every", "256", "--config", source, "--plot")
    assert read_config(trained / "config.json") == config
    assert (trained / "training.png").stat().st_size > 0
    with (trained / "progress.csv").open(newline="", encoding="utf-8") as stream:
        progress = next(csv.DictReader(stream))
        assert int(progress["time/total_timesteps"]) == 512
        assert float(progress["train/approx_kl"]) >= 0
    with (trained / "monitor.csv").open(newline="", encoding="utf-8") as stream:
        stream.readline()
        row = next(csv.DictReader(stream))
        assert int(row["l"]) == 1
        assert {"food_eaten", "move_steps", "sleep_steps"} <= row.keys()

    checkpoint = trained / "ppo_512_steps.zip"
    assert sb3.PPO.load(checkpoint, device="cpu")._n_updates > 0
    before = checkpoint.read_bytes()
    evaluation = run("evaluate.py", checkpoint, "--episodes", "2")
    assert "4294967296" in evaluation and "4294967297" in evaluation
    assert "survival_steps=1" in evaluation
    assert checkpoint.read_bytes() == before

    resumed = tmp_path / "resumed"
    run("train.py", "--steps", "512", "--output", resumed, "--resume", checkpoint)
    assert read_config(resumed / "config.json") == config
    assert sb3.PPO.load(resumed / "ppo_final.zip", device="cpu").num_timesteps == 1024
