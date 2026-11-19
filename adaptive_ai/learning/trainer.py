import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env as check_sb3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

type SnapshotCallback = Callable[[Env, int, list[dict[str, Any]], bool], None]


class _RolloutOutputs(BaseCallback):
    """Write metrics and checkpoints only after a rollout's PPO update."""

    def __init__(self, output: Path, checkpoint_every: int) -> None:
        super().__init__()
        self.output = output
        self.checkpoint_every = checkpoint_every

    def _on_training_start(self) -> None:
        self.start_step = self.last_step = self.model.num_timesteps
        self.next_checkpoint = self.start_step + self.checkpoint_every

    def _on_rollout_start(self) -> None:
        self._write_update()

    def _on_training_end(self) -> None:
        self._write_update()

    def _on_step(self) -> bool:
        return True

    def _write_update(self) -> None:
        steps = self.model.num_timesteps
        if steps == self.last_step:
            return
        self.last_step = steps
        assert isinstance(self.model, PPO)
        self.model.dump_logs()
        if steps >= self.next_checkpoint:
            self.model.save(self.output / f"ppo_{steps}_steps.zip")
            intervals = (steps - self.start_step) // self.checkpoint_every
            self.next_checkpoint = self.start_step + (intervals + 1) * self.checkpoint_every


class _LiveSnapshotCallback(BaseCallback):
    def __init__(
        self, emit: SnapshotCallback, stop_file: Path, interval: float = 0.1,
    ) -> None:
        super().__init__()
        self.emit = emit
        self.stop_file = stop_file
        self.interval = interval
        self.last_emit = 0.0
        self.pending_infos: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.pending_infos.extend(info for info in infos if "episode" in info)
        now = time.monotonic()
        should_emit = now - self.last_emit >= self.interval
        stopping = should_emit and self.stop_file.exists()
        if should_emit:
            env = self.training_env.envs[0].unwrapped
            self.emit(env, self.num_timesteps, self.pending_infos, not stopping)
            self.pending_infos.clear()
            self.last_emit = now
        return not stopping

    def _on_training_end(self) -> None:
        delay = self.interval - (time.monotonic() - self.last_emit)
        if delay > 0:
            time.sleep(delay)
        env = self.training_env.envs[0].unwrapped
        self.emit(env, self.num_timesteps, self.pending_infos, False)
        self.pending_infos.clear()
        self.last_emit = time.monotonic()


def train_ppo(
    env: Env, output: Path, *, total_steps: int, seed: int,
    checkpoint_every: int, episode_metrics: tuple[str, ...] = (),
    resume: Path | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    stop_file: Path | None = None,
) -> Path:
    """Train on a numerical Gymnasium environment. The caller owns its lifecycle."""
    check_sb3(env, skip_render_check=True)
    monitored = Monitor(env, str(output), info_keywords=episode_metrics)
    try:
        if resume is None:
            model = PPO("MlpPolicy", monitored, seed=seed, device="cpu",
                        n_steps=512, batch_size=64, gamma=0.995, ent_coef=0.01)
        else:
            model = PPO.load(resume, env=monitored, device="cpu", seed=seed)
        logger = configure(str(output), ["stdout", "csv"])
        try:
            callbacks: list[BaseCallback] = [_RolloutOutputs(output, checkpoint_every)]
            if snapshot_callback is not None and stop_file is not None:
                callbacks.append(_LiveSnapshotCallback(snapshot_callback, stop_file))
            model.set_logger(logger)
            model.learn(
                total_timesteps=total_steps, reset_num_timesteps=resume is None,
                callback=CallbackList(callbacks), log_interval=None,
            )
            final = output / "ppo_final.zip"
            model.save(final)
            return final
        finally:
            logger.close()
    finally:
        if monitored.results_writer is not None:
            monitored.results_writer.close()
