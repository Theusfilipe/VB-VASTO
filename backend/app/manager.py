import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalConfig, RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.actions import Action
from adaptive_ai.environments.rabbit_survival.run_config import read_config, write_config
from gymnasium.spaces.utils import flatdim

from backend.app.schemas import EpisodeMetric, OperationResult, SimulationSnapshot
from backend.app.snapshot import snapshot_from_env

logger = logging.getLogger(__name__)
SNAPSHOT_RATE = 10
ALLOWED_SPEEDS = (1, 2, 5, 10)


class SimulationManager:
    """Owns the one shared environment and serializes every environment operation."""

    def __init__(self) -> None:
        self.config = RabbitSurvivalConfig()
        self.env = RabbitSurvivalEnv(self.config)
        self.observation, _ = self.env.reset(seed=0)
        self.episode = 1
        self.speed: int = 1
        self.running = False
        self.policy: Any = None
        self.policy_name: str | None = None
        self.metrics: list[EpisodeMetric] = []
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._closed = False
        self._training_process: subprocess.Popen[bytes] | None = None
        self._training_task: asyncio.Task[None] | None = None
        self._training_snapshot: SimulationSnapshot | None = None
        self._training_stop_file: Path | None = None

    async def start_runner(self) -> None:
        self._runner = asyncio.create_task(self._run(), name="simulation-runner")

    async def close(self) -> None:
        self._closed = True
        self.running = False
        self._wake.set()
        if self._runner is not None:
            await self._runner
        await self.stop_live_training()
        async with self._lock:
            await asyncio.to_thread(self.env.close)

    async def start(self) -> SimulationSnapshot:
        async with self._lock:
            self._ensure_training_stopped()
            self._training_snapshot = None
            if self.env._needs_reset:
                await asyncio.to_thread(self._reset)
            self.running = True
            self._wake.set()
            return await asyncio.to_thread(self._snapshot)

    async def pause(self) -> SimulationSnapshot:
        async with self._lock:
            self._ensure_training_stopped()
            self._training_snapshot = None
            self.running = False
            return await asyncio.to_thread(self._snapshot)

    async def reset(self, seed: int | None) -> SimulationSnapshot:
        async with self._lock:
            self._ensure_training_stopped()
            snapshot = await asyncio.to_thread(self._reset, seed)
            return snapshot

    async def set_speed(self, speed: int) -> SimulationSnapshot:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"Speed must be one of {ALLOWED_SPEEDS}")
        async with self._lock:
            self._ensure_training_stopped()
            self._training_snapshot = None
            self.speed = speed
            self._wake.set()
            return await asyncio.to_thread(self._snapshot)

    async def step(self, action: int | None = None) -> OperationResult:
        async with self._lock:
            self._ensure_training_stopped()
            self._training_snapshot = None
            if self.running:
                raise RuntimeError("Pause the simulation before stepping manually")
            return await asyncio.to_thread(self._advance, action)

    async def snapshot(self) -> SimulationSnapshot:
        async with self._lock:
            if self._training_snapshot is not None:
                return self._training_snapshot
            return await asyncio.to_thread(self._snapshot)

    async def configuration(self) -> dict[str, Any]:
        async with self._lock:
            return asdict(self.config)

    async def load_policy(self, model_path: Path, model_name: str) -> SimulationSnapshot:
        candidate = await asyncio.to_thread(self._prepare_policy, model_path)
        async with self._lock:
            self._ensure_training_stopped()
            return await asyncio.to_thread(self._install_policy, *candidate, model_name)

    async def start_live_training(
        self, total_steps: int | None, seed: int, models_directory: Path, *,
        total_episodes: int | None = None, max_episode_steps: int | None = None,
    ) -> SimulationSnapshot:
        async with self._lock:
            self._ensure_training_stopped()
            if self.running:
                raise RuntimeError("Pause the simulation before starting live training")
            run_name = f"live-{uuid4().hex[:12]}"
            output = models_directory / run_name
            config = self.config if max_episode_steps is None else replace(
                self.config, max_episode_steps=max_episode_steps,
            )
            await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
            await asyncio.to_thread(write_config, output / "config.json", config)
            stop_file = output / "STOP"
            training_snapshot = self._snapshot().model_copy(update={
                "status": "running", "execution_mode": "training",
                "training_run": run_name, "training_timesteps": 0,
                "training_episodes": 0, "training_episode_limit": config.max_episode_steps,
            })
            repository_root = Path(__file__).resolve().parents[2]
            budget_args = ["--episodes", str(total_episodes)] if total_episodes is not None else [
                "--steps", str(total_steps if total_steps is not None else 150_000),
            ]
            process = await asyncio.to_thread(
                subprocess.Popen,
                [
                    sys.executable, "-m", "backend.app.live_training",
                    "--output", str(output), *budget_args, "--seed", str(seed),
                    "--stop-file", str(stop_file),
                ],
                cwd=repository_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self._training_stop_file = stop_file
            self._training_process = process
            self._training_snapshot = training_snapshot
            self._training_task = asyncio.create_task(
                self._collect_live_training(process, run_name),
                name=f"live-training-{run_name}",
            )
            return training_snapshot

    async def stop_live_training(self) -> SimulationSnapshot:
        async with self._lock:
            process = self._training_process
            task = self._training_task
            stop_file = self._training_stop_file
            if process is not None and process.poll() is None and stop_file is not None:
                await asyncio.to_thread(stop_file.touch, exist_ok=True)
        if task is not None:
            await task
        elif process is not None:
            await asyncio.to_thread(process.wait)
        return await self.snapshot()

    async def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._closed:
            if not self.running:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except TimeoutError:
                    pass
                next_tick = time.monotonic()
                continue
            next_tick += 1 / (SNAPSHOT_RATE * self.speed)
            async with self._lock:
                if self.running and not self._closed:
                    try:
                        await asyncio.to_thread(self._advance)
                    except Exception:
                        logger.exception("Simulation runner paused after an error")
                        self.running = False
            await asyncio.sleep(max(0, next_tick - time.monotonic()))

    async def _collect_live_training(
        self, process: subprocess.Popen[bytes], run_name: str,
    ) -> None:
        if process.stdout is None:
            raise RuntimeError("Live training worker stdout was not captured")
        prefix = "ADAPTIVE_AI_SNAPSHOT:"
        while line := await asyncio.to_thread(process.stdout.readline):
            decoded = line.decode(errors="replace").strip()
            if decoded.startswith(prefix):
                try:
                    snapshot = SimulationSnapshot.model_validate_json(decoded[len(prefix):])
                    # Keep controls locked until the worker has exited.
                    self._training_snapshot = snapshot.model_copy(update={"status": "running"})
                except ValueError:
                    logger.exception("Ignoring invalid live training snapshot")
            elif decoded:
                logger.info("Live training %s: %s", run_name, decoded)
        process.stdout.close()
        return_code = await asyncio.to_thread(process.wait)
        if return_code:
            logger.error("Live training process exited with status %d", return_code)
        if self._training_snapshot is not None:
            self._training_snapshot = self._training_snapshot.model_copy(update={
                "status": "paused",
                "training_error": (
                    f"Training worker exited with code {return_code}. Check the backend terminal."
                    if return_code else None
                ),
            })
        self._training_process = None

    def _ensure_training_stopped(self) -> None:
        if self._training_task is not None and not self._training_task.done():
            raise RuntimeError("Stop live training before controlling the simulation")

    def _reset(self, seed: int | None = None) -> SimulationSnapshot:
        self._training_snapshot = None
        self.observation, _ = self.env.reset(seed=seed)
        self.episode += 1
        return self._snapshot()

    def _advance(self, action: int | None = None) -> OperationResult:
        if self.env._needs_reset:
            raise RuntimeError("Episode has ended; reset the simulation before stepping")
        if action is None:
            if self.policy is None:
                action = int(Action.IDLE)
            else:
                predicted, _ = self.policy.predict(self.observation, deterministic=True)
                action = int(predicted.item() if hasattr(predicted, "item") else predicted)
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._record_metric(info)
        if (terminated or truncated) and self.running:
            self._reset()
        snapshot = self._snapshot()
        result = OperationResult(
            snapshot=snapshot, reward=reward, terminated=terminated, truncated=truncated,
        )
        return result

    def _record_metric(self, info: dict[str, Any]) -> None:
        duration = int(info["survival_steps"])
        previous = [metric.survival_duration for metric in self.metrics[-9:]]
        mean_duration = (sum(previous) + duration) / (len(previous) + 1)
        self.metrics.append(EpisodeMetric(
            episode=self.episode,
            reward=float(info["cumulative_reward"]),
            survival_duration=duration,
            mean_survival_duration=mean_duration,
            food_consumption=int(info["food_eaten"]),
            sleeping_frequency=float(info["sleep_steps"] / duration) if duration else 0,
        ))
        self.metrics = self.metrics[-100:]

    def _snapshot(self) -> SimulationSnapshot:
        return snapshot_from_env(
            self.env,
            episode=self.episode,
            running=self.running,
            speed=self.speed,
            metrics=self.metrics,
            execution_mode="evaluation" if self.policy is not None else "simulation",
            policy_name=self.policy_name,
        )

    def _prepare_policy(
        self, model_path: Path,
    ) -> tuple[RabbitSurvivalEnv, Any, Any, RabbitSurvivalConfig]:
        from stable_baselines3 import PPO

        config = read_config(model_path.with_name("config.json"))
        candidate_env = RabbitSurvivalEnv(config)
        observation, _ = candidate_env.reset(seed=0)
        try:
            model = PPO.load(model_path, device="cpu")
            if flatdim(model.observation_space) != flatdim(candidate_env.observation_space):
                raise ValueError("Policy observation space does not match its saved simulation config")
            if model.action_space != candidate_env.action_space:
                raise ValueError("Policy action space does not match this simulation")
        except Exception:
            candidate_env.close()
            raise
        return candidate_env, observation, model, config

    def _install_policy(
        self, candidate_env: RabbitSurvivalEnv, observation: Any, model: Any,
        config: RabbitSurvivalConfig, model_name: str,
    ) -> SimulationSnapshot:
        self.env.close()
        self.env = candidate_env
        self.config = config
        self.observation = observation
        self.policy = model
        self.policy_name = model_name
        self._training_snapshot = None
        self.episode += 1
        self.metrics.clear()
        return self._snapshot()
