import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalConfig, RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.actions import Action
from adaptive_ai.environments.rabbit_survival.config import RewardConfig
from adaptive_ai.mechanics.metabolism import MetabolismConfig
from adaptive_ai.resources.plant_fiber import PlantFiber
from gymnasium import spaces

from backend.app.schemas import (
    AgentSnapshot,
    EpisodeMetric,
    OperationResult,
    Position,
    ResourceSnapshot,
    SimulationSnapshot,
    WorldDimensions,
)

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

    async def start_runner(self) -> None:
        self._runner = asyncio.create_task(self._run(), name="simulation-runner")

    async def close(self) -> None:
        self._closed = True
        self.running = False
        self._wake.set()
        if self._runner is not None:
            await self._runner
        async with self._lock:
            await asyncio.to_thread(self.env.close)

    async def start(self) -> SimulationSnapshot:
        async with self._lock:
            if self.env._needs_reset:
                await asyncio.to_thread(self._reset)
            self.running = True
            self._wake.set()
            return await asyncio.to_thread(self._snapshot)

    async def pause(self) -> SimulationSnapshot:
        async with self._lock:
            self.running = False
            return await asyncio.to_thread(self._snapshot)

    async def reset(self, seed: int | None) -> SimulationSnapshot:
        async with self._lock:
            snapshot = await asyncio.to_thread(self._reset, seed)
            return snapshot

    async def set_speed(self, speed: int) -> SimulationSnapshot:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"Speed must be one of {ALLOWED_SPEEDS}")
        async with self._lock:
            self.speed = speed
            self._wake.set()
            return await asyncio.to_thread(self._snapshot)

    async def step(self, action: int | None = None) -> OperationResult:
        async with self._lock:
            if self.running:
                raise RuntimeError("Pause the simulation before stepping manually")
            return await asyncio.to_thread(self._advance, action)

    async def snapshot(self) -> SimulationSnapshot:
        async with self._lock:
            return await asyncio.to_thread(self._snapshot)

    async def configuration(self) -> dict[str, Any]:
        async with self._lock:
            return asdict(self.config)

    async def load_policy(self, model_path: Path) -> SimulationSnapshot:
        candidate = await asyncio.to_thread(self._prepare_policy, model_path)
        async with self._lock:
            return await asyncio.to_thread(self._install_policy, *candidate)

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

    def _reset(self, seed: int | None = None) -> SimulationSnapshot:
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
        self.metrics = self.metrics[-1000:]

    def _snapshot(self) -> SimulationSnapshot:
        rabbit = self.env.rabbit
        agent = AgentSnapshot(
            identifier=rabbit.identifier,
            position=Position(x=rabbit.position[0], y=rabbit.position[1]),
            energy=rabbit.energy,
            max_energy=rabbit.max_energy,
            fatigue=rabbit.fatigue,
            max_fatigue=rabbit.max_fatigue,
            inventory=len(rabbit.inventory),
            inventory_capacity=rabbit.inventory_capacity,
            current_action=self.env.last_action,
            perception_radius=self.config.perception_radius,
            age=rabbit.age,
            alive=rabbit.alive,
        )
        resources = [
            ResourceSnapshot(kind=resource.resource_type, position=Position(x=x, y=y),
                             quantity=resource.quantity)
            for (x, y), resource in sorted(self.env.world.plants.items())
            if isinstance(resource, PlantFiber)
        ]
        return SimulationSnapshot(
            dimensions=WorldDimensions(width=self.config.width, height=self.config.height),
            simulation_step=rabbit.age,
            episode=self.episode,
            status="running" if self.running else "paused",
            speed=self.speed,
            resources=tuple(resources),
            agents=(agent,),
            cumulative_reward=float(self.env.cumulative_reward),
            metrics=tuple(self.metrics),
            policy_name=self.policy_name,
        )

    def _prepare_policy(self, model_path: Path) -> tuple[RabbitSurvivalEnv, Any, Any, RabbitSurvivalConfig]:
        from stable_baselines3 import PPO

        with model_path.with_name("config.json").open(encoding="utf-8") as stream:
            config_data = json.load(stream)
        if not isinstance(config_data, dict):
            raise ValueError("The model's config.json must contain a JSON object")
        config_data["metabolism"] = MetabolismConfig(**config_data.get("metabolism", {}))
        config_data["rewards"] = RewardConfig(**config_data.get("rewards", {}))
        config = RabbitSurvivalConfig(**config_data)
        candidate_env = RabbitSurvivalEnv(config)
        observation, _ = candidate_env.reset(seed=0)
        try:
            model = PPO.load(model_path, device="cpu")
            if not spaces.utils.flatdim(model.observation_space) == spaces.utils.flatdim(candidate_env.observation_space):
                raise ValueError("Policy observation space does not match its saved simulation config")
            if model.action_space != candidate_env.action_space:
                raise ValueError("Policy action space does not match this simulation")
        except Exception:
            candidate_env.close()
            raise
        return candidate_env, observation, model, config

    def _install_policy(
        self, candidate_env: RabbitSurvivalEnv, observation: Any, model: Any,
        config: RabbitSurvivalConfig,
    ) -> SimulationSnapshot:
        self.env.close()
        self.env = candidate_env
        self.config = config
        self.observation = observation
        self.policy = model
        self.policy_name = model_path.name
        self.episode += 1
        self.metrics.clear()
        return self._snapshot()
