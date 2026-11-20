from dataclasses import replace
from typing import TYPE_CHECKING, Any

import gymnasium as gym
from numpy import uint8
from numpy.typing import NDArray

from adaptive_ai.agents.rabbit import Rabbit
from adaptive_ai.core.environment import Observation
from adaptive_ai.environments.rabbit_survival.actions import MOVES, Action, execute
from adaptive_ai.environments.rabbit_survival.config import RabbitSurvivalConfig
from adaptive_ai.environments.rabbit_survival.observations import LocalObservation
from adaptive_ai.environments.rabbit_survival.rewards import survival_reward
from adaptive_ai.environments.rabbit_survival.world import World
from adaptive_ai.mechanics.inventory import Inventory
from adaptive_ai.mechanics.metabolism import Metabolism, MetabolismConfig

if TYPE_CHECKING:
    from adaptive_ai.visualization.renderer import RabbitRenderer


class RabbitSurvivalEnv(gym.Env[Observation, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self, config: RabbitSurvivalConfig | None = None, render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unknown render mode: {render_mode}")
        self.config = config if config is not None else RabbitSurvivalConfig()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observer = LocalObservation(self.config.perception_radius)
        self.observation_space = self.observer.space
        self._renderer: RabbitRenderer | None = None
        self._needs_reset = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self.world = World(self.config, self.np_random)
        position = self.world.cells[int(self.np_random.integers(len(self.world.cells)))]
        self.rabbit = Rabbit(position, Metabolism(self.config.metabolism),
                             Inventory(self.config.inventory_capacity))
        self.cumulative_reward = 0.0
        self.food_eaten = self.move_steps = self.sleep_steps = 0
        self.last_action = "RESET"
        self._needs_reset = False
        if self.render_mode == "human":
            self.render()
        return self.observer.observe(self.rabbit, self.world), self._info()

    def update_metabolism(self, config: MetabolismConfig) -> Observation:
        """Change an initialized rabbit's rates without advancing or restarting its episode."""
        if not hasattr(self, "rabbit"):
            raise gym.error.ResetNeeded("Call reset() before updating the active rabbit")
        self.config = replace(self.config, metabolism=config)
        self.world.config = self.config
        metabolism = self.rabbit.metabolism
        metabolism.config = config
        metabolism.energy = min(metabolism.energy, config.max_energy)
        metabolism.fatigue = min(metabolism.fatigue, config.max_fatigue)
        return self.observer.observe(self.rabbit, self.world)

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before starting another episode")
        outcome = execute(action, self.rabbit, self.world)
        sleeping = outcome.action == Action.SLEEP
        self.last_action = outcome.action.name if outcome.action is not None else "INVALID"
        if not outcome.invalid:
            self.food_eaten += int(outcome.action == Action.EAT)
            self.move_steps += int(outcome.action in MOVES)
            self.sleep_steps += int(sleeping)
        previous_energy = self.rabbit.energy
        restored = self.rabbit.metabolism.advance(
            sleeping=sleeping, energy_cost=outcome.energy_cost,
            fatigue_gain=outcome.fatigue_gain, nutrition=outcome.nutrition,
        )
        self.rabbit.age += 1
        self.rabbit.alive = self.rabbit.energy > 0
        terminated = not self.rabbit.alive
        truncated = not terminated and self.rabbit.age >= self.config.max_episode_steps
        self._needs_reset = terminated or truncated
        self.world.regenerate()
        reward = survival_reward(
            self.config.rewards, alive=self.rabbit.alive,
            energy_gained=max(0.0, self.rabbit.energy - previous_energy), invalid=outcome.invalid,
        )
        self.cumulative_reward += reward
        if self.render_mode == "human":
            self.render()
        info = self._info() | {"invalid_action": outcome.invalid, "energy_restored": restored}
        return self.observer.observe(self.rabbit, self.world), reward, terminated, truncated, info

    def _info(self) -> dict[str, Any]:
        return {
            "survival_steps": self.rabbit.age, "food_eaten": self.food_eaten,
            "move_steps": self.move_steps, "sleep_steps": self.sleep_steps,
            "cumulative_reward": self.cumulative_reward,
        }

    def render(self) -> NDArray[uint8] | None:
        if self.render_mode is None or not hasattr(self, "world"):
            return None
        if self._renderer is None:
            from adaptive_ai.visualization.renderer import RabbitRenderer

            self._renderer = RabbitRenderer(self.render_mode, self.metadata["render_fps"])
        return self._renderer.draw(
            self.world, self.rabbit, self.last_action, self.cumulative_reward,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
