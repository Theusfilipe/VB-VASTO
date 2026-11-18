from dataclasses import astuple, dataclass, field
from math import isfinite

from adaptive_ai.mechanics.metabolism import MetabolismConfig


@dataclass(frozen=True)
class RewardConfig:
    alive: float = 0.1
    nutrition: float = 0.025
    death: float = -4.0
    invalid: float = -0.15

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in astuple(self)):
            raise ValueError("Reward coefficients must be finite")


@dataclass(frozen=True)
class RabbitSurvivalConfig:
    width: int = 5
    height: int = 5
    max_episode_steps: int = 200
    resource_density: float = 0.32
    regeneration_probability: float = 0.25
    perception_radius: int = 1
    inventory_capacity: int = 3
    plant_nutrition: float = 14.0
    move_energy_cost: float = 0.75
    move_fatigue_gain: float = 0.75
    collect_energy_cost: float = 0.25
    collect_fatigue_gain: float = 0.25
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        counts = (self.width, self.height, self.max_episode_steps,
                  self.perception_radius, self.inventory_capacity)
        if any(type(value) is not int for value in counts):
            raise ValueError("World dimensions, episode duration, perception radius, and capacity must be integers")
        if min(self.width, self.height, self.max_episode_steps) < 1:
            raise ValueError("World dimensions and episode duration must be positive")
        if min(self.perception_radius, self.inventory_capacity) < 0:
            raise ValueError("Perception radius and inventory capacity cannot be negative")
        if not 0 <= self.resource_density <= 1 or not 0 <= self.regeneration_probability <= 1:
            raise ValueError("Resource density and regeneration probability must be in [0, 1]")
        costs = (self.plant_nutrition, self.move_energy_cost, self.move_fatigue_gain,
                 self.collect_energy_cost, self.collect_fatigue_gain)
        if not all(isfinite(value) and value >= 0 for value in costs):
            raise ValueError("Action costs, fatigue gains, and nutrition must be finite and nonnegative")
