import numpy as np
from gymnasium.spaces import Box

from adaptive_ai.agents.rabbit import Rabbit
from adaptive_ai.core.environment import Observation
from adaptive_ai.environments.rabbit_survival.world import World


class LocalObservation:
    """State, current plant, left/right/top/bottom distances, then local plant bits."""

    def __init__(self, radius: int) -> None:
        self.offsets = tuple(
            (dx, dy) for dy in range(-radius, radius + 1)
            for dx in range(-radius + abs(dy), radius - abs(dy) + 1)
            if dx or dy
        )
        self.space = Box(0.0, 1.0, shape=(8 + len(self.offsets),), dtype=np.float32)

    def observe(self, rabbit: Rabbit, world: World) -> Observation:
        x, y = rabbit.position
        width, height = world.config.width - 1, world.config.height - 1
        return np.array([
            rabbit.energy / rabbit.max_energy,
            rabbit.fatigue / rabbit.max_fatigue,
            len(rabbit.inventory) / max(1, rabbit.inventory_capacity),
            rabbit.position in world.plants,
            x / max(1, width), (width - x) / max(1, width),
            y / max(1, height), (height - y) / max(1, height),
            *((x + dx, y + dy) in world.plants for dx, dy in self.offsets),
        ], dtype=np.float32)
