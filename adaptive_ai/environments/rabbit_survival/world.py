from math import ceil

from numpy.random import Generator

from adaptive_ai.core.agent import Position
from adaptive_ai.environments.rabbit_survival.config import RabbitSurvivalConfig
from adaptive_ai.resources.plant_fiber import PlantFiber


class World:
    def __init__(self, config: RabbitSurvivalConfig, rng: Generator) -> None:
        self.config = config
        self.rng = rng
        self.cells = tuple((x, y) for y in range(config.height) for x in range(config.width))
        self.resource_target = ceil(len(self.cells) * config.resource_density)
        indices = rng.choice(len(self.cells), size=self.resource_target, replace=False)
        self.plants = {self.cells[i]: self._plant(self.cells[i]) for i in indices}

    def contains(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _plant(self, position: Position) -> PlantFiber:
        return PlantFiber(nutritional_value=self.config.plant_nutrition, position=position)

    def regenerate(self) -> None:
        if (len(self.plants) < self.resource_target
                and self.rng.random() < self.config.regeneration_probability):
            empty = [cell for cell in self.cells if cell not in self.plants]
            position = empty[int(self.rng.integers(len(empty)))]
            self.plants[position] = self._plant(position)
