from dataclasses import dataclass

from adaptive_ai.core.agent import Agent
from adaptive_ai.mechanics.inventory import Inventory
from adaptive_ai.mechanics.metabolism import Metabolism


@dataclass
class Rabbit(Agent):
    metabolism: Metabolism
    inventory: Inventory

    @property
    def energy(self) -> float:
        return self.metabolism.energy

    @property
    def max_energy(self) -> float:
        return self.metabolism.config.max_energy

    @property
    def fatigue(self) -> float:
        return self.metabolism.fatigue

    @property
    def max_fatigue(self) -> float:
        return self.metabolism.config.max_fatigue

    @property
    def inventory_capacity(self) -> int:
        return self.inventory.capacity
