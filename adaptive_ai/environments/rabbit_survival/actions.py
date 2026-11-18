from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

from adaptive_ai.agents.rabbit import Rabbit
from adaptive_ai.environments.rabbit_survival.world import World
from adaptive_ai.resources.plant_fiber import PlantFiber


class Action(IntEnum):
    IDLE = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    COLLECT = 5
    EAT = 6
    SLEEP = 7


MOVES = {
    Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0), Action.MOVE_RIGHT: (1, 0),
}


@dataclass(frozen=True)
class ActionOutcome:
    action: Action | None
    invalid: bool = False
    energy_cost: float = 0
    fatigue_gain: float = 0
    nutrition: float = 0


def execute(action: int, rabbit: Rabbit, world: World) -> ActionOutcome:
    """Apply position/inventory effects; the environment advances metabolism separately."""
    if not isinstance(action, Integral) or isinstance(action, bool) or not 0 <= action < len(Action):
        return ActionOutcome(None, invalid=True)
    selected = Action(action)
    cfg = world.config
    if selected in MOVES or selected == Action.COLLECT:
        if not rabbit.metabolism.can_be_active:
            return ActionOutcome(selected, invalid=True)
    if selected in MOVES:
        x, y = rabbit.position
        dx, dy = MOVES[selected]
        destination = (x + dx, y + dy)
        if not world.contains(destination):
            return ActionOutcome(selected, invalid=True)
        rabbit.position = destination
        return ActionOutcome(selected, energy_cost=cfg.move_energy_cost, fatigue_gain=cfg.move_fatigue_gain)
    if selected == Action.COLLECT:
        plant = world.plants.get(rabbit.position)
        if plant is None or not rabbit.inventory.add(plant):
            return ActionOutcome(selected, invalid=True)
        del world.plants[rabbit.position]
        return ActionOutcome(selected, energy_cost=cfg.collect_energy_cost, fatigue_gain=cfg.collect_fatigue_gain)
    if selected == Action.EAT:
        food = rabbit.inventory.consume(PlantFiber.KIND)
        if food is None:
            return ActionOutcome(selected, invalid=True)
        return ActionOutcome(selected, nutrition=food.nutritional_value)
    return ActionOutcome(selected)
