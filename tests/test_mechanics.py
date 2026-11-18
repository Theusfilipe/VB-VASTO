import pytest

from adaptive_ai.mechanics.inventory import Inventory
from adaptive_ai.mechanics.metabolism import Metabolism, MetabolismConfig
from adaptive_ai.resources.plant_fiber import PlantFiber
from adaptive_ai.resources.resource import Resource


def test_energy_and_fatigue_are_independent() -> None:
    cfg = MetabolismConfig(initial_energy_fraction=0.5)
    state = Metabolism(cfg)
    state.advance(energy_cost=2, fatigue_gain=3)
    assert state.energy == pytest.approx(cfg.max_energy / 2 - cfg.basal_energy_cost - 2)
    assert state.fatigue == pytest.approx(cfg.awake_fatigue_gain + 3)

    fatigue = state.fatigue
    restored = state.advance(nutrition=100)
    assert restored < 100
    assert state.energy == cfg.max_energy
    assert state.fatigue == fatigue + cfg.awake_fatigue_gain

    state.advance(sleeping=True)
    assert state.energy == cfg.max_energy - cfg.sleep_energy_cost
    assert state.fatigue == max(0, fatigue + cfg.awake_fatigue_gain - cfg.sleep_recovery)
    state.advance(sleeping=True)
    assert state.fatigue == 0


def test_inventory_uses_unit_capacity_and_preserves_nutrition() -> None:
    inventory = Inventory(2)
    assert inventory.add(PlantFiber(2, 8, (1, 2)))
    assert not inventory.add(PlantFiber())
    assert inventory.consume("unknown") is None
    assert inventory.consume(PlantFiber.KIND) == Resource(PlantFiber.KIND, 1, 8)
    assert len(inventory) == 1
    assert inventory.add(Resource("stone", 1))
    assert len(inventory) == 2
