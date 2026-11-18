import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalConfig, RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.actions import Action
from adaptive_ai.mechanics.metabolism import MetabolismConfig
from adaptive_ai.resources.plant_fiber import PlantFiber


def test_collection_consumption_and_reward_are_separate() -> None:
    config = RabbitSurvivalConfig(width=1, height=1, resource_density=1,
                                  regeneration_probability=0, inventory_capacity=1,
                                  metabolism=MetabolismConfig(initial_energy_fraction=0.5))
    env = RabbitSurvivalEnv(config)
    env.reset(seed=5)
    energy = env.rabbit.energy
    obs, reward, _, _, info = env.step(Action.COLLECT)
    assert obs[2:4].tolist() == [1, 0]
    assert env.rabbit.energy == energy - config.metabolism.basal_energy_cost - config.collect_energy_cost
    assert info["food_eaten"] == 0
    assert reward == config.rewards.alive

    energy = env.rabbit.energy
    obs, reward, _, _, info = env.step(Action.EAT)
    assert obs[2:4].tolist() == [0, 0]
    assert info["food_eaten"] == 1
    assert info["energy_restored"] == config.plant_nutrition
    assert reward == pytest.approx(config.rewards.alive + config.rewards.nutrition * (env.rabbit.energy - energy))
    assert env.step(Action.EAT)[4]["invalid_action"]


def test_exhaustion_blocks_movement_and_collection_but_sleep_recovers() -> None:
    config = RabbitSurvivalConfig(width=2, height=1, resource_density=1,
                                  regeneration_probability=0,
                                  metabolism=MetabolismConfig(initial_fatigue_fraction=1))
    env = RabbitSurvivalEnv(config)
    env.reset(seed=5)
    x, y = env.rabbit.position
    move = Action.MOVE_RIGHT if x == 0 else Action.MOVE_LEFT
    assert env.step(move)[4]["invalid_action"]
    assert env.step(Action.COLLECT)[4]["invalid_action"]
    assert env.rabbit.position == (x, y)
    assert env.rabbit.fatigue == env.rabbit.max_fatigue
    assert not env.step(Action.SLEEP)[4]["invalid_action"]
    assert env.step(move)[4]["move_steps"] == 1
    assert env.rabbit.position == (1 - x, y)
    assert not env.step(Action.COLLECT)[4]["invalid_action"]


@pytest.mark.parametrize("action", [Action.MOVE_UP, -1, 8, 2.5, None])
def test_boundary_or_invalid_action_is_an_awake_noop(action: int) -> None:
    env = RabbitSurvivalEnv(RabbitSurvivalConfig(width=1, height=1))
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(action)
    assert env.rabbit.position == (0, 0)
    assert env.rabbit.age == 1
    assert env.rabbit.energy == env.rabbit.max_energy - env.config.metabolism.basal_energy_cost
    assert not terminated and not truncated
    assert info["invalid_action"]
    assert reward == pytest.approx(env.config.rewards.alive + env.config.rewards.invalid)


def test_starvation_and_time_limit_have_distinct_flags() -> None:
    starving = RabbitSurvivalEnv(RabbitSurvivalConfig(
        max_episode_steps=2, metabolism=MetabolismConfig(max_energy=2),
    ))
    starving.reset()
    assert starving.step(Action.IDLE)[2:4] == (False, False)
    obs, reward, terminated, truncated, _ = starving.step(Action.IDLE)
    assert obs[0] == 0 and not starving.rabbit.alive
    assert reward == starving.config.rewards.death
    assert terminated and not truncated
    with pytest.raises(ResetNeeded):
        starving.step(Action.SLEEP)

    healthy = RabbitSurvivalEnv(RabbitSurvivalConfig(max_episode_steps=1))
    healthy.reset()
    assert healthy.step(Action.IDLE)[2:4] == (False, True)


def test_observation_order_bounds_and_perception() -> None:
    env = RabbitSurvivalEnv(RabbitSurvivalConfig(resource_density=0))
    env.reset(seed=0)
    env.rabbit.position = (1, 2)
    env.world.plants[(1, 1)] = PlantFiber(position=(1, 1))
    observed = env.observer.observe(env.rabbit, env.world)
    assert env.observation_space.contains(observed)
    assert observed.tolist() == [1, 0, 0, 0, 0.25, 0.75, 0.5, 0.5, 1, 0, 0, 0]
    env.world.plants[(2, 3)] = PlantFiber(position=(2, 3))
    np.testing.assert_array_equal(env.observer.observe(env.rabbit, env.world), observed)


def test_seeded_spawn_and_regeneration() -> None:
    config = RabbitSurvivalConfig(width=3, height=3, resource_density=1, regeneration_probability=1)
    first, second = RabbitSurvivalEnv(config), RabbitSurvivalEnv(config)
    np.testing.assert_array_equal(first.reset(seed=15)[0], second.reset(seed=15)[0])
    assert first.rabbit.identifier != second.rabbit.identifier
    for action in (Action.COLLECT, Action.EAT, Action.MOVE_LEFT, Action.COLLECT, Action.SLEEP):
        a = first.step(action)
        b = second.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        assert first.world.plants == second.world.plants
        assert len(first.world.plants) == first.world.resource_target


def test_gymnasium_contract() -> None:
    check_env(RabbitSurvivalEnv(), skip_render_check=True)
