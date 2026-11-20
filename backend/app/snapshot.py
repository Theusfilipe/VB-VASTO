from adaptive_ai.environments.rabbit_survival.environment import RabbitSurvivalEnv
from adaptive_ai.resources.plant_fiber import PlantFiber

from backend.app.schemas import (
    AgentSnapshot,
    EpisodeMetric,
    Position,
    ResourceSnapshot,
    SimulationSnapshot,
    WorldDimensions,
)


def snapshot_from_env(
    env: RabbitSurvivalEnv,
    *,
    episode: int,
    running: bool,
    speed: int,
    metrics: tuple[EpisodeMetric, ...] | list[EpisodeMetric],
    execution_mode: str,
    policy_name: str | None = None,
    training_run: str | None = None,
    training_timesteps: int | None = None,
    training_episodes: int | None = None,
    training_episode_limit: int | None = None,
) -> SimulationSnapshot:
    rabbit = env.rabbit
    agent = AgentSnapshot(
        identifier=rabbit.identifier,
        position=Position(x=rabbit.position[0], y=rabbit.position[1]),
        energy=rabbit.energy,
        max_energy=rabbit.max_energy,
        fatigue=rabbit.fatigue,
        max_fatigue=rabbit.max_fatigue,
        inventory=len(rabbit.inventory),
        inventory_capacity=rabbit.inventory_capacity,
        current_action=env.last_action,
        perception_radius=env.config.perception_radius,
        age=rabbit.age,
        alive=rabbit.alive,
    )
    resources = tuple(
        ResourceSnapshot(kind=resource.resource_type, position=Position(x=x, y=y),
                         quantity=resource.quantity)
        for (x, y), resource in sorted(env.world.plants.items())
        if isinstance(resource, PlantFiber)
    )
    return SimulationSnapshot(
        dimensions=WorldDimensions(width=env.config.width, height=env.config.height),
        simulation_step=rabbit.age,
        episode=episode,
        status="running" if running else "paused",
        execution_mode=execution_mode,
        speed=speed,
        resources=resources,
        agents=(agent,),
        cumulative_reward=float(env.cumulative_reward),
        metrics=tuple(metrics),
        policy_name=policy_name,
        training_run=training_run,
        training_timesteps=training_timesteps,
        training_episodes=training_episodes,
        training_episode_limit=training_episode_limit,
    )
