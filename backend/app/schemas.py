from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetabolismConfigSchema(Schema):
    max_energy: float = Field(default=36, gt=0)
    max_fatigue: float = Field(default=12, gt=0)
    initial_energy_fraction: float = Field(default=1, gt=0, le=1)
    initial_fatigue_fraction: float = Field(default=0, ge=0, le=1)
    basal_energy_cost: float = Field(default=1, gt=0)
    sleep_energy_cost: float = Field(default=0.5, gt=0)
    awake_fatigue_gain: float = Field(default=0.75, gt=0)
    sleep_recovery: float = Field(default=5, gt=0)
    activity_fatigue_limit: float = Field(default=0.85, gt=0, le=1)


class RewardConfigSchema(Schema):
    alive: float = 0.1
    nutrition: float = 0.025
    death: float = -4
    invalid: float = -0.15


class SimulationConfig(Schema):
    width: int = Field(default=5, ge=1)
    height: int = Field(default=5, ge=1)
    max_episode_steps: int = Field(default=200, ge=1)
    resource_density: float = Field(default=0.32, ge=0, le=1)
    regeneration_probability: float = Field(default=0.25, ge=0, le=1)
    perception_radius: int = Field(default=1, ge=0)
    inventory_capacity: int = Field(default=3, ge=0)
    plant_nutrition: float = Field(default=14, ge=0)
    move_energy_cost: float = Field(default=0.75, ge=0)
    move_fatigue_gain: float = Field(default=0.75, ge=0)
    collect_energy_cost: float = Field(default=0.25, ge=0)
    collect_fatigue_gain: float = Field(default=0.25, ge=0)
    metabolism: MetabolismConfigSchema = Field(default_factory=MetabolismConfigSchema)
    rewards: RewardConfigSchema = Field(default_factory=RewardConfigSchema)


class Position(Schema):
    x: int
    y: int


class WorldDimensions(Schema):
    width: int
    height: int


class ResourceSnapshot(Schema):
    kind: str
    position: Position
    quantity: int


class AgentSnapshot(Schema):
    identifier: UUID
    position: Position
    energy: float
    max_energy: float
    fatigue: float
    max_fatigue: float
    inventory: int
    inventory_capacity: int
    current_action: str
    perception_radius: int
    age: int
    alive: bool


class EpisodeMetric(Schema):
    episode: int
    reward: float
    survival_duration: int
    mean_survival_duration: float
    food_consumption: int
    sleeping_frequency: float


class SimulationSnapshot(Schema):
    dimensions: WorldDimensions
    simulation_step: int
    episode: int
    status: Literal["running", "paused"]
    execution_mode: Literal["simulation", "evaluation", "training"] = "simulation"
    speed: Literal[1, 2, 5, 10]
    resources: tuple[ResourceSnapshot, ...]
    agents: tuple[AgentSnapshot, ...]
    cumulative_reward: float
    metrics: tuple[EpisodeMetric, ...]
    policy_name: str | None = None
    training_run: str | None = None
    training_timesteps: int | None = None


class StepRequest(Schema):
    action: int = Field(ge=0, le=7)


class ResetRequest(Schema):
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class SpeedRequest(Schema):
    speed: Literal[1, 2, 5, 10]


class PolicyRequest(Schema):
    model_name: str = Field(
        min_length=1, max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*\.zip$",
    )


class PolicyCatalog(Schema):
    configured: bool
    policies: tuple[str, ...] = ()


class TrainingRequest(Schema):
    total_steps: int = Field(default=150_000, ge=512, le=5_000_000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)


class OperationResult(Schema):
    snapshot: SimulationSnapshot
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
