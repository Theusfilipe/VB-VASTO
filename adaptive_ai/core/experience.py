from dataclasses import dataclass
from typing import Generic, TypeVar

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Experience(Generic[ObservationT, ActionT]):
    """One transition, retaining the distinction needed for bootstrapping."""

    observation: ObservationT
    action: ActionT
    reward: float
    next_observation: ObservationT
    terminated: bool
    truncated: bool
