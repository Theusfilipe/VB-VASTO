from dataclasses import dataclass, field
from uuid import UUID, uuid4

type Position = tuple[int, int]


@dataclass
class Agent:
    """Entity state only; policies and action execution live outside the entity."""

    position: Position
    identifier: UUID = field(default_factory=uuid4, init=False)
    age: int = field(default=0, init=False)
    alive: bool = field(default=True, init=False)
