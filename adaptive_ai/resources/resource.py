from dataclasses import dataclass
from math import isfinite

from adaptive_ai.core.agent import Position


@dataclass(frozen=True)
class Resource:
    resource_type: str
    quantity: int
    nutritional_value: float = 0
    position: Position | None = None

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise ValueError("Resource quantity must be positive")
        if not isfinite(self.nutritional_value) or self.nutritional_value < 0:
            raise ValueError("Nutritional value must be finite and nonnegative")
