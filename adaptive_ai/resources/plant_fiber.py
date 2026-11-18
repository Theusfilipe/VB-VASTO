from typing import ClassVar

from adaptive_ai.core.agent import Position
from adaptive_ai.resources.resource import Resource


class PlantFiber(Resource):
    KIND: ClassVar[str] = "plant_fiber"

    def __init__(
        self, quantity: int = 1, nutritional_value: float = 14.0, position: Position | None = None,
    ) -> None:
        super().__init__(self.KIND, quantity, nutritional_value, position)
