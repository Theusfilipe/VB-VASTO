from adaptive_ai.resources.resource import Resource


class Inventory:
    """A unit-capacity container; items retain their individual nutritional values."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Inventory capacity cannot be negative")
        self._capacity = capacity
        self._items: list[Resource] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def add(self, resource: Resource) -> bool:
        if len(self) + resource.quantity > self.capacity:
            return False
        unit = Resource(resource.resource_type, 1, resource.nutritional_value)
        self._items.extend([unit] * resource.quantity)
        return True

    def consume(self, resource_type: str) -> Resource | None:
        for index, item in enumerate(self._items):
            if item.resource_type == resource_type:
                return self._items.pop(index)
        return None
