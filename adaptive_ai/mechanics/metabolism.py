from dataclasses import astuple, dataclass, field
from math import isfinite


@dataclass(frozen=True)
class MetabolismConfig:
    max_energy: float = 36.0
    max_fatigue: float = 12.0
    initial_energy_fraction: float = 1.0
    initial_fatigue_fraction: float = 0.0
    basal_energy_cost: float = 1.0
    sleep_energy_cost: float = 0.5
    awake_fatigue_gain: float = 0.75
    sleep_recovery: float = 5.0
    activity_fatigue_limit: float = 0.85

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in astuple(self)):
            raise ValueError("Metabolic parameters must be finite")
        if min(self.max_energy, self.max_fatigue, self.basal_energy_cost,
               self.sleep_energy_cost) <= 0 or self.sleep_energy_cost > self.basal_energy_cost:
            raise ValueError("Maxima and metabolic costs must be positive; sleep cannot cost more than waking")
        if not 0 < self.initial_energy_fraction <= 1 or not 0 <= self.initial_fatigue_fraction <= 1:
            raise ValueError("Invalid initial energy or fatigue fraction")
        if not 0 < self.activity_fatigue_limit <= 1:
            raise ValueError("Activity fatigue limit must be in (0, 1]")
        if min(self.awake_fatigue_gain, self.sleep_recovery) <= 0:
            raise ValueError("Fatigue gain and recovery must be positive")


@dataclass
class Metabolism:
    config: MetabolismConfig
    energy: float = field(init=False)
    fatigue: float = field(init=False)

    def __post_init__(self) -> None:
        self.energy = self.config.max_energy * self.config.initial_energy_fraction
        self.fatigue = self.config.max_fatigue * self.config.initial_fatigue_fraction

    @property
    def can_be_active(self) -> bool:
        return self.fatigue < self.config.max_fatigue * self.config.activity_fatigue_limit

    def advance(
        self, *, sleeping: bool = False, energy_cost: float = 0,
        fatigue_gain: float = 0, nutrition: float = 0,
    ) -> float:
        """Apply one tick and return nutritional energy used, excluding overflow."""
        cfg = self.config
        cost = cfg.sleep_energy_cost if sleeping else cfg.basal_energy_cost
        after_cost = self.energy - cost - energy_cost
        restored = min(nutrition, cfg.max_energy - after_cost)
        self.energy = max(0.0, after_cost + restored)
        self.fatigue = (
            max(0.0, self.fatigue - cfg.sleep_recovery) if sleeping else
            min(cfg.max_fatigue, self.fatigue + cfg.awake_fatigue_gain + fatigue_gain)
        )
        return restored
