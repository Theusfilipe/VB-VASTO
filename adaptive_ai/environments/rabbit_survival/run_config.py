import json
from dataclasses import asdict
from pathlib import Path

from adaptive_ai.environments.rabbit_survival.config import RabbitSurvivalConfig, RewardConfig
from adaptive_ai.mechanics.metabolism import MetabolismConfig


def read_config(path: Path) -> RabbitSurvivalConfig:
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("Simulation configuration must be a JSON object")
    values["metabolism"] = MetabolismConfig(**values.get("metabolism", {}))
    values["rewards"] = RewardConfig(**values.get("rewards", {}))
    return RabbitSurvivalConfig(**values)


def write_config(path: Path, config: RabbitSurvivalConfig) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(config), stream, indent=2)
        stream.write("\n")
