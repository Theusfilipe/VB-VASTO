from adaptive_ai.environments.rabbit_survival.config import RewardConfig


def survival_reward(config: RewardConfig, *, alive: bool, energy_gained: float, invalid: bool) -> float:
    return ((config.alive if alive else config.death)
            + config.nutrition * energy_gained + (config.invalid if invalid else 0.0))
