import argparse
from pathlib import Path
from typing import Any

import torch
from gymnasium import Env

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.run_config import read_config
from adaptive_ai.learning.trainer import train_ppo
from backend.app.schemas import EpisodeMetric
from backend.app.snapshot import snapshot_from_env

PREFIX = "ADAPTIVE_AI_SNAPSHOT:"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO and stream throttled simulation snapshots.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()

    config = read_config(args.output / "config.json")
    metrics: list[EpisodeMetric] = []
    episode_number = 1

    def emit(env: Env, timesteps: int, infos: list[dict[str, Any]], running: bool) -> None:
        nonlocal episode_number
        simulation = env.unwrapped
        for info in infos:
            completed = info.get("episode")
            if not completed:
                continue
            duration = int(completed["l"])
            prior = [metric.survival_duration for metric in metrics[-9:]]
            mean_duration = (sum(prior) + duration) / (len(prior) + 1)
            metrics.append(EpisodeMetric(
                episode=episode_number,
                reward=float(completed["r"]),
                survival_duration=duration,
                mean_survival_duration=mean_duration,
                food_consumption=int(completed["food_eaten"]),
                sleeping_frequency=(
                    float(completed["sleep_steps"] / duration) if duration else 0
                ),
            ))
            metrics[:] = metrics[-100:]
            episode_number += 1
        snapshot = snapshot_from_env(
            simulation,
            episode=episode_number,
            running=running,
            speed=1,
            metrics=metrics,
            execution_mode="training",
            training_run=args.output.name,
            training_timesteps=timesteps,
        )
        print(PREFIX + snapshot.model_dump_json(), flush=True)

    torch.set_num_threads(1)
    env = RabbitSurvivalEnv(config)
    try:
        env.reset(seed=args.seed)
        emit(env, 0, [], True)
        train_ppo(
            env, args.output, total_steps=args.steps, seed=args.seed,
            checkpoint_every=max(512, min(25_000, args.steps)),
            episode_metrics=("food_eaten", "sleep_steps", "move_steps"),
            snapshot_callback=emit,
            stop_file=args.stop_file,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
