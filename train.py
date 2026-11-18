import argparse
from pathlib import Path

import torch
from gymnasium.utils.env_checker import check_env

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalConfig, RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.run_config import read_config, write_config
from adaptive_ai.learning.trainer import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO rabbit survival policy.")
    parser.add_argument("--steps", type=int, default=150_000, help="Minimum additional timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Training seed in [0, 2**32)")
    parser.add_argument("--output", type=Path, default=Path("runs/rabbit"), help="New run directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON simulation configuration")
    source.add_argument("--resume", type=Path, help="Trusted PPO checkpoint (.zip)")
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--plot", action="store_true", help="Save a PNG of completed episodes")
    args = parser.parse_args()
    if args.steps < 1 or args.checkpoint_every < 1 or not 0 <= args.seed < 2**32:
        parser.error("steps and checkpoint interval must be positive, and seed must be in [0, 2**32)")
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume must point to an existing checkpoint file")
    config_path = args.resume.with_name("config.json") if args.resume else args.config
    try:
        config = read_config(config_path) if config_path else RabbitSurvivalConfig()
        args.output.mkdir(parents=True, exist_ok=False)
        write_config(args.output / "config.json", config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    torch.set_num_threads(1)
    # Gymnasium's determinism checker assumes the first transition cannot truncate.
    with RabbitSurvivalEnv() as reference:
        check_env(reference, skip_render_check=True)
    env = RabbitSurvivalEnv(config)
    try:
        model = train_ppo(
            env, args.output, total_steps=args.steps, seed=args.seed,
            checkpoint_every=args.checkpoint_every, episode_metrics=("food_eaten", "sleep_steps", "move_steps"),
            resume=args.resume,
        )
    finally:
        env.close()
    print(f"Saved model: {model}")

    if args.plot:
        from adaptive_ai.visualization.training import plot_training

        plot_path = args.output / "training.png"
        if plot_training(args.output / "monitor.csv", plot_path):
            print(f"Saved training plot: {plot_path}")
        else:
            print("No completed episodes to plot.")


if __name__ == "__main__":
    main()
