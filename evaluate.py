import argparse
from pathlib import Path
from statistics import mean

from stable_baselines3 import PPO

from adaptive_ai.environments.rabbit_survival import RabbitSurvivalEnv
from adaptive_ai.environments.rabbit_survival.run_config import read_config

METRICS = ("survival_steps", "food_eaten", "move_steps", "sleep_steps", "cumulative_reward")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trusted PPO checkpoint without training.")
    parser.add_argument("model", type=Path, help="Trusted PPO checkpoint (.zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="First index in a separate evaluation seed range")
    parser.add_argument("--render", action="store_true", help="Display a Matplotlib window")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and the seed index nonnegative")
    if not args.model.is_file():
        parser.error("model must point to an existing checkpoint file")
    try:
        config = read_config(args.model.with_name("config.json"))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    model = PPO.load(args.model, device="cpu")
    env = RabbitSurvivalEnv(config, render_mode="human" if args.render else None)
    results: list[dict[str, float]] = []
    try:
        for episode in range(args.episodes):
            seed = 2**32 + args.seed + episode
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(int(action.item()))
            results.append({metric: info[metric] for metric in METRICS})
            status = "starved" if terminated else "time limit"
            fields = ", ".join(f"{key}={value:.3g}" for key, value in results[-1].items())
            print(f"Episode {episode + 1}, seed {seed} ({status}): {fields}")
    finally:
        env.close()
    print("Means: " + ", ".join(f"{key}={mean(row[key] for row in results):.2f}" for key in METRICS))


if __name__ == "__main__":
    main()
