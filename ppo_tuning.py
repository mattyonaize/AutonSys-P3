from itertools import product
from pathlib import Path
from collections import Counter
from stable_baselines3 import PPO
from ppo_trainer import make_env
import json
import numpy as np
import numpy as np


def evaluate_model(model, env, n_eval_episodes=10):
    episode_rewards = []
    action_counter = Counter()

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=False)

            for a in np.array(action).flatten():
                action_counter[int(a)] += 1

            obs, reward, dones, infos = env.step(action)

            total_reward += float(np.mean(reward))
            done = bool(np.any(dones))

        episode_rewards.append(total_reward)

    mean_reward = float(np.mean(episode_rewards))

    total_actions = sum(action_counter.values())
    action_distribution = {
        action: count / total_actions
        for action, count in action_counter.items()
    }

    return {
        "mean_episode_reward": mean_reward,
        "action_distribution": action_distribution,
    }

def compute_tuning_score(eval_stats):
    mean_reward = eval_stats["mean_episode_reward"]
    action_distribution = eval_stats["action_distribution"]

    max_action_ratio = max(action_distribution.values())

    # Straf als één actie extreem vaak wordt gekozen
    action_collapse_penalty = max_action_ratio

    score = mean_reward - 0.1 * action_collapse_penalty

    return score

def run_experiment(config):
    train_env = make_env(
        seed=config["seed"],
        num_vec_envs=config["num_vec_envs"],
    )

    eval_env = make_env(
        seed=config["seed"] + 1000,
        num_vec_envs=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        verbose=0,
        tensorboard_log="logs/tensorboard_tuning",
    )

    model.learn(total_timesteps=config["total_timesteps"])

    score = evaluate_model(model, eval_env)

    train_env.close()
    eval_env.close()

    return model, score


def main():
    Path("models/tuning").mkdir(parents=True, exist_ok=True)
    Path("logs/tuning").mkdir(parents=True, exist_ok=True)

    search_space = {
        "learning_rate": [1e-4, 2.5e-4],
        "ent_coef": [0.02, 0.05, 0.1],
        "n_steps": [256, 512, 1024],
        "clip_range": [0.1, 0.2],
    }

    fixed_config = {
        "batch_size": 256,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_vec_envs": 4,
        "total_timesteps": 500_000,
        "seed": 42,
    }

    keys = list(search_space.keys())
    combinations = list(product(*search_space.values()))

    best_score = -float("inf")
    best_config = None

    all_results = []

    for run_id, values in enumerate(combinations):
        config = fixed_config.copy()
        config.update(dict(zip(keys, values)))

        print(f"\n=== Run {run_id + 1}/{len(combinations)} ===")
        print(config)

        model, score = run_experiment(config)

        result = {
            "run_id": run_id,
            "score": score,
            "config": config,
        }

        all_results.append(result)

        print(f"Score: {score:.4f}")

        if score > best_score:
            print("Nieuwe beste configuratie gevonden. Model wordt opgeslagen.")

            best_score = score
            best_config = config

            model.save("models/tuning/best_ppo_model")

            with open("models/tuning/best_hyperparameters.json", "w") as f:
                json.dump(
                    {
                        "best_score": best_score,
                        "best_config": best_config,
                    },
                    f,
                    indent=4,
                )
        else:
            print("Niet beter dan huidige beste model. Model wordt niet opgeslagen.")

        # Resultatenlog bijhouden zonder alle modellen op te slaan
        with open("logs/tuning/all_results.json", "w") as f:
            json.dump(all_results, f, indent=4)

    print("\nTuning klaar.")
    print(f"Beste score: {best_score:.4f}")
    print("Beste hyperparameters:")
    print(best_config)


if __name__ == "__main__":
    main()