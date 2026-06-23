from itertools import product
from pathlib import Path
from collections import Counter
from stable_baselines3 import PPO
from ppo_trainer import make_env
import json
import numpy as np

def evaluate_model(model, env, n_eval_episodes=10, max_steps=5000, deterministic=False):
    episode_rewards = []
    episode_lengths = []
    action_counter = Counter()

    for episode in range(n_eval_episodes):
        obs = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)

            for a in np.array(action).flatten():
                action_counter[int(a)] += 1

            obs, reward, dones, infos = env.step(action)

            # Gebruik sum in plaats van mean, anders verdwijnt reward snel in nullen
            total_reward += float(np.sum(reward))
            step_count += 1

            # Stop pas als alle vector-envs klaar zijn
            done = bool(np.all(dones))

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

    total_actions = sum(action_counter.values())

    action_distribution = {
        int(action): count / total_actions
        for action, count in action_counter.items()
    } if total_actions > 0 else {}

    return {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "episode_rewards": episode_rewards,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episode_lengths": episode_lengths,
        "action_distribution": action_distribution,
    }

def compute_tuning_score(eval_stats):
    mean_reward = eval_stats["mean_episode_reward"]
    action_distribution = eval_stats["action_distribution"]

    if len(action_distribution) == 0:
        return float(mean_reward)

    max_action_ratio = max(action_distribution.values())

    if max_action_ratio > 0.90:
        collapse_penalty = 0.1
    else:
        collapse_penalty = 0.0

    return float(mean_reward - collapse_penalty)

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
        verbose=1,
        tensorboard_log="logs/tensorboard_tuning",
    )

    model.learn(total_timesteps=config["total_timesteps"])

    eval_stats = evaluate_model(
        model,
        eval_env,
        n_eval_episodes=10
    )

    score = compute_tuning_score(eval_stats)

    train_env.close()
    eval_env.close()

    return model, score, eval_stats


def main():
    Path("models/tuning").mkdir(parents=True, exist_ok=True)
    Path("logs/tuning").mkdir(parents=True, exist_ok=True)

    search_space = {
        "learning_rate": [1e-4, 2.5e-4, 5e-4],
        "ent_coef": [0.0, 0.001, 0.005, 0.01],
        "n_steps": [512, 1024, 2048],
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

        model, score, eval_stats = run_experiment(config)

        result = {
            "run_id": run_id,
            "score": score,
            "mean_episode_reward": eval_stats["mean_episode_reward"],
            "action_distribution": eval_stats["action_distribution"],
            "config": config,
        }

        all_results.append(result)

        print(f"Score: {score:.4f}")
        print(f"Mean episode reward: {eval_stats['mean_episode_reward']:.4f}")
        print(f"Action distribution: {eval_stats['action_distribution']}")

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