from __future__ import annotations

import argparse
import os
from pathlib import Path

import ale_py  # noqa: F401
import supersuit as ss
from pettingzoo.atari import warlords_v3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


DEFAULT_MODEL_PATH = "models/ppo_warlords_shared_policy"


def make_env(seed: int = 42, num_vec_envs: int = 4):
    """Create an SB3-compatible vectorized Warlords environment."""
    set_random_seed(seed)

    env = warlords_v3.parallel_env(obs_type="ram", render_mode=None)

    env = ss.black_death_v3(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    return env


def train(
    total_timesteps: int = 100_000,
    seed: int = 42,
    num_vec_envs: int = 4,
    model_path: str = DEFAULT_MODEL_PATH,
):
    Path("models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # If the ROM file is stored in the project root, this helps ALE find it.
    os.environ.setdefault("ALE_ROMS", os.getcwd())

    env = make_env(seed=seed, num_vec_envs=num_vec_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 10_000),
        save_path="checkpoints",
        name_prefix="ppo_warlords",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="logs/tensorboard",
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(model_path)
    env.close()
    print(f"Saved trained PPO model to: {model_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        num_vec_envs=args.num_vec_envs,
        model_path=args.model_path,
    )
