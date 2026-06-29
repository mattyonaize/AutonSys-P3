from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Callable

import ale_py  # noqa: F401 - registers Atari/ALE
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.atari import warlords_v3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


ACTION_NAMES = {
    0: "noop",
    1: "fire",
    2: "up",
    3: "right",
    4: "left",
    5: "down",
}


class ActionHistogramCallback(BaseCallback):
    def __init__(self, print_freq: int = 50_000):
        super().__init__()
        self.print_freq = print_freq
        self.counts = np.zeros(6, dtype=np.int64)

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []), dtype=int).flatten()
        for action in actions:
            if 0 <= action < len(self.counts):
                self.counts[action] += 1

        if self.num_timesteps > 0 and self.num_timesteps % self.print_freq < max(len(actions), 1):
            total = self.counts.sum()
            if total > 0:
                dist = {
                    f"{i}:{ACTION_NAMES.get(i, '?')}": round(float(c / total), 3)
                    for i, c in enumerate(self.counts)
                }
                print(f"Action distribution @ {self.num_timesteps}: {dist}")
        return True


class WarlordsVisualSingleAgentEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper voor PettingZoo Atari Warlords.

    PPO neemt controle van een Warlords agent.
    Dit versie gebruikt beeldobservaties met frame stacking zodat PPO theoretisch
    de positie van het bal zou afkunnen lezen, samen met de snelheid vanuit visueel
    beweging in plaats van alleen van RAM bytes.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        controlled_agent: str = "third_0",
        opponent_policy: Callable | None = None,
        render_mode: str | None = None,
        frame_size: int = 84,
        stack_size: int = 4,
        terminal_reward_scale: float = 5.0,
        alive_reward: float = 0.001,
        no_op_penalty: float = 0.001,
        repeat_action_penalty: float = 0.0005,
        max_cycles: int = 4500,
        seed: int | None = None,
    ):
        super().__init__()
        self.controlled_agent = controlled_agent
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.terminal_reward_scale = terminal_reward_scale
        self.alive_reward = alive_reward
        self.no_op_penalty = no_op_penalty
        self.repeat_action_penalty = repeat_action_penalty
        self.max_cycles = max_cycles
        self._rng = np.random.default_rng(seed)

        self.env = self._make_pz_env()
        if controlled_agent not in self.env.possible_agents:
            raise ValueError(
                f"controlled_agent={controlled_agent!r} is invalid. "
                f"Possible agents: {self.env.possible_agents}"
            )

        self.action_space = self.env.action_space(controlled_agent)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, frame_size, frame_size),
            dtype=np.uint8,
        )

        self.valid_actions = [0, 1, 2, 3, 4, 5]
        self.action_space = spaces.Discrete(6)

        self.frames: deque[np.ndarray] = deque(maxlen=stack_size)
        self.last_action: int | None = None
        self.same_action_count = 0

    def _make_pz_env(self):
        try:
            return warlords_v3.env(
                obs_type="grayscale_image",
                render_mode=self.render_mode,
                max_cycles=self.max_cycles,
            )
        except TypeError:
            return warlords_v3.env(obs_type="grayscale_image", render_mode=self.render_mode)

    def _preprocess_frame(self, obs) -> np.ndarray:
        if obs is None:
            return np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)

        arr = np.asarray(obs)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        arr = arr.astype(np.uint8, copy=False)
        arr = cv2.resize(arr, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        return arr

    def _get_stacked_obs(self, obs) -> np.ndarray:
        frame = self._preprocess_frame(obs)

        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(frame.copy())
        else:
            self.frames.append(frame.copy())

        return np.stack(list(self.frames), axis=0).astype(np.uint8)

    def _shape_reward(self, raw_reward: float, action: int | None, terminal: bool) -> float:
        #reward = float(raw_reward)

        #if raw_reward != 0.0:
        #    reward *= self.terminal_reward_scale

        #if not terminal:
        #    reward += self.alive_reward 

        #if action == 0:
        #    reward -= self.no_op_penalty

        #if self.same_action_count >= 8:
        #    reward -= self.repeat_action_penalty * min(self.same_action_count / 8, 5)

        return float(raw_reward)

    def _sample_opponent_action(self, agent: str, obs):
        if self.opponent_policy is not None:
            return int(self.opponent_policy(agent, obs, self.env.action_space(agent)))
        return int(self.env.action_space(agent).sample())

    def _advance_to_controlled_agent(self):
        while self.env.agents and self.env.agent_selection != self.controlled_agent:
            agent = self.env.agent_selection
            obs, reward, termination, truncation, info = self.env.last()
            if termination or truncation:
                action = None
            else:
                action = self._sample_opponent_action(agent, obs)
            self.env.step(action)

        if not self.env.agents or self.controlled_agent not in self.env.agents:
            return self.observation_space.sample() * 0, 0.0, True, False, {}

        obs, reward, termination, truncation, info = self.env.last()
        return self._get_stacked_obs(obs), float(reward), bool(termination), bool(truncation), info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.frames.clear()
        self.last_action = None
        self.same_action_count = 0

        try:
            self.env.reset(seed=seed, options=options)
        except TypeError:
            self.env.reset(seed=seed)

        obs, reward, termination, truncation, info = self._advance_to_controlled_agent()
        return obs, info

    def step(self, action):
        action_idx = int(np.asarray(action).item())
        action = self.valid_actions[action_idx]

        if self.env.agents and self.env.agent_selection != self.controlled_agent:
            self._advance_to_controlled_agent()

        if not self.env.agents or self.controlled_agent not in self.env.agents:
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            return zero_obs, 0.0, True, False, {}

        obs, raw_reward, termination, truncation, info = self.env.last()
        if termination or truncation:
            self.env.step(None)
            reward = self._shape_reward(float(raw_reward), None, terminal=True)
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            return zero_obs, reward, True, bool(truncation), info

        if self.last_action == action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
        self.last_action = action

        self.env.step(action)
        next_obs, raw_reward, termination, truncation, info = self._advance_to_controlled_agent()
        terminal = bool(termination or truncation)
        reward = self._shape_reward(float(raw_reward), action, terminal=terminal)
        return next_obs, reward, bool(termination), bool(truncation), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_vec_env(
    controlled_agent: str,
    n_envs: int,
    seed: int,
    frame_size: int,
    stack_size: int,
    terminal_reward_scale: float,
    alive_reward: float,
    no_op_penalty: float,
    repeat_action_penalty: float,
    max_cycles: int,
):
    def make_one(rank: int):
        def _init():
            env = WarlordsVisualSingleAgentEnv(
                controlled_agent=controlled_agent,
                frame_size=frame_size,
                stack_size=stack_size,
                terminal_reward_scale=terminal_reward_scale,
                alive_reward=alive_reward,
                no_op_penalty=no_op_penalty,
                repeat_action_penalty=repeat_action_penalty,
                max_cycles=max_cycles,
                seed=seed + rank,
            )
            return Monitor(env)

        return _init

    env_fns = [make_one(i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def train(
    total_timesteps: int = 1_000_000,
    controlled_agent: str = "third_0",
    seed: int = 42,
    n_envs: int = 8,
    model_path: str = "models/ppo_warlords_visual_third_0",
    frame_size: int = 84,
    stack_size: int = 4,
    terminal_reward_scale: float = 1.0,
    alive_reward: float = 0.0,
    no_op_penalty: float = 0.0,
    repeat_action_penalty: float = 0.0,
    max_cycles: int = 4500,
    tensorboard_log: str | None = None,
):
    Path("models").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    os.environ.setdefault("ALE_ROMS", os.getcwd())

    train_env = make_vec_env(
        controlled_agent=controlled_agent,
        n_envs=n_envs,
        seed=seed,
        frame_size=frame_size,
        stack_size=stack_size,
        terminal_reward_scale=terminal_reward_scale,
        alive_reward=alive_reward,
        no_op_penalty=no_op_penalty,
        repeat_action_penalty=repeat_action_penalty,
        max_cycles=max_cycles,
    )

    eval_env = make_vec_env(
        controlled_agent=controlled_agent,
        n_envs=1,
        seed=seed + 10_000,
        frame_size=frame_size,
        stack_size=stack_size,
        terminal_reward_scale=terminal_reward_scale,
        alive_reward=0.0,
        no_op_penalty=0.0,
        repeat_action_penalty=0.0,
        max_cycles=max_cycles,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(250_000 // max(n_envs, 1), 1),
        save_path="checkpoints",
        name_prefix=f"ppo_warlords_visual_{controlled_agent}",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_visual_eval",
        log_path="logs/visual_eval",
        eval_freq=max(250_000 // max(n_envs, 1), 1),
        n_eval_episodes=20,
        deterministic=False,
        render=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, ActionHistogramCallback()],
    )
    model.save(model_path)

    metadata_path = Path(model_path).with_suffix(".metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "controlled_agent": controlled_agent,
                "obs_type": "grayscale_image",
                "frame_size": frame_size,
                "stack_size": stack_size,
                "terminal_reward_scale": terminal_reward_scale,
                "alive_reward": alive_reward,
                "no_op_penalty": no_op_penalty,
                "repeat_action_penalty": repeat_action_penalty,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "seed": seed,
                "policy": "CnnPolicy",
            },
            f,
            indent=4,
        )

    train_env.close()
    eval_env.close()
    print(f"Saved trained visual PPO model to: {model_path}.zip")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--controlled-agent", type=str, default="third_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--model-path", type=str, default="models/ppo_warlords_visual_third_0")
    parser.add_argument("--frame-size", type=int, default=84)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=4500)
    parser.add_argument("--terminal-reward-scale", type=float, default=5.0)
    parser.add_argument("--alive-reward", type=float, default=0.0)
    parser.add_argument("--no-op-penalty", type=float, default=0.05)
    parser.add_argument("--repeat-action-penalty", type=float, default=0.005)
    parser.add_argument("--tensorboard-log", type=str, default=None)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        controlled_agent=args.controlled_agent,
        seed=args.seed,
        n_envs=args.n_envs,
        model_path=args.model_path,
        frame_size=args.frame_size,
        stack_size=args.stack_size,
        terminal_reward_scale=args.terminal_reward_scale,
        alive_reward=args.alive_reward,
        no_op_penalty=args.no_op_penalty,
        repeat_action_penalty=args.repeat_action_penalty,
        max_cycles=args.max_cycles,
        tensorboard_log=args.tensorboard_log,
    )
