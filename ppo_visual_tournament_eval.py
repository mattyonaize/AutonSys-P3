from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import imageio
import numpy as np
from pettingzoo.atari import warlords_v3

from ppo_visual_agent import VisualPPOAgent


ACTION_NAMES = {
    0: "noop",
    1: "fire",
    2: "up",
    3: "right",
    4: "left",
    5: "down",
}


class RandomAgent:
    def act(self, observation, action_space=None):
        if action_space is not None:
            return int(action_space.sample())
        return int(np.random.randint(6))


def run_tournament(
    ppo_model_path: str = "models/ppo_warlords_visual_third_0.zip",
    ppo_slot: int = 2,
    n_games: int = 20,
    video_dir: str | None = None,
    deterministic: bool = True,
):
    if video_dir is not None:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    wins = Counter()
    total_scores = defaultdict(float)
    ppo_actions = Counter()

    for game_id in range(n_games):
        ppo_agent = VisualPPOAgent(ppo_model_path, deterministic=deterministic)
        env = warlords_v3.env(obs_type="grayscale_image", render_mode="rgb_array")
        env.reset(seed=10_000 + game_id)
        ppo_agent.reset()

        if game_id == 0:
            print(f"Game {game_id + 1} agent order: {env.agents}")
            print("Make sure ppo_slot points to the same agent used during training.")

        agent_names = [f"Random_{i}" for i in range(len(env.agents))]
        agent_names[ppo_slot] = "VisualPPO"

        agent_instances = [RandomAgent() for _ in env.agents]
        agent_instances[ppo_slot] = ppo_agent

        agent_mapping = {agent: agent_instances[i] for i, agent in enumerate(env.agents)}
        name_mapping = {agent: agent_names[i] for i, agent in enumerate(env.agents)}

        game_scores = defaultdict(float)
        frames = []

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            name = name_mapping[agent]
            game_scores[name] += float(reward)
            total_scores[name] += float(reward)

            if termination or truncation:
                action = None
            else:
                agent_obj = agent_mapping[agent]
                if isinstance(agent_obj, RandomAgent):
                    action = agent_obj.act(observation, env.action_space(agent))
                else:
                    action = agent_obj.act(observation)
                    ppo_actions[int(action)] += 1

            env.step(action)

            if video_dir is not None:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        max_score = max(game_scores.values()) if game_scores else 0.0
        game_winners = [name for name, score in game_scores.items() if score == max_score]
        if len(game_winners) == 1:
            wins[game_winners[0]] += 1
        else:
            wins["TIE"] += 1

        env.close()

        if video_dir is not None and frames:
            imageio.mimsave(str(Path(video_dir) / f"visual_game_{game_id}.mp4"), frames, fps=15)

        print(f"Game {game_id + 1}: {dict(game_scores)} winner(s): {game_winners}")

    print("\nTotal scores:")
    for name, score in sorted(total_scores.items()):
        print(f"{name}: {score:.2f}")

    print("\nWins:")
    for name, count in wins.most_common():
        print(f"{name}: {count}")

    total_ppo_actions = sum(ppo_actions.values())
    print("\nVisual PPO action distribution:")
    if total_ppo_actions == 0:
        print("No PPO actions recorded. Check ppo_slot and agent order.")
    else:
        for action, count in sorted(ppo_actions.items()):
            print(f"{action} ({ACTION_NAMES.get(action, '?')}): {count / total_ppo_actions:.3f}")


if __name__ == "__main__":
    run_tournament()
