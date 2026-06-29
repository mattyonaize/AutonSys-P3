from pathlib import Path
import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, model_path: str = "models/ppo_warlords_shared_policy.zip"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PPO model not found at {model_path}. Train it first with: "
                "python ppo_trainer.py --timesteps 100000"
            )
        self.model = PPO.load(str(model_path))

    def act(self, observation):
        obs = np.asarray(observation)
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)
