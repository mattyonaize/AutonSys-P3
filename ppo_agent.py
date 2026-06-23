from stable_baselines3 import PPO

class PPOAgent:

    def __init__(self, model_path):

        self.model = PPO.load(model_path)

    def act(self, observation):

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        return int(action)