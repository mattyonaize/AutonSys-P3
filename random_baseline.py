import numpy as np

class RandomAgent:
    def __init__(self, n_actions: int = 6, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return self.rng.integers(self.n_actions)