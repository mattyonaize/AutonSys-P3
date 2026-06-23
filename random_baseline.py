import numpy as np

class RandomAgent:
    def act(self, observation):
        return np.random.randint(6)