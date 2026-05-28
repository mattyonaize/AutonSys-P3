
import numpy as np

class Agent1:
    def act(self, observation):
        # Return a random action (6 possible in ALE Warlords)
        return np.random.randint(6)
