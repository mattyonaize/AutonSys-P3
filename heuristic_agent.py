import numpy as np


class HeuristicAgent:
    def act(self, observation):
        obs = np.asarray(observation)

        # Safety fallback: if the observation is not RAM-like, do nothing.
        if obs.ndim != 1 or len(obs) <= 55:
            return 0

        ball_x = obs[42]
        paddle_x = obs[55]

        if ball_x < paddle_x:
            return 4
        if ball_x > paddle_x:
            return 3
        return 0