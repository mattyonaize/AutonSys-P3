class HeuristicAgent:

    def act(self, observation):

        ball_x = observation[42]
        paddle_x = observation[55]

        if ball_x < paddle_x:
            return 4

        if ball_x > paddle_x:
            return 3

        return 0