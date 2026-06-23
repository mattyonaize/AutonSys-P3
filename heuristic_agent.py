class SimpleMovingAgent:
    def __init__(self):
        self.last_action = 1

    def act(self, observation):
        # Wissel simpelweg tussen twee bewegingsacties
        self.last_action = 2 if self.last_action == 3 else 3
        return self.last_action