import numpy as np

class Corridor:
    def __init__(self, length=6):
        self.length = length
        self.state = 1  # Start at s_s (position 1)
        self.actions = 2 # Action space: 1 = Left, 2 = Right

        self.feature_map = {
            1: {"goal_dim": 1.0},
            2: {"goal_dim": 2.0},
            3: {"goal_dim": 3.0},
            4: {"goal_dim": 4.0},
            5: {"goal_dim": 5.0},
            6: {"goal_dim": 6.0}
        }

    def step(self, action):
        if action == 2: # Right
            self.state = min(self.length, self.state + 1)
        elif action == 1: # Left
            self.state = max(1, self.state - 1)
            
        return self.state
    
    def get_features(self):
        # The agent only sees this vector, not the 'self.state' number
        return self.feature_map[self.state]
    
    def reset(self):
        self.state = 1
        return self.state