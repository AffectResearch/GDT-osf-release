import numpy as np

class Corridor:
    def __init__(self, seed=None length=5, trap_prob=0.1):
        self.length = length    # total steps to reach s_e
        self.state = 0  # Start at s_s (position 0)
        self.trap_prob = self.trap_prob

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.actions = 1 # Action space: "Advance"

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