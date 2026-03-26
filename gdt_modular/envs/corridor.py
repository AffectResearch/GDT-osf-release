import numpy as np

class Corridor:
    def __init__(self, seed=None, length=5, trap_prob=0.1):
        self.length = length    # total steps to reach s_e
        self.state = 0  # Start at s_s (position 0)
        self.trap_prob = trap_prob

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.actions = 1 # Action space: "Advance"

        self.trap_state_idx = length +1     # Make the Trap State scalable with the length of the corridor
        self.feature_map = {i: {"goal_dim": float(i)} for i in range(self.length + 1)}  # Feature = index for every state
        self.feature_map[self.trap_state_idx] = {"goal_dim": -3.0}   # Trap State Feature = -3.0 for max discrepancy

    def step(self, action):
        if self.state >= self.length:
            return self.state           # if at s_e or in trap we're stuck
        
        if self.state > 0:               # Traps start at s_a
            if self.rng.random() < self.trap_prob:
                self.state = self.trap_state_idx
                return self.state
        
        self.state += 1                 # successful advance
        return self.state

    
    def get_features(self):
        # The agent only sees this vector, not the 'self.state' number
        return self.feature_map[self.state]
    
    def is_terminal(self):
        # Terminal if at the end (length) or in the trap (length + 1)
        return self.state >= self.length
    
    def reset(self):
        self.state = 0
        return self.state
    
    def transition_prob(self, state=None):
        if state is None:
            state = self.state

        if state == self.length:
            return 1.0  # Goal state: Success is achieved/maintained
        if state < self.length:
            return 1.0 - self.trap_prob
        return 0.0      # Trap state: No way to reach the goal dim