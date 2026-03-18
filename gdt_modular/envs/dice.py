# ---- ENVIRONMENTs ----
# The agent can interact with 3 environments: Dice, Doors, CorriDOOR

# -- DICE --
# In the Dice environment, the agent can throw a dice and will receive a number back. The target is always f = 5. Goal Importance is 1. 
# The objective of this environment is to test 
import numpy as np

class Dice:
    # 7 states are possible, with s being the starting state (before throwing) and a-f are the outcome states (Dice showing 1-6)
    # 1 action is possible (throw)

    def __init__(self, seed=None, sides=6):
        self.sides = sides if sides is not None else 6
        self.state = 0 # Starting state
        self.feature_map = {i: {"goal_dim": float(i - 1) if i > 1 else 0.0} for i in range(self.sides + 2)}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.actions = 1    # "throw"
        self.outcome_space = np.arange(2, self.sides + 2)


    def step(self, action=None):
        if self.state == 0:
            self.state = 1  # s_start -> s_throw
        elif self.state == 1:
            self.state = self.rng.integers(2, self.sides + 2)   # s_throw -> roll the die (outcome states 2 through 7)
        else:
            self.state = 1  # s_outcome -> s_throw
        return self.state
    
    def get_features(self):
        # Always returns the raw vector associated with the state
        return self.feature_map[self.state]
    
    def reset (self):
        self.state = 0
        return self.state
    
    def transition_prob(self): # TODO: add transition=1 for s_start to s_throw
        p = 1/self.sides
        return p
