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
        self.sides = sides if sides else np.random.default_rng(seed)
        self.state = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.actions = 1    # "throw"
        self.outcome_space = np.arange(1, self.sides + 1)


    def step(self, action):
        if self.state==0 and action==1:
            self.state = self.rng.choice(self.outcome_space)
        return self.state
    
    def get_features(self):
        if self.state == 0:
            return {"feature value": None}
        return {"feature value": self.state}
    
    def reset (self):
        self.state = 0
        return self.state
    
    def transition_prob(self):
        p = 1/self.sides
        return p
