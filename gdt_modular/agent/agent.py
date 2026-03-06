import numpy as np

class Agent:
    def __init__(self, env, affect_model):
        self.env = env
        self.affect_model = affect_model

        self.current_state = 0
        self.current_features = {}

        self.actions = env.actions 
        self.p = env.transition_prob()
        

    def decide(self):
        if self.actions == 1:
            action = 1
        else:
            action = np.random.randint(1, self.actions + 1)

        outcome_state = self.env.step(action)

        self.current_state = outcome_state
        self.current_features = self.env.get_features()

        return self.current_state
    
    def expectancy(self):
        e = self.p
        return e
    
    def get_affect(self):
        aff_comp, ad, aas = self.affect_model.aff_comp(self.current_features)
        return aff_comp, ad, aas


        