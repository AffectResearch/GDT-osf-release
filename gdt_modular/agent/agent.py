import numpy as np

class Agent:
    def __init__(self, env, affect_model):
        self.env = env
        self.affect_model = affect_model

        self.current_state = 0
        self.current_features = {}

        self.actions = env.actions 
        #self.p = env.transition_prob()
        self.last_action_p = 0
        

    def decide(self):
        probs = self.env.transition_prob()

        if self.actions == 1:
            action = 1
            self.last_action_p = probs
        else:
            action = np.random.randint(1, self.actions + 1)
            self.last_action_p = probs[action - 1]

        self.current_state = self.env.step(action)
        self.current_features = self.env.get_features()

        return self.current_state
    
    
    def get_affect(self):
        aff_comp, ad, aas = self.affect_model.aff_comp(self.current_features, self.last_action_p)
        return aff_comp, ad, aas


        