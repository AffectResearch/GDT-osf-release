import numpy as np

class Agent:
    def __init__(self, env, affect_model, targets, perception_filter=None):
        self.env = env
        self.affect_model = affect_model
        self.targets = targets
        self.perception_filter = perception_filter
        self.beliefs = []

        self.current_state = 0
        self.current_features = {}

        self.actions = env.actions 
        #self.p = env.transition_prob()
        self.last_action_p = 0

    def perceive(self):
        raw_features = self.env.get_features()
        if self.perception_filter:
            # The filter handles the "Binary vs Gradual" logic externally
            self.current_features = self.perception_filter(raw_features)
            return
        if isinstance(raw_features, dict):
            # Just ensure the values inside are floats
            self.current_features = {k: float(v) for k, v in raw_features.items()}
        else:
            val = raw_features[0] if isinstance(raw_features, list) else raw_features
            self.current_features = {"feature value": float(val)}
        

    def decide(self):
        # If we have specific beliefs (Doors), use them. 
        # Otherwise (Dice), ask the environment.
        probs = self.beliefs if len(self.beliefs) > 0 else self.env.transition_prob()

        if self.actions == 1:
            action = 1
            self.last_action_p = probs
        else:
            door_indices = np.arange(len(probs))
            action_idx = np.random.choice(door_indices, p=probs) 
            action = action_idx + 1
            self.last_action_p = probs[action_idx]

        self.current_state = self.env.step(action, target_dict=self.targets)
        self.perceive()

        return self.current_state
    
    
    def get_affect(self):
        aff_comp, ad, aas = self.affect_model.aff_comp(self.current_features, self.last_action_p)
        return aff_comp, ad, aas


        