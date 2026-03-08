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
        self.last_action_p = 0

    def perceive(self):
        raw_features = self.env.get_features()
        if self.perception_filter:
            # This is the "Psychological Lens"
            self.current_features = self.perception_filter(raw_features)
        else:
            # Default behavior if no lens is provided
            self.current_features = {k: float(v) for k, v in raw_features.items()}
        
    def set_policy(self, policy_type="perfect"):
        if self.env.actions == 2:
            if policy_type == "perfect":
                self.beliefs = [0.0, 1.0]  # 100% Right (Action 2)
            elif policy_type == "good":
                self.beliefs = [0.2, 0.8]  # 80% Right
            elif policy_type == "random":
                self.beliefs = [0.5, 0.5]  # Coin flip
        else:
            # Fallback for single-action envs (like Dice)
            self.beliefs = [1.0]

    def decide(self):
        # Use established beliefs or fallback to env default
        probs = self.beliefs if len(self.beliefs) > 0 else self.env.transition_prob()

        if self.env.actions == 1:
            action = 1
            self.last_action_p = probs[0] if isinstance(probs, (list, np.ndarray)) else probs
        else:
            # Choose action based on the probability distribution
            action_idx = np.random.choice(len(probs), p=probs)
            action = action_idx + 1 # Actions are 1-indexed
            self.last_action_p = probs[action_idx]

        self.current_state = self.env.step(action)
        self.perceive()
        return self.current_state
    
    def get_affect(self):
        # We pass the subjective features and the expectancy of the action taken
        aff_comp, ad, aas = self.affect_model.aff_comp(self.current_features, self.last_action_p)
        return aff_comp, ad, aas