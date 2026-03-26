import numpy as np

class Agent:
    def __init__(self, env, affect_model, targets, perception_filter=None, expectancy_filter=None):
        self.env = env
        self.affect_model = affect_model
        self.targets = targets
        self.perception_filter = perception_filter
        self.expectancy_filter = expectancy_filter
        self.beliefs = []

        self.current_state = 0
        self.current_features = {}

        self.actions = env.actions 
        self.last_action_p = 0

    def perceive(self):
        # 1. Update Features (Psychological Lens for AD)
        raw_features = self.env.get_features()
        if self.perception_filter:
            self.current_features = self.perception_filter(raw_features)
        else:
            self.current_features = {k: float(v) for k, v in raw_features.items()}
        
        # 2. Update Beliefs (Subjective Expectancy Lens for AAS)
        if self.expectancy_filter:
            # Pass the env to the filter to allow state-based discounting
            self.beliefs = self.expectancy_filter(self.env)
        else:
            env_probs = self.env.transition_prob() 
            
            if isinstance(env_probs, (list, np.ndarray)):
                self.beliefs = env_probs
            else:
                # Fallback if transition_prob returns a single float
                self.beliefs = [env_probs]
        
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
        if self.env.is_terminal():
            self.last_action_p = 0.0 # No more actions possible
            return self.env.state
        
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

        if self.env.is_terminal():
            # If the environment has a transition_prob of 0 at terminal
            self.last_action_p = self.env.transition_prob() 
        else:
            #   Otherwise, it's the best possible expectancy from here
            self.last_action_p = max(self.beliefs) if isinstance(self.beliefs, list) else self.beliefs

        return self.current_state
    
    def get_affect(self):
        # Pass the subjective features and the expectancy of the action taken
        aff_comp, ad, aas = self.affect_model.aff_comp(self.current_features, self.last_action_p)
        return aff_comp, ad, aas