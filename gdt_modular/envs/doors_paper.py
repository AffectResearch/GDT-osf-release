import numpy as np

class Doors:
    def __init__(self, outcome_space, p_list):
        self.p_list = p_list
        self.actions = len(p_list)
        self.feature_map = {
            "success": {"goal_dim": 1.0},
            "failure": {"goal_dim": 0.0},
            "start": {"goal_dim": -1.0}
        }
        self.current_outcome = "start"

        self.outcome_space = outcome_space
        self.current_state = None
        self.current_features = {}
        self.precision = 1.0

    def step(self, action):
        p_success = self.p_list[action-1]
        if np.random.random() < p_success:
            self.current_outcome = "success"
        else:
            self.current_outcome = "failure"
        return action
    
    def get_features(self):
        return self.feature_map[self.current_outcome]
    
    
    def transition_prob(self):
        return self.p_list
    
    def reset(self):
        self.current_state = 0
        self.current_features = {}
        return self.current_state