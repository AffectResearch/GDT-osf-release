# ---- ENVIRONMENTs ----
# Simple Money world environment

class MoneyMDPEnv:
    # States are 0 to 5 (5 is terminal)
    # Actions: "go" (to the right), "stay"
    # Rewards: +1 for every go, 0 otherwise


    def __init__(self, max_state=5):
        self.state = 0
        self.max_state = max_state
        self.actions = ["go", "back", "stay"]

    def step(self, action): 
        prev = self.state
        if action == "go" and self.state < self.max_state:
            self.state += 1
        elif action == "back" and self.state > 0:
            self.state -= 1
        elif action == "stay":
            pass
        if self.state > prev:
            reward = 1
        elif self.state < prev:
            reward = -1
        else:
            reward = 0
        done = self.state == self.max_state
        return self.state, reward, done
    
    def reset(self):
        self.state = 0
        return self.state
    

def money_discrepancy(state, goal, env):
    return abs(state - goal)