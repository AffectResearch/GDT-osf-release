
# ---- AFFECT MODEL ----
# Affect Model class, designed to be modular

class AffectModel:

    def __init__(self, w=1.0, B0=0.5, pi=1.0, max_discrepancy=None): # Maximum discrepancy: a cap so that discrepancy cannot grow infintely
            self.w = w          # Goal Value/Importance: Relative importance of a goal; goal_importance in Tomis model
            self.B0 = B0        # Affect at goal completion
            self.pi = pi        # Expectedness/Subjective Probability of a future/anticipated state; probability in Tomis Model
            self.max_discrepancy = max_discrepancy

    # 1. Calculation of discrepancy
    def discrepancy (self, state, goal):
        """
        Calculates the sum of the feature differences as discrepancy, 
        for those features in the goal that are not set to None (those are ignored, this allows for multiple goals later)
        a = state feature; TODO rename it to s? But s is salience...
        b = goal feature; TODO rename it to g
        Remark: No homeostatic goals yet, only positive achievement goals
        """
        d = sum(abs(a - b) if b is not None else 0 for a, b in zip(state, goal))
        if self.max_discrepancy is not None:
            d = min(d, self.max_discrepancy)
        return d


    # 2. Calculation of change in discrepancy

    def delta_discrepancy (self, prev_d, current_d):
        delta_d = prev_d - current_d
        # TODO: Right now I dropped the normalization of this delta_d that we had noted down in the original version of the code. 
        # it was dropped because it seemed not to fit anymore with the code that we had written (in my understanding, please correct me)
        return delta_d
        
    # 3. Calculation of the three types of Affect + a combined Affect

    # a) Affect from goal completion (baseline affect)
    def affect_from_completion(self, s0):
        return s0 * self.w * self.B0

    # b) Affect from Discrepancy
    def affect_discrepancy(self, d, s1):
        return s1 * self.w * (self.B0 + d)

    # c) Affect from Change in Discrepancy
    def affect_delta_discrepancy(self, delta_d, s2):
        return -s2 * self.w * (self.B0 + delta_d)

    # d) Calculation of Combined Affect
    def combined_affect(self, d, delta_d, s0, s1, s2):
        """
        Combines the three affect components and optionally applies subjective probability (pi).
        """
        a0 = self.affect_from_completion(s0)
        a_discr = self.affect_discrepancy(d, s1)
        a_deltadiscr = self.affect_delta_discrepancy(delta_d, s2)

        a_total = a0 - a_discr - a_deltadiscr

        if self.pi is not None:
            return a_total * self.pi
        return a_total


# ---- SALIENCE MANAGER ----
#class to manage salience. It makes sure salience is always normalized (i.e. cannot go above 1), it allows to retrieve salience and it allows to update salience
class SalienceManager:
    
    #Example dictionary. In this case I made sure they add up to 1, but they can be initialized with any value
    salience_manager = SalienceManager({
        "goal_completion": 0.3, # Salience for goal completion/baseline; salience_c in Tomis model
        "discrepancy": 0.4, # Salience for discrepancy; salience_d in Tomis model
        "delta_discrepancy": 0.3 # Salience for change in discrepancy; salience_delta_d in Tomis model
    }) 
    
    # initialize based on dictionary
    def __init__(self, initial_saliences):
        self.saliences = initial_saliences
        self.normalize()
    
    # Normalizaition, making sure it cannot go above 1
    def normalize(self):
        total = sum(self.saliences.values())
        if total == 0:
            return  # I don't know but it cannot become zero
        for k in self.saliences:
            self.saliences[k] /= total

    # Get function, allows to retrieve salience when called
    def get(self, k):
        return self.saliences.get(k)

    # Updates salience if necessary
    def update(self, k, new_value):
        self.saliences[k] = new_value
        self.normalize()


# ---- AGENT COGNITIVE ARCHITECTURE ---- TODO: this should also become a class at some point

# 1. Decide
def decide(actions, state, goal, prev_d, affect_model, salience_manager):
    # This simulates a very simple policy: the agent takes random actons when there is discrepancy otherwise it chooses to "stay"
    d = affect_model.discrepancy(state, goal)
    delta_d = affect_model.delta_discrepancy(prev_d, d)
    if d>0:
        action=random.choice(actions)
    else:
        action="stay"

    affect = affect_model.combined_affect(d=d, delta_d=delta_d, s0=salience_manager.get("goal_completion"), s1=salience_manager.get("discrepancy"), s2=salience_manager.get("delta_discrepancy"))
    
    return action, affect

# 2. Act
def do_action(action, state, salience_manager):   
    # simulates action execution; 5 actions possible: 'up', 'down', 'left', 'right', 'stay'
    # right now it needs states as a list of [x,y] TODO: make this more flexible
    # since we for now do not look into homeostatic goals i removed the energy variable, keeping it abstract. This includes removing the eat action
    newState=state.copy()
    if action == "up":
        newState[0]=newState[0]-1 #move up
    elif action == "down":
        newState[0]=newState[0]+1 #move down
    elif action == "left":
        newState[1] = newState[1]-1 #move left
    elif action == "right":
        newState[1]=newState[1]+1 #move right
    elif action == "stay":
        newState=newState

    # Possibility to update saliences after taking an action (here i based it on actions, but we can do that based on what we want)

    salience_updates = {
        "up": {"goal_completion": 0.05},
        "down": {"discrepancy": 0.02},
        "left": {"delta_discrepancy": -0.01},
        "right": {"goal_completion": -0.03, "discrepancy": 0.01},
        "stay": {"goal_completion": -0.8, "discrepancy": 0.9},
    }

    if action in salience_updates:
        for key, delta in salience_updates[action].items():
            new_salience = salience_manager.get(key) + delta
            salience_manager.update(key, new_salience)

    return newState

