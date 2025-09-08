from collections import defaultdict
import random
import time


# ---- AFFECT MODEL ----
# Affect Model class, designed to be modular

class AffectModel:

    def __init__(self, w=1.0, B0=0.5, pi=1.0, max_discrepancy=None): # Maximum discrepancy: a cap so that discrepancy cannot grow infintely; TODO: Feed this in from the environment?
            self.w = w          # Goal Value/Importance: Relative importance of a goal; goal_importance in Tomis model; TODO: Not initialize it here
            self.B0 = B0        # Baseline Affect
            #self.pi = pi        # Expectedness/Subjective Probability of a future/anticipated state; probability in Tomis Model; remove pi?
            self.max_discrepancy = max_discrepancy

    # 1. Calculation of discrepancy
    def discrepancy (self, state, goal):
        """
        Calculates the sum of the feature differences as discrepancy, 
        for those features in the goal that are not set to None (those are ignored, this allows for multiple goals later)
        As for now there are no environments with features yet
        a = state feature; TODO rename it to s? But s is salience...
        b = goal feature; TODO rename it to g
        Remark: No homeostatic goals yet, only positive achievement goals
        """
        if not isinstance(state, (list, tuple)):
            state = [state]
        if not isinstance(goal, (list, tuple)):
            goal = [goal]
        d = sum(abs(a - b) if b is not None else 0 for a, b in zip(state, goal)) # TODO: Discuss distance function
        if self.max_discrepancy is not None:
            d = min(d, self.max_discrepancy)
        return d


    # 2. Calculation of change in discrepancy

    def delta_discrepancy (self, prev_d, current_d): # Where do they come from? maybe store them at each time step
        delta_d = prev_d - current_d # add current_d = d ; TODO: Store discrepancy so we can calculate prev_d and delta_d
        # TODO: Right now I dropped the normalization of this delta_d that we had noted down in the original version of the code. 
        # it was dropped because it seemed not to fit anymore with the code that we had written (in my understanding, please correct me)
        return delta_d
        
    # 3. Calculation of the three types of Affect + a combined Affect

    # a) Affect from goal completion (baseline affect)
    def affect_from_completion(self, s0):
        return s0 * self.w * self.B0 # insert an if clause if goal is not achieved

    # b) Affect from Discrepancy
    def affect_discrepancy(self, d, s1):
        return -s1 * self.w * d 

    # c) Affect from Change in Discrepancy
    def affect_delta_discrepancy(self, delta_d, s2):
        return s2 * self.w * delta_d

    # d) Calculation of Combined Affect
    def combined_affect(self, d, delta_d, s0, s1, s2):
        """
        Combines the three affect components and optionally applies subjective probability (pi).
        """
        a0 = self.affect_from_completion(s0)
        a_discr = self.affect_discrepancy(d, s1)
        a_deltadiscr = self.affect_delta_discrepancy(delta_d, s2)

        a_total = a0 + a_discr + a_deltadiscr

        # if self.pi is not None:
        #     return a_total * self.pi
        # return a_total
        # this needs to be in the planning function 


# ---- SALIENCE MANAGER ----
#class to manage salience. It makes sure salience is always normalized (i.e. cannot go above 1), it allows to retrieve salience (for explanability reasons) and it allows to update salience
class SalienceManager:
    
    #Example dictionary. In this case I made sure they add up to 1, but they can be initialized with any value. 
    # salience_manager = SalienceManager({
    #     "goal_completion": 0.3, # Salience for goal completion/baseline; salience_c in Tomis model
    #     "discrepancy": 0.4, # Salience for discrepancy; salience_d in Tomis model
    #     "delta_discrepancy": 0.3 # Salience for change in discrepancy; salience_delta_d in Tomis model
    # }) 
    
    # initialize based on dictionary
    def __init__(self, initial_saliences): # initial_saliences are the once fed in through the agent architecture (or the dictionary above)
        self.saliences = initial_saliences
        self.normalize()
    
    # Normalizaition, making sure it cannot go above 1
    def normalize(self):
        total = sum(self.saliences.values())
        if total == 0:
            return  # I don't know but it cannot become zero
        for k in self.saliences: # k being the salience of each affect component; taken from Tomis Code
            self.saliences[k] /= total

    # Get function to retrieve salience
    def retrieve(self, k):
        return self.saliences.get(k)

    # Updates salience if necessary
    def update(self, k, new_value):
        self.saliences[k] = new_value
        self.normalize()


# ---- AGENT DECISION MAKING ---- 

class DecisionMaking:
    def __init__(self, affect_model, salience_manager, memory, planning): # the features fed in there have to be initialized correctly in the agent class below
        self.affect_model = affect_model
        self.salience_manager = salience_manager
        self.memory = memory
        self.planning = planning

    # 1. Decide
    def decide(self, actions, state, goal, prev_d):
        # This simulates a very simple policy: the agent takes random actons when there is discrepancy otherwise it chooses to "stay"
        d = self.affect_model.discrepancy(state, goal) # first the decision mechanism retrieves d and delta_d from the affect calculation
        delta_d = self.affect_model.delta_discrepancy(prev_d, d)

        # Choose action
        if d>0:
            if self.planning: # if we have a planning agent, we will use the planning
                action, anticipated_discrepancy = self.plan(state, goal, depth=0)
                if action is None: # there is no plan, so just do random stuff again
                    action = random.choice(actions)
            else: # if we have no planning agent, we will use random choice TODO: I think as soon as we have a learning agent this has to be changed
                action=random.choice(actions)
        else:
            action="stay" # if the discrepancy is already 0 we should stay

        affect = self.affect_model.combined_affect(d=d, delta_d=delta_d, s0=self.salience_manager.retrieve("goal_completion"), s1=self.salience_manager.retrieve("discrepancy"), s2=self.salience_manager.retrieve("delta_discrepancy"))
        #calculating the combined affect just so we have it

        return action, affect, d

    # 2. Act
    def do_action(self, action, state, salience_manager):   
        # simulates action execution; 5 actions possible: 'up', 'down', 'left', 'right', 'stay' TODO: Make this work with any action space that can be taken in
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

        # Possibility to update saliences after taking an action (here i based it on actions, but we can do that based on what we want); TODO: Make this more flexible
        salience_updates = {
            "up": {"goal_completion": 0.05},
            "down": {"discrepancy": 0.02},
            "left": {"delta_discrepancy": -0.01},
            "right": {"goal_completion": -0.03, "discrepancy": 0.01},
            "stay": {"goal_completion": -0.8, "discrepancy": 0.9},
        }

        if action in salience_updates:
            for key, delta in salience_updates[action].items():
                new_salience = salience_manager.retrieve(key) + delta
                salience_manager.update(key, new_salience)

        return newState

    # 3. Plan
    def plan(self, state, goal, depth):
        if depth > 3: # deth of planning is limited to prevent infinite planning; TODO: substitute 3 by max_depth variable?
            return None, self.affect_model.discrepancy(state, goal) # TODO: discrepancy could be a variable defined before, to make it cleaner
        
        k = self.memory._key(state)

        if k in self.memory.prediction_memory:
            anticipated_actions = self.memory.prediction_memory[k] # list of anticipated actions (with states) if there are memories
        else:
            anticipated_actions = {} # empty list if there are no memories

        for action, anticipated_states in anticipated_actions.items():
            for anticipated_state, probability in anticipated_states.items():
                if self.affect_model.discrepancy(anticipated_state, goal) == 0: 
                    return action, self.affect_model.discrepancy(anticipated_state, goal)  # goal is achieved, returns action and discrepancy

            return self.plan(anticipated_state, goal, depth + 1) # TODO: This is wrong somehow, but I don't understand recursion enough with this probabilistic dict
        
        anticipated_discrepancy = self.affect_model.discrepancy(anticipated_state, goal) # I am unsure if we also need to calculate an anticipated affect. I would say no
        return action, anticipated_discrepancy


# ---- MEMORY ----
# Agent memory class, designed to be modular. it makes sure to log transitions, normalize them and then store their probabilities.
# Also can store trajectories (experiences) for future replay (optional)

class Memory:
    def __init__(self):
        self.transition_memory = {} # Counts and stores transitions
        self.experience_memory = [] # Stores trajectories
        self.prediction_memory = {} # Stores probabilities for transitions. THIS should be used for decision making and planning

    def _key(self, s):
        return (s,) if not isinstance(s, (list, tuple)) else tuple(s)


    def store(self, state, action, newState):
        # Logs state transitions and stores trajectories
        # Also calls the update function to caclulate and store probabities
        # Edit: changed to probabilistic settings
        # Datastructure is a nested dictionary
        # TODO: Clean up that whole tuple(state) mess
        k = self._key(state)
        if k not in self.transition_memory: # makes sure not to use something that has never been seen before
            self.transition_memory[k]= {} # initializes the dictionary if it doesn't exist yet

        if action not in self.transition_memory[k]: # makes sure not to use something that has never been seen before
            self.transition_memory[k][action] = defaultdict(int) # initializes the dictionary if it doesn't exist yet

        self.transition_memory[k][action][self._key(newState)] += 1 # counts transitions and stores them in transition_memory

        # experience_memory.append((tuple(state), action, tuple(newState))) # TODO: Make this so that we can distinguish between episodes and selectively recall them; I don't know how to do this yet


        # Update probabilities after logging
        self.update_transition_probabilities(state, action)

    def update_transition_probabilities(self, state, action): # Normalizes transitions so that they can be used as probabilities
        k = self._key(state)
        counts = self.transition_memory[k][action]
        total = sum(counts.values())
        if k not in self.prediction_memory:
            self.prediction_memory[k] = {}
        if action not in self.prediction_memory[k]:
            self.prediction_memory[k][action] = {}
        for newState, count in counts.items():
            self.prediction_memory[k][action][newState] = count / total if total > 0 else 0.0

# ---- ENVIRONMENTs ----
# Simple Money world environment

class MoneyMDPEnv:
    # States are 0 to 5 (5 is terminal)
    # Actions: "go" (to the right), "stay"
    # Rewards: +1 for every go, 0 otherwise

    @property
    def actions(self):
        return ["go", "stay"]

    def __init__(self, max_state=5):
        self.state = 0
        self.max_state = max_state

    def step(self, action): 
        prev = self.state
        if action == "go" and self.state < self.max_state:
            self.state += 1
        elif action == "stay":
            pass
        if self.state > prev:
            reward = 1
        else:
            reward = 0
        done = self.state == self.max_state
        return self.state, reward, done
    
    def reset(self):
        self.state = 0
        return self.state




# ---- OVERALL AGENT ----
# Agent class with all the core components
 
class Agent:
    def __init__(self, planning, env):
        self.affect_model = AffectModel(max_discrepancy=None)
        self.salience_manager = SalienceManager({
            "goal_completion": 0.3,
            "discrepancy": 0.4,
            "delta_discrepancy": 0.3
        }) # Maybe this should be fed in with the environment?
        self.memory = Memory()
        self.decision_making = DecisionMaking(
            affect_model=self.affect_model,
            salience_manager=self.salience_manager,
            memory=self.memory,
            planning=planning
        )
        

        self.planning = planning
        self.env = env
        self.state = self.env.reset()   # scalar 0..5
        self.goal = 5                   # scalar goal (€5)
        self.actions = self.env.actions # ["go","stay"]

        self.prev_d = self.affect_model.discrepancy(self.state, self.goal)

    def run(self):
        while True:
            print(f"Current state: {self.state}")
            action, affect, d = self.decision_making.decide(
                actions=self.actions,
                state=self.state,
                goal=self.goal,
                prev_d=self.prev_d
            )
            print(f"Chosen action: {action}, Affect: {affect:.2f}, Discrepancy: {d:.2f}")

            new_state, reward, done = self.env.step(action)
            self.memory.store(self.state, action, new_state)

            self.prev_d = d
            self.state = new_state

            if done:
                print("Reached goal (5). Episode finished.")
                break

            time.sleep(0.3)



env = MoneyMDPEnv(max_state=5)
agent = Agent(planning=True, env=env)
agent.run()


# TODO: Discuss if goalValue calculation (empty function from Joosts code) should be added: 
# def goalValue():
    #Value of the goal moderates affect intensity
    #The feature dimension has does not need a weight, unless we want to use it for a "personality" (e.g. money goals are valued a lot when they are adopted)
    #The goal itself has a value
    #Goals always have non-zero positive value
    
    #Goal have activation level! 
    #return activation*value
