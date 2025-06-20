import random
import time

MAX_ENERGY=2
PLANNING= True
defaultGoalEffect=0 #effect of goals when the goals are completely reached (0 means no effect of achieved goals on affect)


def affect(state):
    #Assuming the current state, returns affect associated with a particular state, wieghted according to the probablity of thet state happening.
    
def store(state, action, newState, memory):
    #Stores state transitions in determinstic MDP form. (state,action,nextstate)
    #Change to probablity
    memState={}
    if tuple(state) in memory:
        memState=memory[tuple(state)]
    memState[action]=newState
    memory[tuple(state)]=memState

def discrepancy(state, goal):
    #just calculates the sum of the feature differences as discrepancy, for those features in the goal that are not set to None (those are ignored, this allows for multiple goals later)
    return sum(abs(a - b) if b is not None else 0 for a, b in zip(state, goal))
    #add a cap to the discrep, meaning you cant go over the discrepancy.
    #no homeostatic goals yet! only positive achievement goals
    #Goals with 0 discr may (discr is a constant) keep generating positive affect

def deltaDiscrepancy():
    #to implement
    #Size of the change?
    #Reletaive to where I am now (state s) I have 2
    #ref2 is where I came from (eg 0 euro)
    #ref1 is where I want to go (eg 5 euro)
    #delateDiscrep=2
    #desired reduction=5, actual reducation is 2, so 2/5? (delateDisc weight, discrepancy weight)

def goalValue():
    #Value of the goal moderates affect intensity
    #The feature dimension has does not need a weight, unless we want to use it for a "personality" (e.g. money goals are valued a lot when they are adopted)
    #The goal itself has a value
    #Goals always have non-zero positive value
    
    #Goal have activation level! 
    return activation*value
    
def combineAffect():
    #affect from 3 sources need to be include
    
    #Saliency is typically between 0-1
    #Deiscrep and deltaDiscr are assumed now to include the goal salience in some way (so not a seprated factor there for the representation salience).
    
    if (discrepancy()==0):
        return goalRepresentationSalience(goal) * goalValue(goal) 
    else:
         return goalDiscrepancySaliency(goal) * goalValue() * discrepancy() + goalDealtDiscrepancySaliency(goal) * goalValue()  * deltaDiscrepancy()
    
def decide(actions, state, goal):
    # This simulates a very simple policy: the agent takes random actons when there is discrepancy otherwise it chooses to "stay"
    if discrepancy(state,goal)>0:
        action=random.choice(actions)
    else:
        action="stay"
    
    return action, affect(discrepancy(state, goal))

def plan(state, goal, memory, depth):
    #Returns the first action that will lead to a state with a reduced discrepancy
    if depth>3:
        return None, discrepancy(state, goal)
    
    if tuple(state) in memory:
        actions=memory[tuple(state)]
    else:
        actions={}
    
    #does not work with probs I think
    for a, s in actions.items():
        if discrepancy(s,goal)==0:
            return a, discrepancy(s,goal)
            
        return plan(s, goal, memory, depth+1)
    
    return None, discrepancy(state, goal)

def decide_with_plan(actions, state, goal, memory):
    # This simulates a very simple policy:
    #If global planning is set to False the agent takes random actons when there is discrepancy otherwise it chooses to "stay"
    #If planning is set to True it will first look in its MDP to find an action that might reduce discrepancy, and select that one, otherwise do a random one
    anticipation=None
    
    if discrepancy(state,goal)>0:
        action, anticipation = plan(state, goal, memory, 0)
        if action is None: # there is no plan, do some random stuff again
            action=random.choice(actions)
    else:
        action="stay"
    
    return action, affect_with_anticipation(discrepancy(state, goal), anticipation)

def doAction(action, state):
    #this simulates the world and the agent body implictly
    
    global MAX_ENERGY
    newState=state.copy()
    if action == "up":
        newState[0]=newState[0]-1 #move up
    if action == "down":
        newState[0]=newState[0]+1 #move down
    if action == "left":
        newState[1] = newState[1]-1 #move left
    if action == "right":
        newState[1]=newState[1]+1 #move right
        
    newState[2]=newState[2]-1     #every action takes 1 energy
    
    if action=="eat" and newState[0]==1 and newState[1]==1: #we are at the food (1,1)
        newState[2]=MAX_ENERGY
    
    newState[0]=newState[0]%2
    newState[1]=newState[1]%2
    newState[2]=0 if newState[2] < 0 else (MAX_ENERGY if newState[2]>MAX_ENERGY else newState[2])
    return newState
    

def agent():
    global  MAX_ENERGY
    global PLANNING
    
    #This is the agent
    state=[] #[x,y,energy]
    
    memory={} #the MDP of the agent
    actions=["up","down","left","right","stay","eat"]

    state =[0,0,MAX_ENERGY]
    goal  =[None,None,MAX_ENERGY]
    #goals have salience (representational activation), not the same as attention/awareness, and value
    
    #It keeps doing this forever
    while True:
        print(state)
        
        if PLANNING:
            action, affect = decide_with_plan(actions, state, goal, memory)
        else:
            action, affect = decide(actions, state, goal)
        
        print(action, affect)
        
        newState=doAction(action, state)
        store(state,action,newState,memory)#Store a new state in the MDP
        
        state=newState
        time.sleep(0.5)
        

agent()