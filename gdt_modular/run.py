import matplotlib.pyplot as plt
import numpy as np

from envs.dice import Dice
from envs.doors_paper import Doors
from envs.corridor import Corridor
from agent.agent import Agent
from agent.affect import AffectModel

# --- GENERAL RUN ---
def run_experiment(env, agent, trials=100, flip_trial=None, flip_beliefs=False):
    outcome_log = []
    affect_log = []

    for t in range(trials):
        # 1. "Flip" logic
        if flip_trial and t == flip_trial:
            new_probs = [0.2, 0.8] # Example flip
            env.p_list = new_probs
            if flip_beliefs:
                agent.beliefs = new_probs 

        # 2. Standard Interaction Cycle
        agent.decide()
        total, ad, aas = agent.get_affect()
        
        # 3. Logging
        affect_log.append({"total": total, "ad": ad, "aas": aas})
        outcome_log.append(agent.current_state)
        
        env.reset()

    return {"rolls": outcome_log, "affect_scores": affect_log}

# --- TASK RUNNERS ---
def run_dice_task(target_val=6, binary=False):
    env = Dice()
    
    if binary:
        # Experiment 1a: Binary logic
        # Perception: If I see the goal_val, I perceive '6'. Otherwise, I perceive '0'.
        # (This ensures d is 0 for success and 6 for failure)
        def binary_lens(raw_input):
            # Handles if raw_input is [6.0] or just 6.0
            val = raw_input[0] if isinstance(raw_input, (list, np.ndarray)) else raw_input
            return {"feature value": 6.0 if val == target_val else 0.0}
        
        targets = {"feature value": 6.0}
        p_filter = binary_lens
    else:
        # Experiment 1b: Gradual logic
        # Perception: Just look at the raw face value.
        targets = {"feature value": float(target_val)}
        p_filter = None # No transformation needed

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=p_filter)
    
    return run_experiment(env, agent, trials=100)

def run_doors_task(agent_knows_flip=False):
    initial_p = [0.8, 0.2]
    #initial_p = [1.0, 0.0]
    #env = Doors(outcome_space=[1,2,3,4,5,6], true_p_list=initial_p)
    env = Doors(outcome_space=[1,2], true_p_list=initial_p)
    targets = {"feature value": 6}
    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    
    agent = Agent(env, affect_model, targets, perception_filter=None)
    agent.beliefs = initial_p 
    
    return run_experiment(env, agent, trials=100, flip_trial=50, flip_beliefs=agent_knows_flip)


# Corridor
def run_binary_corridor(agent_type="perfect"):
    env = Corridor(length=6)
    
    targets = {"goal_dim": 1.0}
    
    def binary_lens(raw_features):
        val = raw_features["goal_dim"]
        # If at the end (6.0), perceive 1.0. Otherwise, perceive 0.0.
        return {"goal_dim": 1.0 if val == 6.0 else 0.0}

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=binary_lens)
    
    # Policy setup
    if agent_type == "perfect":
        agent.beliefs = [0.0, 1.0] # 0% Left, 100% Right
    
    # We run until the agent reaches the end, or a set number of trials
    return run_experiment(env, agent, trials=6)


def run_gradual_corridor(agent_type="perfect"):
    env = Corridor(length=6)
    
    # Target: The agent wants to reach the value 6.0
    targets = {"goal_dim": 6.0}
    
    def gradual_lens(raw_features):
        # Pass the raw map values (1.0, 2.0, 3.0...) directly
        return {"goal_dim": raw_features["goal_dim"]}

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=gradual_lens)
    
    if agent_type == "perfect":
        agent.beliefs = [0.0, 1.0]
    
    return run_experiment(env, agent, trials=6)


def plot_affectvsoutcome(data, title="GDT Model", ylabel="Outcome Value"):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- Axis 1: Environment Outcomes ---
    ax1.set_xlabel('Trial')
    ax1.set_ylabel(ylabel, color='gray')
    # Plotting the states/rolls
    ax1.scatter(range(len(data["rolls"])), data["rolls"], color='gray', alpha=0.3, label="Actual Outcome")
    # Target line (the goal)
    ax1.axhline(y=6, color='green', linestyle='--', alpha=0.5, label="Target Goal (6)")
    ax1.set_ylim(0, max(data["rolls"]) + 1)

    # --- Axis 2: Affective States ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Affective Intensity')

    total_scores = [entry["total"] for entry in data["affect_scores"]]
    ad_scores = [entry["ad"] for entry in data["affect_scores"]]
    aas_scores = [entry["aas"] for entry in data["affect_scores"]]

    # Traces
    ax2.plot(total_scores, color='blue', linewidth=2, label="Total Affect (A_it)")
    ax2.plot(ad_scores, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
    ax2.plot(aas_scores, color='orange', linestyle='-', alpha=0.6, label="AAS (Action Selection)")

    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()
    plt.show() # This triggers a separate window/plot for each call

def plot_corridor_comparison(binary_data, gradual_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Top Plot: Binary Goal (All-or-Nothing) ---
    b_total = [e["total"] for e in binary_data["affect_scores"]]
    b_ad = [e["ad"] for e in binary_data["affect_scores"]]
    
    ax1.plot(b_total, label="Total Affect (Binary)", color='blue', linewidth=3)
    ax1.plot(b_ad, label="AD (Discrepancy)", color='red', linestyle='--')
    ax1.set_title("Experiment A: Binary Goal (Only State 6 is 'Success')")
    ax1.set_ylabel("Affect Intensity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Bottom Plot: Gradual Goal (Distance) ---
    g_total = [e["total"] for e in gradual_data["affect_scores"]]
    g_ad = [e["ad"] for e in gradual_data["affect_scores"]]
    
    ax2.plot(g_total, label="Total Affect (Gradual)", color='green', linewidth=3)
    ax2.plot(g_ad, label="AD (Discrepancy)", color='orange', linestyle='--')
    ax2.set_title("Experiment B: Gradual Goal (Distance to State 6)")
    ax2.set_ylabel("Affect Intensity")
    ax2.set_xlabel("Trial (Step in Corridor)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --- EXPERIMENT 1a: DICE (Binary Success) ---
# Here we treat the goal as "Did I hit the 6?" 
# Target is 1. We assume the environment/agent maps 6 -> 1 and everything else -> 0.
print("Simulating Dice Task 1a (Binary)...")
dice_data_1a = run_dice_task(target_val=1) 
plot_affectvsoutcome(dice_data_1a, title="Exp 1a: Dice (Binary Goal)", ylabel="Success (0 or 1)")

# --- EXPERIMENT 1b: DICE (Gradual Distance) ---
# Here we treat the goal as the raw face value.
print("Simulating Dice Task 1b (Gradual)...")
dice_data_1b = run_dice_task(target_val=6) 
plot_affectvsoutcome(dice_data_1b, title="Exp 1b: Dice (Gradual Goal)", ylabel="Dice Face Value")

# --- EXPERIMENT 2: DOORS ---
print("Simulating Doors Task...")
# Using False for 'agent_knows_flip' to ensure we see the disappointment spike
doors_data = run_doors_task(agent_knows_flip=False)
plot_affectvsoutcome(
    doors_data, 
    title="Task 2: Doors Task (Choice, Hidden Probability Flip at T=50)", 
    ylabel="Door Number Selected"
)

# --- EXPERIMENT 3: CORRIDORS ---
def run_comparison():
    # 1. Run Binary
    print("Running Binary Corridor...")
    binary_results = run_binary_corridor(agent_type="perfect")
    
    # 2. Run Gradual
    print("Running Gradual Corridor...")
    gradual_results = run_gradual_corridor(agent_type="perfect")
    
    # 3. Plot
    plot_corridor_comparison(binary_results, gradual_results)

# Execute the test
run_comparison()
