import matplotlib.pyplot as plt
import numpy as np

from envs.dice import Dice
from envs.doors_paper import Doors
from envs.corridor import Corridor
from agent.agent import Agent
from agent.affect import AffectModel

# --- GENERAL RUNNER ---
def run_experiment(env, agent, trials=100, flip_trial=None, flip_beliefs=False):
    outcome_log = []
    affect_log = []

    # Initialize Environment and Agent baseline
    env.reset()
    agent.perceive() # Look at current state (State 1)
    
    # Record Baseline (Time 0)
    total, ad, aas = agent.get_affect()
    affect_log.append({"total": total, "ad": ad, "aas": aas})
    outcome_log.append(1.0) # The starting state is State 1

    for t in range(trials):
        # 1. Logic for the 'Doors' flip experiment
        if flip_trial and t == flip_trial:
            new_probs = [0.2, 0.8] 
            env.p_list = new_probs
            if flip_beliefs:
                agent.beliefs = new_probs 

        # 2. The GDT Cycle
        agent.decide()
        total, ad, aas = agent.get_affect()
        
        # 3. Data Collection
        affect_log.append({"total": total, "ad": ad, "aas": aas})
        # We log the 'state' or 'outcome' for plotting
        outcome_log.append(agent.current_state)
        
        #env.reset()

    return {"rolls": outcome_log, "affect_scores": affect_log}


# --- TASK RUNNERS ---
def run_dice_task(mode="binary"):
    env = Dice()
    
    if mode == "binary":
        targets = {"goal_dim": 1.0} # Target is 'Success'
        def lens(raw):
            # Only a 6 is perceived as a 1.0 (Success)
            return {"goal_dim": 1.0 if raw["goal_dim"] == 6.0 else 0.0}
    else:
        targets = {"goal_dim": 6.0} # Target is the value 6
        def lens(raw):
            return raw # Gradual: sees the actual face value 1-6

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=lens)
    
    return run_experiment(env, agent, trials=10)


def run_doors_task(agent_knows_flip=False):
    # Standardizing to our unified feature name
    initial_p = [0.8, 0.2]
    env = Doors(outcome_space=[1, 2], p_list=initial_p) 
    
    # In Doors, 'Success' is objectively 1.0 in our new Env map
    targets = {"goal_dim": 1.0}
    
    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=None)
    agent.beliefs = initial_p 
    
    return run_experiment(env, agent, trials=100, flip_trial=50, flip_beliefs=agent_knows_flip)

def run_corridor_task(mode="gradual", agent_type="perfect"):
    env = Corridor(length=6)
    
    if mode == "binary":
        # Target is "Success" (1.0)
        targets = {"goal_dim": 1.0}
        def lens(raw):
            # Only state 6 is perceived as 1.0
            return {"goal_dim": 1.0 if raw["goal_dim"] == 6.0 else 0.0}
    else:
        # Target is the 6th position
        targets = {"goal_dim": 6.0}
        def lens(raw):
            # Gradual: perceive the raw distance/progress
            return raw

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=lens)
    
    agent.set_policy(agent_type)

    # Run for 6 trials (enough for a perfect agent to finish)
    return run_experiment(env, agent, trials=25)

def run_corridor_comparison_data(mode="gradual"):
    policies = ["perfect", "good", "random"]
    results = {}

    for p in policies:
        results[p] = run_corridor_task(mode=mode, agent_type=p)
        
    return results

# --------- PLOTTING ------------

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

def plot_policy_overlay(all_results, mode_title="Gradual"):
    plt.figure(figsize=(14, 8))
    
    # Define colors so we can distinguish the three agents
    colors = {"perfect": "blue", "good": "green", "random": "red"}
    
    # Use the 'perfect' run to generate x-axis labels
    first_run = all_results["perfect"]
    x_labels = [f"Start\n(S1)" if i == 0 else f"T{i-1}\n(S{int(s)})" 
                for i, s in enumerate(first_run["rolls"])]
    x_ticks = range(len(x_labels))

    for policy, data in all_results.items():
        c = colors[policy]
        total = [e["total"] for e in data["affect_scores"]]
        ad = [e["ad"] for e in data["affect_scores"]]
        aas = [e["aas"] for e in data["affect_scores"]]

        # Line types as requested: total=-, ad=--, aas=:
        plt.plot(x_ticks, total, color=c, linestyle='-', linewidth=2, label=f"{policy.capitalize()} (Total)")
        plt.plot(x_ticks, ad, color=c, linestyle='--', alpha=0.4, label=f"{policy.capitalize()} (AD)")
        plt.plot(x_ticks, aas, color=c, linestyle=':', alpha=0.6, label=f"{policy.capitalize()} (AAS)")

    plt.xticks(x_ticks, x_labels)
    plt.title(f"Policy Comparison: {mode_title} Corridor")
    plt.ylabel("Affective Intensity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


 # --- RUN ---
# --- EXPERIMENT 1: DICE COMPARISON ---
print("Simulating Dice...")
dice_bin = run_dice_task(mode="binary")
dice_grad = run_dice_task(mode="gradual")
plot_affectvsoutcome(dice_bin, title="Exp 1a: Dice (Binary)", ylabel="Success (0 or 1)")
plot_affectvsoutcome(dice_grad, title="Exp 1b: Dice (Gradual)", ylabel="Face Value")

# --- EXPERIMENT 2: DOORS (Probability Flip) ---
doors_data = run_doors_task(agent_knows_flip=False)
plot_affectvsoutcome(doors_data, title="Exp 3: Doors (Hidden Flip)", ylabel="Door Choice")

# --- EXPERIMENT 3: CORRIDOR POLICY COMPARISON ---
print("Simulating Corridor Policy Overlays...")
# Run and plot for Binary mode
binary_comparison = run_corridor_comparison_data(mode="binary")
plot_policy_overlay(binary_comparison, "Binary")
# Run and plot for Gradual mode
gradual_comparison = run_corridor_comparison_data(mode="gradual")
plot_policy_overlay(gradual_comparison, "Gradual")

print("Finished all simulations!")
