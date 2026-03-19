import matplotlib.pyplot as plt
import numpy as np

from envs.dice import Dice
from envs.doors_paper import Doors
from envs.corridor import Corridor
from agent.agent import Agent
from agent.affect import AffectModel

# --- GENERAL RUNNER ---
def run_simulation(env, agent, num_episodes=10, max_steps=25):
    """
    Runs a set number of episodes. 
    In the Corridor, an episode is a full walk.
    In the Dice, an episode is a single throw sequence.
    """
    all_data = []

    for ep in range(num_episodes):
        env.reset()
        agent.perceive() # Initial state perception
        
        episode_log = {
            "outcomes": [],
            "affect": []
        }

        # Record Baseline (Time 0) before any action
        total, ad, aas = agent.get_affect()
        episode_log["affect"].append({"total": total, "ad": ad, "aas": aas})
        episode_log["outcomes"].append(env.state)

        # The Episode Loop: continues until terminal (Goal or Trap)
        step_count = 0
        while not env.is_terminal() and step_count < max_steps: # Safety break
            agent.decide()
            total, ad, aas = agent.get_affect()
            
            episode_log["affect"].append({"total": total, "ad": ad, "aas": aas})
            episode_log["outcomes"].append(agent.current_state)
            step_count += 1
            
        all_data.append(episode_log)

    return all_data



# --- TASK RUNNERS ---
def run_dice_task(mode="binary", num_throws=10):
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
    
    # Each throw is an episode. Max steps is small because Dice is a short cycle.
    return run_simulation(env, agent, num_episodes=num_throws, max_steps=2)

def run_corridor_task(mode="gradual", num_walks=5):
    # Setup Env
    env = Corridor(length=6, trap_prob=0.1)
    
    # Setup Lens/Target
    if mode == "binary":
        # Target is "Success" (1.0)
        targets = {"goal_dim": 1.0}
        def lens(raw):
            # Only state 6 is perceived as 1.0
            return {"goal_dim": 1.0 if raw["goal_dim"] == 6.0 else 0.0}
    else:
        # Target is the 5th position
        targets = {"goal_dim": 5.0}
        def lens(raw):
            # Gradual: perceive the raw distance/progress
            return raw

    affect_model = AffectModel(v=1.0, targets=targets)
    agent = Agent(env, affect_model, targets, perception_filter=lens)
    
    return run_simulation(env, agent, num_episodes=num_walks, max_steps=10)

# --------- PLOTTING ------------
def plot_affectvsoutcome(all_data, title="GDT Model", ylabel="Outcome Value", target_val=6):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- 1. SUBSTITUTION: Flattening multiple episodes into one long list ---
    # Instead of one list 'data["rolls"]', we build new lists by 
    # stitching all episodes together sequentially.
    flat_outcomes = []
    flat_total = []
    flat_ad = []
    flat_aas = []
    episode_separators = [] # Marks the end of each episode for visual clarity
    
    current_step = 0
    for ep in all_data:
        flat_outcomes.extend(ep["outcomes"])
        flat_total.extend([step["total"] for step in ep["affect"]])
        flat_ad.extend([step["ad"] for step in ep["affect"]])
        flat_aas.extend([step["aas"] for step in ep["affect"]])
        
        # Track where the episode ends to draw a divider
        current_step += len(ep["outcomes"])
        episode_separators.append(current_step - 0.5)

    # --- Axis 1: Environment Outcomes ---
    ax1.set_xlabel('Total Steps (Episodes stitched together)')
    ax1.set_ylabel(ylabel, color='gray')
    
    # --- 2. SUBSTITUTION: data["rolls"] -> flat_outcomes ---
    ax1.scatter(range(len(flat_outcomes)), flat_outcomes, color='gray', alpha=0.3, label="Actual Outcome")
    
    # Target line (using the target_val variable)
    ax1.axhline(y=target_val, color='green', linestyle='--', alpha=0.5, label=f"Target Goal ({target_val})")
    ax1.set_ylim(-0.5, max(flat_outcomes + [target_val]) + 1)

    # NEW: Vertical dividers to show where one episode ends and the next begins
    for sep in episode_separators[:-1]:
        ax1.axvline(x=sep, color='black', linestyle='-', alpha=0.1)

    # --- Axis 2: Affective States ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Affective Intensity')

    # --- 3. SUBSTITUTION: data["affect_scores"] variables -> flattened lists ---
    ax2.plot(flat_total, color='blue', linewidth=2, label="Total Affect (A_it)")
    ax2.plot(flat_ad, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
    ax2.plot(flat_aas, color='orange', linestyle='-', alpha=0.6, label="AAS (Action Selection)")

    # Combined Legend (Exactly as you had it)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()
    plt.show()



# def plot_affectvsoutcome(all_episodes_data, title="GDT Model", ylabel="Outcome Value"):
#     fig, ax1 = plt.subplots(figsize=(12, 6))
    
#     # --- Axis 1: Environment Outcomes ---
#     ax1.set_xlabel('Trial')
#     ax1.set_ylabel(ylabel, color='gray')
#     # Plotting the states/rolls
#     ax1.scatter(range(len(data["rolls"])), data["rolls"], color='gray', alpha=0.3, label="Actual Outcome")
#     # Target line (the goal)
#     ax1.axhline(y=6, color='green', linestyle='--', alpha=0.5, label="Target Goal (6)")
#     ax1.set_ylim(0, max(data["rolls"]) + 1)

#     # --- Axis 2: Affective States ---
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Affective Intensity')

#     total_scores = [entry["total"] for entry in data["affect_scores"]]
#     ad_scores = [entry["ad"] for entry in data["affect_scores"]]
#     aas_scores = [entry["aas"] for entry in data["affect_scores"]]

#     # Traces
#     ax2.plot(total_scores, color='blue', linewidth=2, label="Total Affect (A_it)")
#     ax2.plot(ad_scores, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
#     ax2.plot(aas_scores, color='orange', linestyle='-', alpha=0.6, label="AAS (Action Selection)")

#     # Combined Legend
#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper left')

#     plt.title(title)
#     plt.tight_layout()
#     plt.show() # This triggers a separate window/plot for each call

# def plot_policy_overlay(all_results, mode_title="Gradual"):
#     plt.figure(figsize=(14, 8))
    
#     # Define colors so we can distinguish the three agents
#     colors = {"perfect": "blue", "good": "green", "random": "red"}
    
#     # Use the 'perfect' run to generate x-axis labels
#     first_run = all_results["perfect"]
#     x_labels = [f"Start\n(S1)" if i == 0 else f"T{i-1}\n(S{int(s)})" 
#                 for i, s in enumerate(first_run["rolls"])]
#     x_ticks = range(len(x_labels))

#     for policy, data in all_results.items():
#         c = colors[policy]
#         total = [e["total"] for e in data["affect_scores"]]
#         ad = [e["ad"] for e in data["affect_scores"]]
#         aas = [e["aas"] for e in data["affect_scores"]]

#         # Line types as requested: total=-, ad=--, aas=:
#         plt.plot(x_ticks, total, color=c, linestyle='-', linewidth=2, label=f"{policy.capitalize()} (Total)")
#         plt.plot(x_ticks, ad, color=c, linestyle='--', alpha=0.4, label=f"{policy.capitalize()} (AD)")
#         plt.plot(x_ticks, aas, color=c, linestyle=':', alpha=0.6, label=f"{policy.capitalize()} (AAS)")

#     plt.xticks(x_ticks, x_labels)
#     plt.title(f"Policy Comparison: {mode_title} Corridor")
#     plt.ylabel("Affective Intensity")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, alpha=0.2)
#     plt.tight_layout()
#     plt.show()


 # --- RUN ---
# --- EXPERIMENT 1: DICE COMPARISON ---
print("Simulating Dice...")
dice_bin = run_dice_task(mode="binary")
dice_grad = run_dice_task(mode="gradual")
plot_affectvsoutcome(dice_bin, title="GDT Model", ylabel="Outcome Value", target_val=6)
plot_affectvsoutcome(dice_grad, title="GDT Model", ylabel="Outcome Value", target_val=6)

# # --- EXPERIMENT 2: DOORS (Probability Flip) ---
# doors_data = run_doors_task(agent_knows_flip=False)
# plot_affectvsoutcome(doors_data, title="Exp 3: Doors (Hidden Flip)", ylabel="Door Choice")

# # --- EXPERIMENT 3: CORRIDOR POLICY COMPARISON ---
# print("Simulating Corridor Policy Overlays...")
# # Run and plot for Binary mode
# binary_comparison = run_corridor_comparison_data(mode="binary")
# plot_policy_overlay(binary_comparison, "Binary")
# # Run and plot for Gradual mode
# gradual_comparison = run_corridor_comparison_data(mode="gradual")
# plot_policy_overlay(gradual_comparison, "Gradual")

print("Finished all simulations!")
