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
            #episode_log["outcomes"].append(agent.current_state)
            episode_log["outcomes"].append(env.get_features()["goal_dim"])
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

# Corridor Run Function
def run_corridor_task(mode="gradual", seed=None,length=6, trap_prob=0.1):
    # Setup Env
    env = Corridor(length=length, trap_prob=trap_prob, seed=seed)
    targets = {"goal_dim": float(length)}
    
    # Setup Lens/Target
    if mode == "binary":
        # Target is "Success" (1.0)
        #targets = {"goal_dim": 1.0}
        def lens(raw):
            # Only state 6 is perceived as success with maximum feature distance (could also be 1.0)
            val = float(length) if raw["goal_dim"] == float(length) else 0.0 
            return {"goal_dim": val}
    else:
        # Target is the 5th position
        targets = {"goal_dim": float(length)}
        def lens(raw):
            # Gradual: perceive the raw distance/progress
            return raw

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=lens)
    
    return run_simulation(env, agent, num_episodes=1, max_steps=length+2)

# --------- PLOTTING ------------

# Dice
def plot_affectvsoutcome_dice(all_data, title="GDT Model", ylabel="Outcome Value", target_val=6):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    flat_outcomes = []
    flat_total = []
    flat_ad = []
    flat_aas = []
    episode_separators = []
    
    current_step = 0
    for ep in all_data:
        flat_outcomes.extend(ep["outcomes"])
        flat_total.extend([step["total"] for step in ep["affect"]])
        flat_ad.extend([step["ad"] for step in ep["affect"]])
        flat_aas.extend([step["aas"] for step in ep["affect"]])
        
        current_step += len(ep["outcomes"])
        episode_separators.append(current_step - 0.5)

    ax1.set_xlabel('Steps (Sequential Episodes)')
    ax1.set_ylabel(ylabel, color='gray')
    ax1.scatter(range(len(flat_outcomes)), flat_outcomes, color='gray', alpha=0.3, label="Actual Outcome")
    ax1.axhline(y=target_val, color='green', linestyle='--', alpha=0.5, label=f"Goal ({target_val})")
    ax1.set_ylim(-0.5, max(flat_outcomes + [target_val]) + 1)

    for sep in episode_separators[:-1]:
        ax1.axvline(x=sep, color='black', linestyle='-', alpha=0.1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Affective Intensity')
    ax2.plot(flat_total, color='blue', linewidth=2, label="Total Affect")
    ax2.plot(flat_ad, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
    ax2.plot(flat_aas, color='orange', linestyle='-', alpha=0.6, label="AAS (Expectancy)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.tight_layout()
    plt.show()


# Corridor
def plot_affectvsoutcome_corridor(all_data, mode="gradual", title="GDT Model", ylabel="Outcome Value", length=6, target_val=6):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- 1.Flattening multiple episodes into one long list ---
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

    steps = np.arange(len(flat_outcomes))

    # We define the Y1 range (Position)
    y1_min, y1_max = -4, 8 
    # We define the Y2 range (Affect) to perfectly align: Affect = Position - 6
    y2_min, y2_max = y1_min - 6, y1_max - 6 # Results in [-10, 2]

    ax1.set_ylim(y1_min, y1_max)
    ax1.set_xlim(-0.5, length + 1)

    # Preparation
    # 1. Neutral Baseline & "The Floor"
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    # Fill the area below 0 to show the "Sub-baseline" failure zone
    x_lims = ax1.get_xlim()
    ax1.fill_between(x_lims, -4, 0, color='gray', alpha=0.1, label="Failure Zone")

    # 2. The State Walking Line (Step Plot)
    # This shows the progress clearly and stays flat if the agent is stuck/terminal
    ax1.step(steps, flat_outcomes, where='post', color='gray', alpha=0.7, linewidth=2, label="Agent Path")
    ax1.scatter(steps, flat_outcomes, color='gray', s=30) # Add dots at the joints

    # 3. Fixed X-Axis (Lost Opportunity)
    # We force the x-axis to show the full potential length of the corridor
     
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Position / Feature Value', color='gray')
    
    # --- 2. flat_outcomes ---
    ax1.scatter(range(len(flat_outcomes)), flat_outcomes, color='gray', alpha=0.3, label="Actual Outcome")
    
    # Target line (using the target_val variable)
    ax1.axhline(y=target_val, color='green', linestyle='--', alpha=0.5, label=f"Target Goal ({target_val})")
    #ax1.set_ylim(min(flat_outcomes + [-3.5]), max(flat_outcomes + [target_val]) + 1)

    # --- Axis 2: Affective States ---
    ax2 = ax1.twinx()
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel('Affective Intensity')

    # --- 3. SUBSTITUTION: data["affect_scores"] variables -> flattened lists ---
    ax2.plot(flat_total, color='blue', linewidth=2, label="Total Affect (A_it)")
    ax2.plot(flat_ad, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
    ax2.plot(flat_aas, color='orange', linestyle='-', alpha=0.6, label="AAS (Action Selection)")

    # Combined Legend (Exactly as you had it)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f"{title} | Mode: {mode.upper()}")
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
plot_affectvsoutcome_dice(dice_bin, title="GDT Model", ylabel="Outcome Value", target_val=6)
plot_affectvsoutcome_dice(dice_grad, title="GDT Model", ylabel="Outcome Value", target_val=6)

# # --- EXPERIMENT 2: DOORS (Probability Flip) ---
# doors_data = run_doors_task(agent_knows_flip=False)
# plot_affectvsoutcome(doors_data, title="Exp 3: Doors (Hidden Flip)", ylabel="Door Choice")

# # --- EXPERIMENT 3: CORRIDOR POLICY COMPARISON ---

# Choose 6 seeds. 
seeds = [42, 7, 10, 15, 21, 99] 

print("Simulating binary Corridor Case Studies...")

for i, s in enumerate(seeds):
    # Run one specific trajectory
    single_walk_data = run_corridor_task(mode="binary", seed=s)
    
    # Plot it
    plot_affectvsoutcome_corridor(
        single_walk_data, 
        mode="binary", 
        title=f"Walk {i+1} (Seed: {s})", 
        ylabel="Position (0-6)", 
        target_val=6.0
    )

print("Simulating gradual Corridor Case Studies...")

for i, s in enumerate(seeds):
    # Run one specific trajectory
    single_walk_data = run_corridor_task(mode="gradual", seed=s)
    
    # Plot it
    plot_affectvsoutcome_corridor(
        single_walk_data, 
        mode="gradual", 
        title=f"Walk {i+1} (Seed: {s})", 
        ylabel="Position (0-6)", 
        target_val=6.0
    )
# print("Simulating Corridor Policy Overlays...")
# # Run and plot for Binary mode
# binary_comparison = run_corridor_comparison_data(mode="binary")
# plot_policy_overlay(binary_comparison, "Binary")
# # Run and plot for Gradual mode
# gradual_comparison = run_corridor_comparison_data(mode="gradual")
# plot_policy_overlay(gradual_comparison, "Gradual")

print("Finished all simulations!")
