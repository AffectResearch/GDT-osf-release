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

        agent.last_action_p = agent.beliefs[0]
        
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
def run_dice_task(mode="binary", num_throws=10, seed=None):
    env = Dice(seed=seed)
    
    if mode == "binary":
        targets = {"goal_dim": 5.0} # Target is 'Success'
        def lens(raw):
            # Only a 6 is perceived as a 1.0 (Success)
            return {"goal_dim": 5.0 if raw["goal_dim"] == 6.0 else 0.0}
    else:
        targets = {"goal_dim": 5.0} # Target is the value 6
        def lens(raw):
            return raw # Gradual: sees the actual face value 1-6

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, perception_filter=lens)

    all_data = []

    # Initialize Once
    env.reset()
    agent.perceive()
    agent.last_action_p = 0.0  # Explicitly set baseline expectancy to 0

    for _ in range(num_throws):
        
        # 1. Baseline (Before throw)
        total, ad, aas = agent.get_affect()
        
        # 2. THE THROW (Internal logic: 0 -> 1 -> Outcome)
        env.step(1) # s_start -> s_throw
        agent.decide() # s_throw -> Outcome + Perceive
        
        # 3. Log ONLY the outcome
        res_total, res_ad, res_aas = agent.get_affect()
        all_data.append({
            # np.nan so the "pre-roll" state doesn't plot a dot at 0
            "outcomes": [np.nan, env.get_features()["goal_dim"]],
            "affect": [
                {"total": total, "ad": ad, "aas": aas},
                {"total": res_total, "ad": res_ad, "aas": res_aas}
            ]
        })

        # Reset env for next throw but keep agent's internal "readiness"
        env.reset() 
        agent.perceive()
    return all_data
    
    # Each throw is an episode. Max steps is small because Dice is a short cycle.
    #return run_simulation(env, agent, num_episodes=num_throws, max_steps=2)

# Corridor Run Function
def run_corridor_task(mode="gradual", agent_beliefs="accurate", seed=None, length=6, trap_prob=0.1):
    env = Corridor(length=length, trap_prob=trap_prob, seed=seed)
    targets = {"goal_dim": float(length)}
    
    # --- 1. Perception Lens (for AD) ---
    def perception_lens(raw):
        if mode == "binary":
            val = float(length) if raw["goal_dim"] == float(length) else 0.0 
            return {"goal_dim": val}
        return raw # Gradual

    # --- 2. Expectancy Lens (for AAS) ---
    def expectancy_lens(e):
        if agent_beliefs == "oblivious":
            return [1.0] # Flat expectancy 
        else: # accurate
            # Formula: p_ij = (p')^n_steps [cite: 553]
            p_prime = 1.0 - e.trap_prob
            steps_to_go = e.length - e.state
            return [p_prime ** steps_to_go]

    affect_model = AffectModel(v=1.0, targets=targets, w1=1.0, w2=1.0)
    agent = Agent(env, affect_model, targets, 
                  perception_filter=perception_lens, 
                  expectancy_filter=expectancy_lens)
    
    all_data = []
    env.reset()
    
    episode_log = {"outcomes": [], "affect": []}

    # 1. PERCEIVE current state 
    agent.perceive() 
    agent.last_action_p = agent.beliefs[0] if isinstance(agent.beliefs, list) else agent.beliefs

    while True:
        # 1. PERCEIVE current state 
        #agent.perceive() 
        
        # 2. RECORD affect for this state
        total, ad, aas = agent.get_affect()
        episode_log["affect"].append({"total": total, "ad": ad, "aas": aas})
        episode_log["outcomes"].append(env.get_features()["goal_dim"])
        
        # 3. CHECK if we just landed in a terminal state (Goal or Trap)
        if env.is_terminal():
            break
            
        # 4. DECIDE / ACT (Move one step forward)
        agent.decide()
        agent.perceive()
        
    all_data.append(episode_log)
    return all_data
    
    #return run_simulation(env, agent, num_episodes=1, max_steps=length+2)

# --------- PLOTTING ------------

# Dice
def plot_affectvsoutcome_dice(all_data, mode=None, seed=None, ylabel="Outcome Value", target_val=6, num_throws=10):
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

    # --- AXIS 1: Outcomes (Left) ---
    y1_min, y1_max = -1, 8  # Standard range for a 6-sided die

    offset = target_val
    y2_min, y2_max = y1_min - offset, y1_max - offset # Results in [-7, 2]

    ax1.set_ylim(y1_min, y1_max)
    ax1.set_xlabel('Steps (Sequential Episodes)')
    ax1.set_ylabel(ylabel, color='gray')
    # ax1.set_xticks(np.arange(0, len(flat_outcomes), 2))
    # ax1.set_xticklabels([f"Trial {i+1}" for i in range(num_throws)])
    # ax1.set_xlabel("Experimental Trials")
    
    # Plotting Outcomes
    ax1.scatter(range(len(flat_outcomes)), flat_outcomes, color='gray', alpha=0.3, label="Actual Outcome")
    ax1.axhline(y=target_val, color='green', linestyle='--', alpha=0.5, label=f"Goal ({target_val})")
    
    for sep in episode_separators[:-1]:
        ax1.axvline(x=sep, color='black', linestyle='-', alpha=0.1)

    # --- AXIS 2: Affect (Right) ---
    ax2 = ax1.twinx()
    
    # --- ALIGNMENT LOGIC ---
    # Since Affect = Value - Target (6), we shift the limits by exactly the target_val
    y2_min, y2_max = y1_min - target_val, y1_max - target_val # Results in [-6, 1]
    ax2.set_ylim(y2_min, y2_max)
    
    ax2.set_ylabel('Affective Intensity')
    ax2.plot(flat_total, color='blue', linewidth=2, label="Total Affect")
    ax2.plot(flat_ad, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
    ax2.plot(flat_aas, color='orange', linestyle='-', alpha=0.6, label="AAS (Expectancy)")
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax1.set_title(f"GDT Model: {mode.upper()} Dice (Seed {seed} )", 
                     fontsize=14, pad=15)

    #plt.title(title)
    plt.tight_layout()
    plt.show()


# Corridor
def plot_affectvsoutcome_corridor(all_data, mode="gradual", ylabel="Outcome Value", length=6, target_val=6):
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

    #plt.title(f"{title} | Mode: {mode.upper()}")
    plt.tight_layout()
    plt.show()

def plot_stacked_corridor(all_walks_data, seeds, mode="binary", agent_type="oblivious"):
    # 1. Significantly taller figure (8 wide, 20 tall)
    fig, axes = plt.subplots(len(all_walks_data), 1, figsize=(8, 12))
    
    for i, (ep, seed) in enumerate(zip(all_walks_data, seeds)):
        ax1 = axes[i]
        
        # Data Extraction
        outcomes = ep["outcomes"]
        total = [step["total"] for step in ep["affect"]]
        ad = [step["ad"] for step in ep["affect"]]
        aas = [step["aas"] for step in ep["affect"]]
        steps = np.arange(len(outcomes))

        # --- Axis 1: Position ---
        ax1.set_ylim(-4, 8) 
        # 2. Hard-limit the X-axis to the actual corridor length (7 steps)
        ax1.set_xlim(-0.5, 7.5) 
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.fill_between([-0.5, 7.5], -4, 0, color='gray', alpha=0.1, label="Failure Zone")
        
        # Path and Dots
        ax1.step(steps, outcomes, where='post', color='gray', alpha=0.5, linewidth=2, label="Agent Path")
        ax1.scatter(steps, outcomes, color='gray', s=40, alpha=0.6)
        
        # Goal Line
        ax1.axhline(y=6.0, color='green', linestyle='--', alpha=0.5, label="Goal (6)")
        ax1.set_ylabel(f"Seed {seed}\nPos", color='gray', fontsize=12)

        # --- Axis 2: Affect (ALIGNED -6 to 0) ---
        ax2 = ax1.twinx()
        ax2.set_ylim(-10, 2)
        ax2.set_ylabel('Affective Intensity', fontsize=12)

        ax2.plot(total, color='blue', linewidth=2.5, label="Total Affect")
        ax2.plot(ad, color='red', linestyle=':', alpha=0.7, label="AD")
        ax2.plot(aas, color='orange', linestyle='-', alpha=0.7, label="AAS")
        
        # --- Titles and Legend ---
        ax1.set_title(f"GDT Model: {mode.upper()} Corridor ({agent_type.capitalize()} Agent)", 
                     fontsize=14, pad=15)
        
        if i == 0:
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Legend size and placement
            ax2.legend(lines + lines2, labels + labels2, loc='upper right', 
                       fontsize='small', ncol=2, frameon=True)

    # 3. Manual spacing adjustment to prevent vertical overlap
    plt.subplots_adjust(hspace=0.4) 
    plt.xlabel("Time Steps", fontsize=12)
    filename = f"Corridor_{mode.capitalize()}_{agent_type.capitalize()}_Stack.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
dice_bin = run_dice_task(mode="binary", seed=42)
dice_grad = run_dice_task(mode="gradual", seed=42)
plot_affectvsoutcome_dice(dice_bin, mode="binary", seed=42, ylabel="Outcome Value", target_val=6)
plot_affectvsoutcome_dice(dice_grad, mode="gradual", seed=42, ylabel="Outcome Value", target_val=6)

# # --- EXPERIMENT 2: DOORS (Probability Flip) ---
# doors_data = run_doors_task(agent_knows_flip=False)
# plot_affectvsoutcome(doors_data, title="Exp 3: Doors (Hidden Flip)", ylabel="Door Choice")

# # --- EXPERIMENT 3: CORRIDOR POLICY COMPARISON ---

# Choose 6 seeds. 
# seeds = [42, 7, 10, 15, 21, 99] 
seeds = [15, 42, 7]
belief_modes = ["oblivious", "accurate"]

# --- BINARY CORRIDOR ---
print("Simulating Binary Corridor (Oblivious vs. Accurate)...")
for belief in belief_modes:
    print(f"  Mode: Binary | Beliefs: {belief}")
    for i, s in enumerate(seeds):
        # Pass the belief mode to your task runner
        single_walk_data = run_corridor_task(mode="binary", agent_beliefs=belief, seed=s)
        
        plot_affectvsoutcome_corridor(
            single_walk_data, 
            mode="binary", 
            #title=f"Walk {i+1} (Seed: {s}) | Agent: {belief.capitalize()}", 
            ylabel="Position (0-6)", 
            target_val=6.0
        )

# --- GRADUAL CORRIDOR ---
print("Simulating Gradual Corridor (Oblivious vs. Accurate)...")
for belief in belief_modes:
    print(f"  Mode: Gradual | Beliefs: {belief}")
    for i, s in enumerate(seeds):
        single_walk_data = run_corridor_task(mode="gradual", agent_beliefs=belief, seed=s)
        
        plot_affectvsoutcome_corridor(
            single_walk_data, 
            mode="gradual", 
            #title=f"Walk {i+1} (Seed: {s}) | Agent: {belief.capitalize()}", 
            ylabel="Position (0-6)", 
            target_val=6.0
        )

# --- stacked version ----
# Define the sets of seeds/conditions used in the paper
paper_groups = {
    "Binary-Oblivious (Figs 6-8)": {"mode": "binary", "beliefs": "oblivious", "seeds": [15, 42, 7]},
    "Binary-Accurate (Figs 9-11)": {"mode": "binary", "beliefs": "accurate", "seeds": [15, 42, 7]},
    "Gradual-Oblivious (Figs 12-14)": {"mode": "gradual", "beliefs": "oblivious", "seeds": [15, 42, 7]},
    "Gradual-Accurate (Figs 15-17)": {"mode": "gradual", "beliefs": "accurate", "seeds": [15, 42, 7]}
}

for label, config in paper_groups.items():
    print(f"Simulating {label}...")
    group_data = []
    
    for s in config["seeds"]:
        # Run simulation for this specific configuration
        walk_data = run_corridor_task(mode=config["mode"], 
                                      agent_beliefs=config["beliefs"], 
                                      seed=s)
        group_data.append(walk_data[0])
    
    # Plot the 3-stack for this condition
    plot_stacked_corridor(group_data, 
                          config["seeds"], 
                          mode=config["mode"], 
                          agent_type=config["beliefs"])

print("Finished all simulations!")
