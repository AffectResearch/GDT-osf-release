import matplotlib.pyplot as plt

from envs.dice import Dice
from agent.agent import Agent
from agent.affect import AffectModel

def run (
        trials = 100,
        v = 1.0,
        w1 = 1.0,
        w2 = 1.0
):
    env = Dice(sides = 6, seed=42)

    goal_targets = {"feature value": 6}
    affect_model = AffectModel(
        v= v,
        targets = goal_targets,
        expectancy = env.transition_prob(),
        w1=w1,
        w2=w2
    )

    agent = Agent(env, affect_model)
    outcome_log = []
    affect_log = []
    results = {}

    for i in range(trials):
        agent.decide()
        aff_comp, ad, aas = agent.get_affect()
        affect_log.append({
            "total": aff_comp,
            "ad": ad,
            "aas": aas
        })
        outcome_log.append(agent.current_state)
        env.reset()
    results = {"rolls": outcome_log, "affect_scores": affect_log}

    return results



def plot_affectvsoutcome(data):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plotting the Dice Rolls
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Dice Roll', color='gray')
    ax1.scatter(range(len(data["rolls"])), data["rolls"], color='gray', alpha=0.3, label="Dice Roll")
    ax1.axhline(y=6, color='green', linestyle='--', alpha=0.5, label="Target (6)")
    ax1.set_ylim(0, 7)

    # Creating a second axis for Affect
    ax2 = ax1.twinx()
    ax2.set_ylabel('Affective Intensity')

    # Extracting the scores
    total_scores = [entry["total"] for entry in data["affect_scores"]]
    ad_scores = [entry["ad"] for entry in data["affect_scores"]]
    aas_scores = [entry["aas"] for entry in data["affect_scores"]]

    # Plotting the three traces
    ax2.plot(total_scores, color='blue', linewidth=2, label="Total Affect (A_it)")
    ax2.plot(ad_scores, color='red', linestyle=':', alpha=0.6, label="AD (Discrepancy)")
    ax2.plot(aas_scores, color='orange', linestyle='-', alpha=0.6, label="AAS (Action Selection)")

    # Adding a legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title("GDT Model: Affective Decomposition (Dice Task)")
    plt.tight_layout()
    plt.show()

# Run and Plot
simulation_data = run(trials=100,v = 1.0, w1 = 1.0, w2 = 1.0)
plot_affectvsoutcome(simulation_data)

