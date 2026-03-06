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
        act = agent.decide()
        aff_comp, ad, aas = agent.get_affect()
        affect_log.append({
            "total": aff_comp,
            "ad": ad,
            "aas": aas
        })
        outcome_log.append(act)
        env.reset()
    results = {"rolls": outcome_log, "affect_scores": affect_log}

    return results



def plot_affectvsoutcome(data):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Dice Roll', color='tab:red')
    ax1.scatter(range(len(data["rolls"])), data["rolls"], color='tab:red', alpha=0.6, label="Roll")
    ax1.axhline(y=6, color='green', linestyle='--', label="Target (6)") # Updated to match goal_targets
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Affective State', color='tab:blue')
    total_scores = [entry["total"] for entry in data["affect_scores"]]
    ax2.plot(total_scores, color='tab:blue', linewidth=2, label="Affect")
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Agent Affective Response Over Time")
    fig.tight_layout()
    plt.show()

# Run and Plot
simulation_data = run(trials=100,v = 1.0, w1 = 1.0, w2 = 1.0)
plot_affectvsoutcome(simulation_data)

