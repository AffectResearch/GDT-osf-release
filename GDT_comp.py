# my_skeleton_with_affect_classes.py
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------
# Environment
# ----------------------------
class Env:
    def __init__(self, p_left=0.6, satiation_mode="binary", sat_gain=1.0, sat_decay=0.0):
        self.p_left = p_left
        self.satiation_mode = satiation_mode
        self.sat_gain = sat_gain
        self.sat_decay = sat_decay
        self.reset()

    def reset(self):
        self.satiation = 0.0  # hungry
        self.t = 0
        return self.state()

    def state(self):
        if self.satiation_mode == "binary":
            return int(self.satiation > 0.0)  # 0=hunger, 1=full
        else:
            return self.satiation

    def step(self, response):
        self.t += 1
        if response == 0:  # Left
            outcome = 1 if random.random() < self.p_left else 0
        else:  # Right
            outcome = 1 if random.random() < (1 - self.p_left) else 0

        # satiation dynamics
        if self.satiation_mode == "binary":
            if outcome == 1:
                self.satiation = 1.0
        else:
            self.satiation = max(0.0, self.satiation - self.sat_decay)  # decay
            self.satiation = min(1.0, self.satiation + outcome * self.sat_gain)

        done = (self.satiation_mode == "binary" and self.satiation == 1.0)
        return self.state(), outcome, done


# ----------------------------
# Agent
# ----------------------------
class Agent:
    def __init__(self, n_responses=2, alpha=None, policy="epsilon_greedy", epsilon=0.1):
        self.n_responses = n_responses
        self.alpha = alpha
        self.policy = policy
        self.epsilon = epsilon

        # initialize estimates
        self.success = [1] * n_responses
        self.total = [2] * n_responses
        self.p_est = [0.5] * n_responses

    def select_response(self):
        if self.policy == "random":
            return random.randrange(self.n_responses)
        elif self.policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return random.randrange(self.n_responses)
            else:
                return int(np.argmax(self.p_est))
        else:
            raise ValueError(f"Unknown policy {self.policy}")

    def update(self, response, outcome):
        #this needs to change completely
        if self.alpha is None:
            self.total[response] += 1
            self.success[response] += outcome
            self.p_est[response] = self.success[response] / self.total[response]
        else:
            self.p_est[response] = (1 - self.alpha) * self.p_est[response] + self.alpha * outcome


# ----------------------------
# Affect Models
# ----------------------------
class AffectModel: 
    def __init__(self, goal=1.0, beta1=1.0, beta2=-1, gamma=1.0, Sa2 = 0.3, pi = 0.5):
        self.goal = goal
        self.beta1 = beta1 #baseline affect for match
        self.beta2 = beta2 #baseline affect for no match
        self.gamma = gamma #goal importance
        self.Sa2 = Sa2 #salience of the anticipated state
        self.pi = pi #assigned subjective probability of the anticipated state

    def calculate_A(self, state):
        if state == self.goal:
            A = self.beta1 * self.gamma
        else:
            A = self.beta2 * self.gamma
        return A
    
    def calculate_AAI(self, next_state):
        A_next = self.calculate_A(next_state)
        AAI = A_next * self.pi * self.Sa2
        return AAI
    def calculate(self, state, response, agent, next_state):
        #this gets overwritten by the subclasses
        raise NotImplementedError

#subclass 1 
class Affect(AffectModel):
    def calculate(self, state, response, agent, next_state):
        A = self.calculate_A(state)
        return A, 0.0, A

#subclass 2
class AnticipatedAffect(AffectModel):


    
    def calculate(self, state, response, agent, next_state):
        A = self.calculate_A(state)
        AAI = self.calculate_AAI(next_state)
        return A, AAI, A + AAI



# ----------------------------
# Runner
# ----------------------------
class Runner:
    def __init__(self, env, agent, affect_model, n_episodes=50, max_steps=20, seed=None):
        self.env = env
        self.agent = agent
        self.affect_model = affect_model
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def run(self):
        logs = {
            "responses": [],
            "outcomes": [],
            "states": [],
            "p_est": [],
            "A": [],
            "AAI": [],
            "A_total": [],
            "n_episodes": self.n_episodes,
        }

        for ep in range(self.n_episodes):
            ep_responses, ep_outcomes, ep_states = [], [], []
            ep_Ac, ep_Aant, ep_Atot = [], [], []
            ep_p_est = []

            state = self.env.reset()
            for step in range(self.max_steps):
                response = self.agent.select_response()
                next_state, outcome, done = self.env.step(response)

                self.agent.update(response, outcome)

                # affect via chosen model
                A_cur, A_ant, A_tot = self.affect_model.calculate(state, response, self.agent, next_state)

                ep_responses.append(response)
                ep_outcomes.append(outcome)
                ep_states.append(next_state)
                ep_Ac.append(A_cur)
                ep_Aant.append(A_ant)
                ep_Atot.append(A_tot)
                ep_p_est.append(self.agent.p_est.copy())

                state = next_state
                if done:
                    break

            logs["responses"].append(ep_responses)
            logs["outcomes"].append(ep_outcomes)
            logs["states"].append(ep_states)
            logs["A"].append(ep_Ac)
            logs["AAI"].append(ep_Aant)
            logs["A_total"].append(ep_Atot)
            logs["p_est"].append(ep_p_est)

        return logs


# ----------------------------
# Plotting function
# ----------------------------
def plot_case(logs, true_p_left=0.6, case_name="Case"):
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))

    
    # (a) Learning curve = mean outcome per episode
    # plot average outcome(amount of food/reward) per episode
    # shows whether the agent is learning to pick the better door
    avg_outcomes = [np.mean(ep) if len(ep)>0 else 0 for ep in logs["outcomes"]]
    axes[0,0].plot(avg_outcomes)
    axes[0,0].set_title("Average outcome per episode")

    # (b) Probability estimates (last p_est each episode)
    # compare how quickly estimates converge to true probabilities under different learning rules
    left_hat = [ep[-1][0] for ep in logs["p_est"] if len(ep)>0]
    right_hat = [ep[-1][1] for ep in logs["p_est"] if len(ep)>0]
    axes[0,1].plot(left_hat, label="Left")
    axes[0,1].plot(right_hat, label="Right")
    axes[0,1].axhline(true_p_left, color="gray", ls="--", label="True P(left)")
    axes[0,1].axhline(1-true_p_left, color="gray", ls=":", label="True P(right)")
    axes[0,1].set_title("Probability learning")
    axes[0,1].legend()

    # (c) Response distribution (fraction left per episode)
    # shows when the policy stabilizes
    frac_left = [(np.array(ep)==0).mean() if len(ep)>0 else 0 for ep in logs["responses"]]
    axes[0,2].plot(frac_left)
    axes[0,2].set_ylim(0,1)
    axes[0,2].set_title("Fraction left per episode")

    # (d) Satiation (plot first episode only, for clarity)
    if len(logs["states"]) > 0:
        axes[1,0].plot(logs["states"][0])
    axes[1,0].set_title("Satiation dynamics (ep 0)")

    # (e) Affect components (first episode) x axis is timestep within epsiode
    if len(logs["A"]) > 0:
        axes[1,1].plot(logs["A"][0], label="Current")
        axes[1,1].plot(logs["AAI"][0], label="Anticipated")
        axes[1,1].plot(logs["A_total"][0], label="Total")
    axes[1,1].set_title("Affect (ep 0)")
    axes[1,1].legend()

    # (f) Histogram of outcomes (all episodes)
    all_outcomes = [o for ep in logs["outcomes"] for o in ep]
    axes[1,2].hist(all_outcomes, bins=2)
    axes[1,2].set_xticks([0,1])
    axes[1,2].set_title("Outcome histogram")
    # (g) Average affect per episode
    #see trends in how current, anticipated, and total affect evolve across learning.
    avg_A = [np.mean(ep) if len(ep)>0 else 0 for ep in logs["A"]]
    avg_AAI = [np.mean(ep) if len(ep)>0 else 0 for ep in logs["AAI"]]
    avg_Atot = [np.mean(ep) if len(ep)>0 else 0 for ep in logs["A_total"]]
    axes[2,0].plot(avg_A, label="Current")
    axes[2,0].plot(avg_AAI, label="Anticipated")
    axes[2,0].plot(avg_Atot, label="Total")
    axes[2,0].set_title("Average affect per episode")
    axes[2,0].legend()

    # (h) Affect trajectory across episodes (heatmap)
    #each row = episode, each col = timestep. Lets you spot patterns (e.g., anticipated affect shrinking as agent learns).
    if len(logs["A_total"]) > 0:
        max_len = max(len(ep) for ep in logs["A_total"])
        mat = np.full((len(logs["A_total"]), max_len), np.nan)
        for i, ep in enumerate(logs["A_total"]):
            mat[i, :len(ep)] = ep
        im = axes[2,1].imshow(mat, aspect="auto", cmap="coolwarm")
        axes[2,1].set_title("A_total heatmap (episodes x steps)")
        fig.colorbar(im, ax=axes[2,1])

    # (i) Distribution of total affect values
    #histogram of all affect values, shows whether it skews positive/negative across the task.
    all_Atot = [a for ep in logs["A_total"] for a in ep]
    axes[2,2].hist(all_Atot, bins=20, color="purple", alpha=0.7)
    axes[2,2].set_title("Distribution of A_total")

    fig.suptitle(case_name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Case 1: binary satiation, random agent, current-only affect
    env1 = Env(p_left=0.6, satiation_mode="binary")
    agent1 = Agent(alpha=None, policy="random")
    affect1 = Affect()
    runner1 = Runner(env1, agent1, affect1, n_episodes=30, max_steps=10)
    logs1 = runner1.run()
    plot_case(logs1, true_p_left=0.6, case_name="Case 1: Binary + Current Affect")

    # Case 2: binary satiation, random agent, learnd anticipated affect
    env2 = Env(p_left=0.6, satiation_mode="binary")
    agent2 = Agent(alpha=None, policy="random")
    affect2 = AnticipatedAffect(beta1=1.0, beta2=0.5)
    runner2 = Runner(env2, agent2, affect2, n_episodes=30, max_steps=10)
    logs2 = runner2.run()
    plot_case(logs2, true_p_left=0.6, case_name="Case 2: Binary + Anticipated Affect")

    # Case 3: partial satiation + no learning
    env3 = Env(p_left=0.6, satiation_mode="partial", sat_gain=0.5, sat_decay=0.05)
    runner3 = Runner(env3, agent2, affect2, n_episodes=30, max_steps=10)
    logs3 = runner3.run()
    plot_case(logs3, true_p_left=0.6, case_name="Case 3: partial satiation + no learning")
