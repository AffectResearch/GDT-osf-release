 # gdt_modular/run.py
from .envs.money_env import MoneyMDPEnv, money_discrepancy
from .agent.agent import Agent
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

        # collect affect components at current state (before stepping)
        s0 = self.salience_manager.retrieve("goal_completion")
        s1 = self.salience_manager.retrieve("discrepancy")
        s2 = self.salience_manager.retrieve("delta_discrepancy")
        delta_d = self.affect_model.delta_discrepancy(self.prev_d, d)
        a0 = self.affect_model.affect_from_completion(s0)
        a_discr = self.affect_model.affect_discrepancy(d, s1)
        a_delta = self.affect_model.affect_delta_discrepancy(delta_d, s2)

        state_vis = self.state
        if hasattr(self.env, "decode_state"):
            try:
                cell, _bits = self.env.decode_state(self.state)
                state_vis = cell
            except Exception:
                pass

        self.metrics.append({
            "t": self.t,
            "state": self.state,
            "discrepancy": d,
            "delta_discrepancy": delta_d,
            "affect_total": affect,
            "affect_completion": a0,
            "affect_discrepancy": a_discr,
            "affect_delta_discrepancy": a_delta,
            "action": action,
            "state_vis": state_vis,
        })
        self.t += 1

        new_state, reward, done = self.env.step(action)
        self.memory.store(self.state, action, new_state)

        self.prev_d = d
        self.state = new_state

        if done:
            print("Reached goal. Episode finished.")
            # add a terminal-row sample (after reaching goal)
            d_final = self.affect_model.discrepancy(self.state, self.goal)  # should be 0
            delta_final = self.affect_model.delta_discrepancy(self.prev_d, d_final)
            a0f = self.affect_model.affect_from_completion(self.salience_manager.retrieve("goal_completion"))
            a_discr_f = self.affect_model.affect_discrepancy(d_final, self.salience_manager.retrieve("discrepancy"))
            a_delta_f = self.affect_model.affect_delta_discrepancy(delta_final, self.salience_manager.retrieve("delta_discrepancy"))
            a_total_f = self.affect_model.combined_affect(d_final, delta_final,
                                                        self.salience_manager.retrieve("goal_completion"),
                                                        self.salience_manager.retrieve("discrepancy"),
                                                        self.salience_manager.retrieve("delta_discrepancy"))




            self.metrics.append({
                "t": self.t,
                "state": self.state,
                "discrepancy": d_final,
                "delta_discrepancy": delta_final,
                "affect_total": a_total_f,
                "affect_completion": a0f,
                "affect_discrepancy": a_discr_f,
                "affect_delta_discrepancy": a_delta_f,
                "action": "terminal",
                "state_vis": state_vis,
            })

            # make the figure
            #plot_affect_timeseries(self.metrics)
            #plot_trajectory(self.metrics, self.env.max_state)
            #plot_dashboard(self.metrics, self.env.max_state)
            # pick a generic y-range for plotting:
            # use decoded cell if available; otherwise use the raw state
            if hasattr(self, "metrics") and self.metrics:
                state_vis_values = []
                for r in self.metrics:
                    sv = r.get("state")
                    if hasattr(self.env, "decode_state"):
                        try:
                            cell, _bits = self.env.decode_state(r["state"])
                            sv = cell
                        except Exception:
                            pass
                    state_vis_values.append(r.get("state_vis", r["state"]))
                max_state_for_plot = max(state_vis_values) if state_vis_values else 1
            else:
                max_state_for_plot = 1

            plot_combined_affect_and_trajectory(self.metrics, max_state_for_plot, show_components=True)


            break

        time.sleep(getattr(self, "sleep_s", 0.0))









# ---- Visualizations ----
# Useful Visualizations for understanding affect dynamics
# Affect time-series plot
def plot_affect_timeseries(rows, title="Affect components over time (MoneyMDP)"):
    ts = [r["t"] for r in rows]
    total = [r["affect_total"] for r in rows]
    a0 = [r["affect_completion"] for r in rows]
    a_d = [r["affect_discrepancy"] for r in rows]
    a_dd = [r["affect_delta_discrepancy"] for r in rows]

    plt.figure()
    plt.plot(ts, total, label="Total affect")
    plt.plot(ts, a0,    label="Affect: completion (a0)")
    plt.plot(ts, a_d,   label="Affect: discrepancy (− s1·d)")
    plt.plot(ts, a_dd,  label="Affect: Δdiscrepancy (s2·Δd)")
    plt.xlabel("Time step")
    plt.ylabel("Affect")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Trajectory plot
def plot_trajectory(rows, max_state, title="State trajectory over time (MoneyMDP)"):
    ts = [r["t"] for r in rows]
    states = [r["state"] for r in rows]
    actions = [r["action"] for r in rows]

    plt.figure()
    plt.step(ts, states, where="post")
    plt.xlabel("Time step")
    plt.ylabel("State")
    plt.yticks(range(0, max_state + 1))
    plt.title(title)

    # One marker per action type (legend shows each once)
    seen = set()
    for t, s, a in zip(ts, states, actions):
        if a not in seen and a in ("go", "back", "stay"):
            plt.scatter([t], [s], label=a)
            seen.add(a)
        else:
            if a in ("go", "back", "stay"):
                plt.scatter([t], [s])
    plt.legend()
    plt.tight_layout()
    plt.show()

# Dahsboard
def plot_dashboard(rows, max_state):
    # prepare series
    ts = [r["t"] for r in rows]
    total = [r["affect_total"] for r in rows]
    a0 = [r["affect_completion"] for r in rows]
    a_d = [r["affect_discrepancy"] for r in rows]
    a_dd = [r["affect_delta_discrepancy"] for r in rows]
    states = [r.get("state_vis", r["state"]) for r in rows]
    actions = [r["action"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: affect components
    ax1.plot(ts, total, label="Total affect")
    ax1.plot(ts, a0,    label="Affect: completion (a0)")
    ax1.plot(ts, a_d,   label="Affect: discrepancy (− s1·d)")
    ax1.plot(ts, a_dd,  label="Affect: Δdiscrepancy (s2·Δd)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Affect")
    ax1.set_title("Affect components over time")
    ax1.legend()

    # Right: trajectory
    ax2.step(ts, states, where="post")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("State")
    ax2.set_yticks(list(range(0, max_state + 1)))
    ax2.set_title("State trajectory")

    # one marker per action type for legend
    seen = set()
    for t, s, a in zip(ts, states, actions):
        if a in ("go", "back", "stay"):
            if a not in seen:
                ax2.scatter([t], [s], label=a)
                seen.add(a)
            else:
                ax2.scatter([t], [s])
    ax2.legend()

    plt.show()


# Combined figure
def plot_combined_affect_and_trajectory(rows, max_state, show_components=True):
    # series
    ts = [r["t"] for r in rows]
    total = [r["affect_total"] for r in rows]
    a0 = [r["affect_completion"] for r in rows]
    a_d = [r["affect_discrepancy"] for r in rows]
    a_dd = [r["affect_delta_discrepancy"] for r in rows]
    # change to this
    states = [r.get("state_vis", r["state"]) for r in rows]
    actions = [r["action"] for r in rows]

    # colors for background shading
    bg = {"go": "#d9fdd3", "stay": "#fff5cc", "back": "#ffd6d6"}  # green/yellow/red
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # background per step (span t -> t+1)
    for i, (t, a) in enumerate(zip(ts, actions)):
        t0 = t
        t1 = ts[i+1] if i + 1 < len(ts) else t + 1
        ax1.axvspan(t0, t1, color=bg.get(a, "#eeeeee"), alpha=0.6, zorder=0)

    # affect dynamics (left axis)
    line_total, = ax1.plot(ts, total, linewidth=2, label="Total affect")
    lines = [line_total]
    labels = ["Total affect"]
    if show_components:
        l1, = ax1.plot(ts, a0,  linewidth=1.5, label="Affect: completion (a0)")
        l2, = ax1.plot(ts, a_d, linewidth=1.5, label="Affect: discrepancy (− s1·d)")
        l3, = ax1.plot(ts, a_dd, linewidth=1.5, label="Affect: Δdiscrepancy (s2·Δd)")
        lines += [l1, l2, l3]
        labels += ["Affect: completion (a0)", "Affect: discrepancy (− s1·d)", "Affect: Δdiscrepancy (s2·Δd)"]
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Affect")
    ax1.set_title("Affect & Trajectory (background shaded by action)")

    # trajectory (right axis)
    ax2 = ax1.twinx()
    l_state, = ax2.step(ts, states, where="post", linewidth=2, label="State")
    ax2.set_ylabel("State")
    ax2.set_ylim(-0.5, max_state + 0.5)
    ax2.set_yticks(list(range(0, max_state + 1)))

    # legends (lines + background colors)
    action_patches = [
        Patch(facecolor=bg["go"],  alpha=0.6, label="go"),
        Patch(facecolor=bg["stay"], alpha=0.6, label="stay"),
        Patch(facecolor=bg["back"], alpha=0.6, label="back"),
    ]
    lines += [l_state]
    labels += ["State"]
    first_legend = ax1.legend(lines, labels, loc="upper right")
    ax1.add_artist(first_legend)  # keep it when adding second legend
    ax1.legend(handles=action_patches, title="Action (background)", loc="upper left")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from gdt_modular.envs.money_env import MoneyMDPEnv, money_discrepancy
    env = MoneyMDPEnv(max_state=5)
    agent = Agent(env=env, goal=5, planning=False, discrepancy_fn=money_discrepancy)
    agent.run()