"""Helper script to generate test_pipeline.ipynb as valid notebook JSON."""
import json, os

cells = []

def code_cell(source, cell_id=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True),
    }

# ── Cell 0: Title ──
cells.append(md_cell(
"# Among Us RL Pipeline — Integration Test\n"
"\n"
"Integrates **Actor** (Theory of Mind), **Critic** (State Value), and **Reward** (Dopamine) modules\n"
"into a single simulation loop with a Mock LLM (no API keys needed).\n"
))

# ── Cell 1: Imports & Setup ──
cells.append(code_cell("""\
import sys, os, random, json
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the agent package is importable
sys.path.insert(0, os.path.abspath(".."))

from amongagents.agent.actor import ActorModule
from amongagents.agent.critic import CriticModule
from amongagents.agent.rewards import RewardEngine

random.seed(42)
print("✅ All modules imported successfully.")
"""))

# ── Cell 2: Mock Infrastructure ──
cells.append(code_cell("""\
# ═══════════════════════════════════════════════════════════════
# Mock Infrastructure — no API keys required
# ═══════════════════════════════════════════════════════════════

ROOMS = ["Cafeteria", "Admin", "Electrical", "Medbay", "Reactor",
         "Navigation", "Security", "Storage", "Weapons", "Shields"]

ACTIONS_CREW = ["MOVE", "COMPLETE_TASK", "STAY"]
ACTIONS_IMP  = ["MOVE", "KILL", "FAKE_TASK", "VENT", "STAY"]


class MockPlayer:
    \"\"\"Lightweight player for the simulation.\"\"\"
    def __init__(self, name, role, color, location="Cafeteria"):
        self.name = f"{name}: {color}"
        self.identity = role
        self.role = role
        self.team = role
        self.color = color
        self.location = location
        self.alive = True
        self.observation_history = []
        self.verified_presence_log = []
        self.tasks_done = 0
        self.total_tasks = 3


class MockGameState:
    \"\"\"Mutable game state tracking.\"\"\"
    def __init__(self, players):
        self.players = players
        self.task_progress = 0.0   # 0-100
        self.game_over = False
        self.winner = None
        self.sabotage_active = False
        self.turn = 0

    def to_dict(self):
        alive_crew = [p for p in self.players
                      if p.alive and p.role == "Crewmate"]
        alive_imps = [p for p in self.players
                      if p.alive and p.role == "Impostor"]
        return {
            "living_crewmates": len(alive_crew),
            "living_impostors": len(alive_imps),
            "task_completion_pct": self.task_progress,
            "sabotage_active": self.sabotage_active,
            "ejected_roles": [],
            "winner": self.winner,
        }

    def visible_players(self, player):
        \"\"\"Return names of alive players in the same room.\"\"\"
        return [p.name for p in self.players
                if p.alive and p.name != player.name
                and p.location == player.location]


print("✅ Mock infrastructure ready.")
"""))

# ── Cell 3: Simulation Loop ──
cells.append(code_cell("""\
# ═══════════════════════════════════════════════════════════════
# Simulation Loop
# ═══════════════════════════════════════════════════════════════

def run_simulation(num_turns=10):
    \"\"\"Run a full mock game and collect per-turn metrics.\"\"\"

    # --- Initialise players ---
    red    = MockPlayer("Player 1", "Impostor", "red")
    blue   = MockPlayer("Player 2", "Crewmate", "blue")
    green  = MockPlayer("Player 3", "Crewmate", "green")
    yellow = MockPlayer("Player 4", "Crewmate", "yellow")
    all_players = [red, blue, green, yellow]

    gs = MockGameState(all_players)

    # --- Initialise modules ---
    actors  = {p.name: ActorModule(p, all_players) for p in all_players}
    critic  = CriticModule()
    reward  = RewardEngine()

    log_rows = []

    for t in range(num_turns):
        gs.turn = t

        for player in all_players:
            if not player.alive:
                continue

            # ── Step A: Observation (synthetic) ──
            player.location = random.choice(ROOMS)
            visible = gs.visible_players(player)
            player.verified_presence_log.append({
                "timestep": t,
                "room": player.location,
                "players_seen": visible,
            })

            # ── Step B: Belief Update ──
            obs_log = []
            # Simulate random observations
            for other in all_players:
                if other.name == player.name or not other.alive:
                    continue
                # Small chance of witnessing something suspicious
                if random.random() < 0.15 and other.role == "Impostor":
                    obs_log.append({
                        "subject": other.name,
                        "action": random.choice(["FAKE_TASK", "SABOTAGE"]),
                        "witnesses": [player.name],
                    })
                elif random.random() < 0.1 and other.role == "Crewmate":
                    obs_log.append({
                        "subject": other.name,
                        "action": "COMPLETE_TASK",
                        "witnesses": [player.name],
                    })
            actors[player.name].update_beliefs(obs_log)

            # ── Step C: Action (from actor module) ──
            game_dict = {
                "visible_players": visible,
                "dead_bodies": [],
            }
            result = actors[player.name].generate_actor_step(
                game_dict, [f"Turn {t}"]
            )
            action = result.get("action_type", "STAY")

            # ── Step D: Apply Action to Game State ──
            action_log = {"action": action, "witnesses": [], "target": None}

            if action == "KILL" and player.role == "Impostor":
                crew_alive = [p for p in all_players
                              if p.alive and p.role == "Crewmate"
                              and p.location == player.location]
                if crew_alive:
                    victim = random.choice(crew_alive)
                    victim.alive = False
                    action_log["target"] = victim.name
                    # Determine witnesses
                    witnesses = [p.name for p in all_players
                                 if p.alive and p.name != player.name
                                 and p.name != victim.name
                                 and p.location == player.location]
                    action_log["witnesses"] = witnesses

            elif action == "COMPLETE_TASK" and player.role == "Crewmate":
                player.tasks_done += 1
                total_tasks = sum(p.total_tasks for p in all_players
                                  if p.role == "Crewmate")
                done_tasks = sum(p.tasks_done for p in all_players
                                 if p.role == "Crewmate")
                gs.task_progress = min(100.0,
                                       (done_tasks / total_tasks) * 100)

            elif action == "FAKE_TASK" and player.role == "Impostor":
                pass  # Does NOT increase task_progress

            # ── Step E: Critique ──
            state_dict = gs.to_dict()
            v_crew = critic.evaluate_state_value(state_dict, "Crewmate")
            v_imp  = critic.evaluate_state_value(state_dict, "Impostor")

            # ── Step F: Reward ──
            prev_dict = state_dict.copy()
            r = reward.calculate_step_reward(
                player, prev_dict, state_dict,
                action_log=action_log,
            )

            # ── Step G: Log ──
            if player.role == "Crewmate":
                sus = actors[player.name].suspicion_matrix
                max_sus = max(sus.values()) if sus else 0.0
            else:
                threats = actors[player.name].second_order_beliefs
                max_sus = max(threats.values()) if threats else 0.0

            log_rows.append({
                "turn": t,
                "player": player.name,
                "role": player.role,
                "action": action,
                "target": action_log.get("target"),
                "max_belief": round(max_sus, 4),
                "v_crew": round(v_crew, 4),
                "v_imp": round(v_imp, 4),
                "reward": round(r, 2),
                "tasks_pct": round(gs.task_progress, 1),
            })

        # ── Check Win Conditions ──
        alive_crew = [p for p in all_players
                      if p.alive and p.role == "Crewmate"]
        alive_imps = [p for p in all_players
                      if p.alive and p.role == "Impostor"]
        if len(alive_crew) <= len(alive_imps):
            gs.winner = "Impostor"
            gs.game_over = True
        elif gs.task_progress >= 100.0:
            gs.winner = "Crewmate"
            gs.game_over = True

        if gs.game_over:
            print(f"🏁 Game Over at turn {t}! Winner: {gs.winner}")
            break

    if not gs.game_over:
        print(f"⏰ Simulation ended after {num_turns} turns (no winner).")

    return pd.DataFrame(log_rows)


print("✅ run_simulation() defined.")
"""))

# ── Cell 4: Visualization ──
cells.append(code_cell("""\
# ═══════════════════════════════════════════════════════════════
# Visualization Suite
# ═══════════════════════════════════════════════════════════════

def plot_results(df):
    \"\"\"Generate the 3 proof plots from simulation data.\"\"\"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Among Us RL Pipeline — Simulation Metrics",
                 fontsize=14, fontweight="bold")

    # ── Plot 1: The "Brain" Plot (Beliefs over time) ──
    ax1 = axes[0]
    crew_df = df[df["role"] == "Crewmate"].groupby("turn")["max_belief"].max()
    imp_df  = df[df["role"] == "Impostor"].groupby("turn")["max_belief"].max()

    ax1.plot(crew_df.index, crew_df.values,
             "b-o", label="Crew Max Suspicion", markersize=4)
    ax1.plot(imp_df.index, imp_df.values,
             "r-s", label="Imp Max Threat", markersize=4)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Max Belief Score")
    ax1.set_title("🧠 The Brain Plot")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: The "Dopamine" Plot (Rewards over time) ──
    ax2 = axes[1]
    for role, color, marker in [("Crewmate", "blue", "o"),
                                 ("Impostor", "red", "s")]:
        role_df = df[df["role"] == role].groupby("turn")["reward"].sum()
        ax2.plot(role_df.index, role_df.values,
                 f"{color[0]}-{marker}", label=f"{role} Reward",
                 markersize=4)

    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("Step Reward r(t)")
    ax2.set_title("💊 The Dopamine Plot")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: The "Value" Plot (Critic values, zero-sum) ──
    ax3 = axes[2]
    # Use per-turn averages (all players see the same global state)
    turn_vals = df.groupby("turn")[["v_crew", "v_imp"]].mean()

    ax3.plot(turn_vals.index, turn_vals["v_crew"],
             "b-o", label="V(Crewmate)", markersize=4)
    ax3.plot(turn_vals.index, turn_vals["v_imp"],
             "r-s", label="V(Impostor)", markersize=4)
    ax3.fill_between(turn_vals.index, turn_vals["v_crew"],
                     turn_vals["v_imp"], alpha=0.1, color="purple")
    ax3.set_xlabel("Turn")
    ax3.set_ylabel("Win Probability V(s)")
    ax3.set_title("📈 The Value Plot (Zero-Sum)")
    ax3.legend(fontsize=8)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pipeline_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Plots saved to pipeline_results.png")


print("✅ plot_results() defined.")
"""))

# ── Cell 5: Execution ──
cells.append(code_cell("""\
# ═══════════════════════════════════════════════════════════════
# Run Simulation & Display Results
# ═══════════════════════════════════════════════════════════════

df = run_simulation(num_turns=15)

print(f"\\n📋 Collected {len(df)} log entries\\n")
df.head(10)
"""))

cells.append(code_cell("""\
plot_results(df)
"""))

cells.append(code_cell("""\
# Summary statistics
print("=== Per-Role Summary ===")
print(df.groupby("role")[["reward", "max_belief"]].describe()
        .round(3).to_string())
"""))

# ── Assemble notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__) or ".",
                         "..", "test_pipeline.ipynb")
out_path = os.path.abspath(out_path)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Notebook written to {out_path}")
