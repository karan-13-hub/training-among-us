"""
test_pipeline_verify.py — Runs the notebook logic as a plain script
to verify the full integration pipeline works end-to-end.
"""
import sys, os, random, json

# Ensure imports work
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from amongagents.agent.actor import ActorModule
from amongagents.agent.critic import CriticModule
from amongagents.agent.rewards import RewardEngine

random.seed(42)
print("✅ All modules imported successfully.")

# ═══════════════════════════════════════════════════════════════
# Mock Infrastructure
# ═══════════════════════════════════════════════════════════════

ROOMS = ["Cafeteria", "Admin", "Electrical", "Medbay", "Reactor",
         "Navigation", "Security", "Storage", "Weapons", "Shields"]

class MockPlayer:
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
    def __init__(self, players):
        self.players = players
        self.task_progress = 0.0
        self.game_over = False
        self.winner = None
        self.sabotage_active = False
        self.turn = 0

    def to_dict(self):
        alive_crew = [p for p in self.players if p.alive and p.role == "Crewmate"]
        alive_imps = [p for p in self.players if p.alive and p.role == "Impostor"]
        return {
            "living_crewmates": len(alive_crew),
            "living_impostors": len(alive_imps),
            "task_completion_pct": self.task_progress,
            "sabotage_active": self.sabotage_active,
            "ejected_roles": [],
            "winner": self.winner,
        }

    def visible_players(self, player):
        return [p.name for p in self.players
                if p.alive and p.name != player.name
                and p.location == player.location]

print("✅ Mock infrastructure ready.")

# ═══════════════════════════════════════════════════════════════
# Simulation Loop
# ═══════════════════════════════════════════════════════════════

def run_simulation(num_turns=10):
    red    = MockPlayer("Player 1", "Impostor", "red")
    blue   = MockPlayer("Player 2", "Crewmate", "blue")
    green  = MockPlayer("Player 3", "Crewmate", "green")
    yellow = MockPlayer("Player 4", "Crewmate", "yellow")
    all_players = [red, blue, green, yellow]

    gs = MockGameState(all_players)
    actors  = {p.name: ActorModule(p, all_players) for p in all_players}
    critic  = CriticModule()
    reward  = RewardEngine()
    log_rows = []

    for t in range(num_turns):
        gs.turn = t
        for player in all_players:
            if not player.alive:
                continue

            player.location = random.choice(ROOMS)
            visible = gs.visible_players(player)
            player.verified_presence_log.append({
                "timestep": t, "room": player.location,
                "players_seen": visible,
            })

            obs_log = []
            for other in all_players:
                if other.name == player.name or not other.alive:
                    continue
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

            game_dict = {"visible_players": visible, "dead_bodies": []}
            result = actors[player.name].generate_actor_step(game_dict, [f"Turn {t}"])
            action = result.get("action_type", "STAY")

            action_log = {"action": action, "witnesses": [], "target": None}

            if action == "KILL" and player.role == "Impostor":
                crew_alive = [p for p in all_players
                              if p.alive and p.role == "Crewmate"
                              and p.location == player.location]
                if crew_alive:
                    victim = random.choice(crew_alive)
                    victim.alive = False
                    action_log["target"] = victim.name
                    witnesses = [p.name for p in all_players
                                 if p.alive and p.name != player.name
                                 and p.name != victim.name
                                 and p.location == player.location]
                    action_log["witnesses"] = witnesses

            elif action == "COMPLETE_TASK" and player.role == "Crewmate":
                player.tasks_done += 1
                total = sum(p.total_tasks for p in all_players if p.role == "Crewmate")
                done = sum(p.tasks_done for p in all_players if p.role == "Crewmate")
                gs.task_progress = min(100.0, (done / total) * 100)

            state_dict = gs.to_dict()
            v_crew = critic.evaluate_state_value(state_dict, "Crewmate")
            v_imp  = critic.evaluate_state_value(state_dict, "Impostor")

            r = reward.calculate_step_reward(
                player, state_dict, state_dict, action_log=action_log)

            if player.role == "Crewmate":
                sus = actors[player.name].suspicion_matrix
                max_sus = max(sus.values()) if sus else 0.0
            else:
                threats = actors[player.name].second_order_beliefs
                max_sus = max(threats.values()) if threats else 0.0

            log_rows.append({
                "turn": t, "player": player.name, "role": player.role,
                "action": action, "target": action_log.get("target"),
                "max_belief": round(max_sus, 4),
                "v_crew": round(v_crew, 4), "v_imp": round(v_imp, 4),
                "reward": round(r, 2), "tasks_pct": round(gs.task_progress, 1),
            })

        alive_crew = [p for p in all_players if p.alive and p.role == "Crewmate"]
        alive_imps = [p for p in all_players if p.alive and p.role == "Impostor"]
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


# ═══════════════════════════════════════════════════════════════
# Run & Visualize
# ═══════════════════════════════════════════════════════════════

df = run_simulation(num_turns=15)
print(f"\n📋 Collected {len(df)} log entries\n")
print(df.head(10).to_string())

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Among Us RL Pipeline — Simulation Metrics",
             fontsize=14, fontweight="bold")

crew_df = df[df["role"] == "Crewmate"].groupby("turn")["max_belief"].max()
imp_df  = df[df["role"] == "Impostor"].groupby("turn")["max_belief"].max()
axes[0].plot(crew_df.index, crew_df.values, "b-o", label="Crew Max Suspicion", markersize=4)
axes[0].plot(imp_df.index, imp_df.values, "r-s", label="Imp Max Threat", markersize=4)
axes[0].set_xlabel("Turn"); axes[0].set_ylabel("Max Belief Score")
axes[0].set_title("🧠 The Brain Plot"); axes[0].legend(fontsize=8)
axes[0].set_ylim(-0.05, 1.05); axes[0].grid(True, alpha=0.3)

for role, color, marker in [("Crewmate", "blue", "o"), ("Impostor", "red", "s")]:
    role_df = df[df["role"] == role].groupby("turn")["reward"].sum()
    axes[1].plot(role_df.index, role_df.values, f"{color[0]}-{marker}",
                 label=f"{role} Reward", markersize=4)
axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Turn"); axes[1].set_ylabel("Step Reward r(t)")
axes[1].set_title("💊 The Dopamine Plot"); axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

turn_vals = df.groupby("turn")[["v_crew", "v_imp"]].mean()
axes[2].plot(turn_vals.index, turn_vals["v_crew"], "b-o", label="V(Crewmate)", markersize=4)
axes[2].plot(turn_vals.index, turn_vals["v_imp"], "r-s", label="V(Impostor)", markersize=4)
axes[2].fill_between(turn_vals.index, turn_vals["v_crew"], turn_vals["v_imp"],
                     alpha=0.1, color="purple")
axes[2].set_xlabel("Turn"); axes[2].set_ylabel("Win Probability V(s)")
axes[2].set_title("📈 The Value Plot (Zero-Sum)"); axes[2].legend(fontsize=8)
axes[2].set_ylim(-0.05, 1.05); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "pipeline_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n📊 Plots saved to {out_path}")

print("\n=== Per-Role Summary ===")
print(df.groupby("role")[["reward", "max_belief"]].describe().round(3).to_string())
print("\n✅ Full pipeline integration test passed.")
