"""
rewards.py — Reward Engine ("Dopamine System")

Calculates scalar reward r(t) for every agent action, with:
  • Asymmetric reward tables for Crewmates vs Impostors
  • Dynamic Endgame scaling when game reaches Critical State
  • Social/cognitive penalties (hallucination, lying, vote strategy)

Main entry point: RewardEngine.calculate_step_reward(...)
"""

from typing import Any, Dict, List, Optional


class RewardEngine:
    """Reward calculator for the Among Us RL environment.

    Reward Hierarchy (highest priority first):
      1. Terminal rewards   — game over this turn
      2. Social/cognitive   — hallucination, lies, votes
      3. Action rewards     — role-specific per-action values
    """

    def __init__(self):
        self.history_buffer: List[Dict] = []

    # ── Critical State Detection ─────────────────────────────────────

    @staticmethod
    def _is_critical_state(game_state: Dict) -> bool:
        """Return True when the game is in an endgame Critical State.

        Triggers when EITHER condition is met:
          1. Living Crewmates <= 3
          2. Living Crewmates <= Living Impostors + 2
        """
        living_crew = game_state.get("living_crewmates", 0)
        living_imps = game_state.get("living_impostors", 0)

        return living_crew <= 3 or living_crew <= living_imps + 2

    # ── Main Reward Calculation ──────────────────────────────────────

    def calculate_step_reward(
        self,
        agent,
        prev_state: Dict,
        curr_state: Dict,
        action_log: Optional[Dict] = None,
        context_analysis: Optional[Dict] = None,
    ) -> float:
        """Calculate the scalar reward for a single agent step.

        Parameters
        ----------
        agent : object
            Player with at least ``.role`` ("Crewmate" / "Impostor"),
            ``.alive`` (bool), and ``.team`` (str).
        prev_state : dict
            Game state snapshot *before* the action.
        curr_state : dict
            Game state snapshot *after* the action.
        action_log : dict or None
            ``{"action": str, "target": str, "witnesses": list, ...}``
        context_analysis : dict or None
            LLM-Judge output: ``{"hallucination": bool,
            "lie_success": bool, "lie_refuted": bool,
            "vote_target_role": str, "impostor_survived_vote": bool}``

        Returns
        -------
        float
            The scalar reward value.
        """
        reward = 0.0
        role = getattr(agent, "role", "Crewmate")
        critical = self._is_critical_state(curr_state)

        if action_log is None:
            action_log = {}
        if context_analysis is None:
            context_analysis = {}

        # ─── A. Terminal Rewards (override everything) ───────────
        winner = curr_state.get("winner")
        if winner is not None:
            team = getattr(agent, "team", role)
            alive = getattr(agent, "alive", True)

            if team == winner:
                reward += 50.0 if alive else 30.0  # Martyr bonus
            else:
                reward += -20.0

            return reward  # terminal — skip action/social rewards

        # ─── B. Social & Cognitive Rewards ───────────────────────
        if context_analysis.get("hallucination"):
            reward += -100.0

        if role == "Impostor":
            if context_analysis.get("lie_success"):
                reward += 2.0
            if context_analysis.get("lie_refuted"):
                reward += -5.0

        # Vote strategy
        vote_target_role = context_analysis.get("vote_target_role")
        if vote_target_role:
            if role == "Crewmate":
                if vote_target_role == "Impostor":
                    reward += 5.0
                elif vote_target_role == "Crewmate":
                    reward += -2.0
            elif role == "Impostor":
                if vote_target_role == "Crewmate":
                    reward += 3.0

        if (role == "Impostor"
                and context_analysis.get("impostor_survived_vote")):
            reward += 10.0

        # ─── C. Action Rewards (Role-Specific) ──────────────────
        action = action_log.get("action")
        witnesses = action_log.get("witnesses", [])

        if action is None:
            return reward  # turn skipped

        if role == "Impostor":
            reward += self._impostor_action_reward(action, witnesses)
        else:
            reward += self._crewmate_action_reward(
                action, witnesses, critical)

        # Store for history tracking
        self.history_buffer.append({
            "role": role,
            "action": action,
            "reward": reward,
        })

        return reward

    # ── Role-Specific Action Tables ──────────────────────────────────

    @staticmethod
    def _impostor_action_reward(action: str, witnesses: list) -> float:
        """Impostor action reward table."""
        if action == "KILL":
            base = 10.0
            if not witnesses:
                base += 5.0       # unseen kill bonus
            else:
                # Graduated penalty: -8 per witness
                # 1 witness: 10 - 8 = +2 (risky but net positive)
                # 2 witnesses: 10 - 16 = -6 (net negative)
                # 3+ witnesses: increasingly punished
                base -= 8.0 * len(witnesses)
            return base

        if action == "REPORT_BODY":
            return 3.0   # self-report strategy

        if action == "FAKE_TASK":
            return 2.0   # blending in

        if action == "SABOTAGE":
            return 1.0

        if action == "FIX_SABOTAGE":
            return 1.0   # blending in

        if action == "VENT":
            if not witnesses:
                return 1.0   # mobility
            return -10.0      # critical failure

        return 0.0  # unknown action

    @staticmethod
    def _crewmate_action_reward(
        action: str, witnesses: list, critical: bool
    ) -> float:
        """Crewmate action reward table."""
        if action == "COMPLETE_TASK":
            return 5.0 if critical else 2.0

        if action == "FIX_SABOTAGE":
            return 3.0

        if action == "REPORT_BODY":
            return 2.0

        if action == "DIE":
            return -50.0 if critical else -15.0

        return 0.0  # unknown action


# ═══════════════════════════════════════════════════════════════════════════
# Verification Block
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    class _DummyAgent:
        def __init__(self, role, alive=True, team=None):
            self.role = role
            self.team = team or role
            self.alive = alive

    engine = RewardEngine()

    # --- Non-terminal game state (no winner) ---
    base_state = {
        "living_crewmates": 5,
        "living_impostors": 1,
        "winner": None,
    }
    critical_state = {
        "living_crewmates": 2,
        "living_impostors": 1,
        "winner": None,
    }

    # ── Test 1: Impostor Kills Unseen → 10.0 + 5.0 = 15.0 ──
    imp = _DummyAgent("Impostor")
    r = engine.calculate_step_reward(
        imp, base_state, base_state,
        action_log={"action": "KILL", "witnesses": []},
    )
    assert r == 15.0, f"Test 1 FAILED: expected 15.0, got {r}"
    print(f"Test 1 PASSED: Impostor kills unseen → {r}")

    # ── Test 2: Crewmate completes task in Critical State → 5.0 ──
    crew = _DummyAgent("Crewmate")
    r = engine.calculate_step_reward(
        crew, critical_state, critical_state,
        action_log={"action": "COMPLETE_TASK", "witnesses": []},
    )
    assert r == 5.0, f"Test 2 FAILED: expected 5.0, got {r}"
    print(f"Test 2 PASSED: Crewmate task in critical → {r}")

    # ── Test 3: Crewmate hallucinates → -100.0 ──
    r = engine.calculate_step_reward(
        crew, base_state, base_state,
        action_log=None,
        context_analysis={"hallucination": True},
    )
    assert r == -100.0, f"Test 3 FAILED: expected -100.0, got {r}"
    print(f"Test 3 PASSED: Hallucination penalty → {r}")

    # ── Test 4: Impostor vents with witness → -10.0 ──
    r = engine.calculate_step_reward(
        imp, base_state, base_state,
        action_log={"action": "VENT", "witnesses": ["Player 1: blue"]},
    )
    assert r == -10.0, f"Test 4 FAILED: expected -10.0, got {r}"
    print(f"Test 4 PASSED: Impostor vents with witness → {r}")

    # ── Test 5: Impostor kills with 1 witness → 10.0 - 8.0 = 2.0 ──
    r = engine.calculate_step_reward(
        imp, base_state, base_state,
        action_log={"action": "KILL", "witnesses": ["Player 2: green"]},
    )
    assert r == 2.0, f"Test 5 FAILED: expected 2.0, got {r}"
    print(f"Test 5 PASSED: Impostor kills with 1 witness → {r}")

    # ── Test 6: Impostor kills with 2 witnesses → 10.0 - 16.0 = -6.0 ──
    r = engine.calculate_step_reward(
        imp, base_state, base_state,
        action_log={"action": "KILL", "witnesses": ["Player 2: green", "Player 3: yellow"]},
    )
    assert r == -6.0, f"Test 6 FAILED: expected -6.0, got {r}"
    print(f"Test 6 PASSED: Impostor kills with 2 witnesses → {r}")

    print()
    print("✅ All 6 tests passed.")
