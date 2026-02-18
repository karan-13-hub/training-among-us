"""
critic.py — Critic Module ("Scorekeeper")

Evaluates the "goodness" of a game state from a team's perspective,
producing a win probability V(s) ∈ [0.0, 1.0].

  • Central Critic — has access to perfect global state during training
  • Asymmetric   — different heuristics for Crewmate vs Impostor
  • Fake-task aware — only real completed tasks count toward task %

Main entry point: CriticModule.evaluate_state_value(game_state, team)
"""

import json
import re
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Critic Prompt Template
# ═══════════════════════════════════════════════════════════════════════════

CRITIC_SYSTEM_PROMPT = """\
You are the Strategic Critic for an AI playing Among Us.

**Current Focus:** Evaluate the probability that the **{team}** Team \
will WIN this game.

**GAME STATE:**
  • Living Crewmates: {living_crew}
  • Living Impostors: {living_imps}
  • Total Tasks Completed: {task_pct}% (Note: Fake tasks do NOT count)
  • Ejected Players: {ejected}
  • Sabotage Active: {sabotage}

{heuristic_block}

Return ONLY a float between 0.0 and 1.0.
Example: 0.65
"""

CREWMATE_HEURISTICS = """\
**Evaluation Heuristics (Crewmate Perspective):**
  • Win Conditions: Tasks reach 100% OR All Impostors ejected.
  • High Value (>0.8): Tasks > 90%, 0 Impostors alive.
  • Low Value (<0.2): Living Crew <= Impostors + 1, Sabotage active.
  • Risk: If Living Crew is low, value drops drastically even if tasks \
are high.
"""

IMPOSTOR_HEURISTICS = """\
**Evaluation Heuristics (Impostor Perspective):**
  • Win Conditions: Living Crew <= Living Impostors OR Sabotage not \
fixed in time.
  • High Value (>0.8): Living Crew <= Impostors + 2, Sabotage active.
  • Low Value (<0.2): Impostor ejected, Tasks > 90%.
  • Note: Your team wins by killing, not by doing tasks.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CriticModule Class
# ═══════════════════════════════════════════════════════════════════════════

class CriticModule:
    """State-value estimator for the Among Us RL environment.

    Produces V(s, team) ∈ [0.0, 1.0] — the estimated probability that
    *team* wins from the current game state.
    """

    def __init__(self):
        # Placeholder for LLM client or learned model weights
        pass

    # ── public API ───────────────────────────────────────────────────

    def evaluate_state_value(
        self,
        game_state: Dict[str, Any],
        team_perspective: str,
    ) -> float:
        """Predict the win probability for *team_perspective*.

        Parameters
        ----------
        game_state : dict
            Must contain at minimum:
              ``living_crewmates``, ``living_impostors``,
              ``task_completion_pct``, ``sabotage_active``,
              ``ejected_roles`` (list of role strings).
        team_perspective : str
            ``"Crewmate"`` or ``"Impostor"``.

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        prompt = self._construct_prompt(game_state, team_perspective)
        raw = self._call_llm(prompt)
        return self._parse_value(raw)

    # ── prompt construction ──────────────────────────────────────────

    @staticmethod
    def _construct_prompt(
        game_state: Dict[str, Any],
        team: str,
    ) -> str:
        living_crew = game_state.get("living_crewmates", 0)
        living_imps = game_state.get("living_impostors", 0)
        task_pct = game_state.get("task_completion_pct", 0.0)
        sabotage = game_state.get("sabotage_active", False)
        ejected = game_state.get("ejected_roles", [])

        heuristic_block = (
            CREWMATE_HEURISTICS if team == "Crewmate"
            else IMPOSTOR_HEURISTICS
        )

        return CRITIC_SYSTEM_PROMPT.format(
            team=team,
            living_crew=living_crew,
            living_imps=living_imps,
            task_pct=round(task_pct, 1),
            ejected=", ".join(ejected) if ejected else "None",
            sabotage=sabotage,
            heuristic_block=heuristic_block,
        )

    # ── LLM placeholder ─────────────────────────────────────────────

    @staticmethod
    def _call_llm(prompt: str) -> str:
        """Placeholder — returns a heuristic float computed from the
        game-state numbers embedded in the prompt.

        Replace with a real LLM call (or a learned neural critic) later.
        """
        # ── extract numbers from prompt ──
        crew_m = re.search(r"Living Crewmates:\s*(\d+)", prompt)
        imp_m = re.search(r"Living Impostors:\s*(\d+)", prompt)
        task_m = re.search(r"Total Tasks Completed:\s*([\d.]+)%", prompt)
        sab_m = re.search(r"Sabotage Active:\s*(True|False)", prompt)

        crew = int(crew_m.group(1)) if crew_m else 5
        imps = int(imp_m.group(1)) if imp_m else 1
        task_pct = float(task_m.group(1)) if task_m else 0.0
        sabotage = (sab_m.group(1) == "True") if sab_m else False

        is_crew = "Crewmate" in prompt.split("Current Focus")[1].split(
            "GAME STATE")[0] if "Current Focus" in prompt else True

        if is_crew:
            return str(_heuristic_crew_value(crew, imps, task_pct,
                                             sabotage))
        else:
            return str(_heuristic_imp_value(crew, imps, task_pct,
                                            sabotage))

    # ── response parsing ─────────────────────────────────────────────

    @staticmethod
    def _parse_value(raw: str) -> float:
        """Extract a float from the LLM response and clamp to [0, 1]."""
        match = re.search(r"([\d]+\.[\d]+|[\d]+)", raw.strip())
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        return 0.5  # fallback neutral


# ═══════════════════════════════════════════════════════════════════════════
# Heuristic Value Functions (used by the stub)
# ═══════════════════════════════════════════════════════════════════════════

def _heuristic_crew_value(
    crew: int, imps: int, task_pct: float, sabotage: bool
) -> float:
    """Deterministic heuristic for Crewmate win probability.

    Components:
      • task_factor  — how close tasks are to 100%
      • numbers_factor — crew numerical advantage
      • sabotage_penalty — active sabotage hurts crew
    """
    # Terminal checks
    if imps == 0:
        return 1.0      # all impostors ejected
    if crew <= imps:
        return 0.0      # impostor parity = crew loss
    if task_pct >= 100.0:
        return 1.0      # tasks complete

    # Task progress (0.0 – 0.5 contribution)
    task_factor = (task_pct / 100.0) * 0.5

    # Numerical advantage (0.0 – 0.4 contribution)
    # crew_ratio: how many extra crew over the danger threshold
    safe_margin = (crew - imps) / max(crew + imps, 1)
    numbers_factor = safe_margin * 0.4

    # Sabotage penalty
    sab_penalty = 0.1 if sabotage else 0.0

    # Base value (start slightly optimistic at 0.5 neutral)
    value = 0.1 + task_factor + numbers_factor - sab_penalty
    return max(0.0, min(1.0, round(value, 4)))


def _heuristic_imp_value(
    crew: int, imps: int, task_pct: float, sabotage: bool
) -> float:
    """Deterministic heuristic for Impostor win probability.

    Designed so that V(crew) + V(imp) ≈ 1.0 for any given state.
    """
    crew_val = _heuristic_crew_value(crew, imps, task_pct, sabotage)
    return round(1.0 - crew_val, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Verification Block
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    critic = CriticModule()

    # ── Test 1: Balanced Start (7 Crew, 2 Imp, 0% Tasks) ──
    state1 = {
        "living_crewmates": 7,
        "living_impostors": 2,
        "task_completion_pct": 0.0,
        "sabotage_active": False,
        "ejected_roles": [],
    }
    v_crew_1 = critic.evaluate_state_value(state1, "Crewmate")
    v_imp_1 = critic.evaluate_state_value(state1, "Impostor")
    print(f"Test 1 — Balanced Start:  V(Crew)={v_crew_1:.4f}  "
          f"V(Imp)={v_imp_1:.4f}")
    assert 0.3 < v_crew_1 < 0.7, f"Crew should be near 0.5, got {v_crew_1}"
    assert 0.3 < v_imp_1 < 0.7, f"Imp should be near 0.5, got {v_imp_1}"
    print("  ✅ PASSED")

    # ── Test 2: Crew Winning (4 Crew, 1 Imp, 95% Tasks) ──
    state2 = {
        "living_crewmates": 4,
        "living_impostors": 1,
        "task_completion_pct": 95.0,
        "sabotage_active": False,
        "ejected_roles": ["Impostor"],
    }
    v_crew_2 = critic.evaluate_state_value(state2, "Crewmate")
    v_imp_2 = critic.evaluate_state_value(state2, "Impostor")
    print(f"\nTest 2 — Crew Winning:    V(Crew)={v_crew_2:.4f}  "
          f"V(Imp)={v_imp_2:.4f}")
    assert v_crew_2 > 0.7, f"Crew should be >0.7, got {v_crew_2}"
    assert v_imp_2 < 0.3, f"Imp should be <0.3, got {v_imp_2}"
    print("  ✅ PASSED")

    # ── Test 3: Impostor Winning (3 Crew, 2 Imp, 10% Tasks) ──
    state3 = {
        "living_crewmates": 3,
        "living_impostors": 2,
        "task_completion_pct": 10.0,
        "sabotage_active": False,
        "ejected_roles": [],
    }
    v_crew_3 = critic.evaluate_state_value(state3, "Crewmate")
    v_imp_3 = critic.evaluate_state_value(state3, "Impostor")
    print(f"\nTest 3 — Imp Winning:     V(Crew)={v_crew_3:.4f}  "
          f"V(Imp)={v_imp_3:.4f}")
    assert v_crew_3 < 0.3, f"Crew should be <0.3, got {v_crew_3}"
    assert v_imp_3 > 0.7, f"Imp should be >0.7, got {v_imp_3}"
    print("  ✅ PASSED")

    # ── Test 4: Zero-Sum Check ──
    for label, state in [("State1", state1), ("State2", state2),
                          ("State3", state3)]:
        vc = critic.evaluate_state_value(state, "Crewmate")
        vi = critic.evaluate_state_value(state, "Impostor")
        total = vc + vi
        print(f"\nTest 4 — Zero-Sum ({label}): V(C)={vc:.4f} + "
              f"V(I)={vi:.4f} = {total:.4f}")
        assert abs(total - 1.0) < 0.01, \
            f"V(Crew)+V(Imp) should ≈ 1.0, got {total}"
    print("  ✅ PASSED")

    print(f"\n{'=' * 50}")
    print("✅ All 4 critic tests passed.")
