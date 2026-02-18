"""
actor.py — Theory of Mind Actor Module

Class-based actor with deterministic belief updates and LLM action generation.
  • Crewmate  → First-order suspicion matrix   (who is the Impostor?)
  • Impostor  → Second-order belief estimation  (who suspects ME?)

Belief update rules are hard-coded (no LLM needed).
Action generation uses prompts from neutral_prompts.py + ToM overlays.
"""

import json
import re
from typing import Any, Dict, List, Optional

from amongagents.agent.neutral_prompts import CREWMATE_PROMPT, IMPOSTOR_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
# Theory of Mind Overlay Prompts
# ═══════════════════════════════════════════════════════════════════════════

CREWMATE_TOM_OVERLAY = """\

## THEORY OF MIND: SUSPICION ESTIMATION (First-Order)
In addition to your normal action selection, you MUST also perform:

**Task A: SUSPICION ESTIMATION**
- Analyze the logs. Who is the Impostor?
- Assign a `Suspicion Score` (0.0 - 1.0) to every other player.
  * 0.0 = Totally Safe (Visual Task confirmed).
  * 1.0 = Hard Accusation / Seen Venting.

**Task B: ACTION SELECTION**
- **Constraint:** Do not simply run away from high suspicion players.
- **Objective:**
  * **High Suspicion:** Observe them from a safe distance (group up). Do not be alone with them.
  * **Low Suspicion:** Focus on Tasks.
  * **Imminent Danger:** Report Body or Call Meeting.

**Output JSON (you MUST output valid JSON and nothing else):**
```
{{
  "role": "Crewmate",
  "suspicion_matrix": {{"Red": 0.8, "Blue": 0.1, ...}},
  "thought": "Red is sus. I will watch them but stay near Blue.",
  "action_type": "MOVE",
  "target": "Admin"
}}
```
"""

IMPOSTOR_TOM_OVERLAY = """\

## THEORY OF MIND: SECOND-ORDER BELIEF ESTIMATION
In addition to your normal action selection, you MUST also perform:

**Task A: SECOND-ORDER BELIEF ESTIMATION**
- Analyze the logs. **Who suspects YOU?**
- Assign a `Threat Score` (0.0 - 1.0) to every other player.
  * 0.0 = They trust me.
  * 1.0 = They saw me kill/vent.

**Task B: STRATEGIC DECISION**
- **High Threat (Witness):** Risk a kill now? Or frame them in a meeting?
- **Low Threat (Target):** Kill to reduce numbers? Or keep alive as an alibi/frame target?

**Task C: ACTION**
- Execute Strategy: KILL, VENT, FAKE_TASK, REPORT_BODY (Self-Report), SABOTAGE.

**Output JSON (you MUST output valid JSON and nothing else):**
```
{{
  "role": "Impostor",
  "second_order_beliefs": {{"Red": 0.9, "Blue": 0.1, ...}},
  "thought": "Red knows. I must kill Red before they call a meeting.",
  "action_type": "KILL",
  "target": "Red"
}}
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# ActorModule Class
# ═══════════════════════════════════════════════════════════════════════════

class ActorModule:
    """Theory of Mind actor with deterministic belief updates.

    Belief Update Rules (hard-coded, no LLM):
      1. Witness KILL/VENT       → Set to 1.0   (Hard Evidence)
      2. Witness SABOTAGE        → ×1.25        (Strong Suspicion)
      3. Witness FAKE_TASK       → ×1.10        (Weak Suspicion)
      4. Witness VISUAL_TASK     → ×0.90        (Soft Clear)
      5. Witness COMPLETE_TASK   → ×0.90        (Trust Building)
    """

    def __init__(self, player, all_players):
        """Initialise the Actor with uniform beliefs (0.5 = neutral)."""
        self.player = player
        # Real game objects use .identity; ensure .role exists as alias
        if not hasattr(player, 'role'):
            player.role = getattr(player, 'identity', 'Crewmate')
        initial_val = 0.5

        self.suspicion_matrix = {
            p.name: initial_val for p in all_players if p.name != player.name
        }
        self.second_order_beliefs = {
            p.name: initial_val for p in all_players if p.name != player.name
        }

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float) -> float:
        """Keep values in [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

    # ── deterministic belief updates ─────────────────────────────────

    def update_beliefs(self, observation_log: List[Dict]):
        """Update belief state based on observed actions.

        Each event in *observation_log* should be a dict::

            {
                "subject": "Player 2: red",   # who performed the action
                "action":  "KILL",             # action type
                "witnesses": ["Player 1: blue", ...]  # who saw it
            }

        Rules:
          1. KILL / VENT       → 1.0  (Hard Evidence)
          2. SABOTAGE          → ×1.25
          3. FAKE_TASK         → ×1.10
          4. VISUAL_TASK       → ×0.90 (Soft Clear)
          5. COMPLETE_TASK     → ×0.90
        """
        for event in observation_log:
            subject = event["subject"]
            action = event["action"]
            witnesses = event.get("witnesses", [])

            if subject == self.player.name:
                continue

            # ── CREWMATE: first-order suspicion ──
            if self.player.role == "Crewmate":
                cur = self.suspicion_matrix.get(subject, 0.5)

                if action in ("KILL", "VENT"):
                    new = 1.0
                elif action == "SABOTAGE":
                    new = cur * 1.25
                elif action == "FAKE_TASK":
                    new = cur * 1.10
                elif action == "VISUAL_TASK":
                    new = cur * 0.90
                elif action == "COMPLETE_TASK":
                    new = cur * 0.90
                else:
                    new = cur

                self.suspicion_matrix[subject] = self._clamp(new)

            # ── IMPOSTOR: second-order threat ──
            elif self.player.role == "Impostor":
                cur = self.second_order_beliefs.get(subject, 0.5)

                # Did *subject* witness ME doing something incriminating?
                if (event["subject"] == self.player.name
                        and subject in witnesses):
                    if action in ("KILL", "VENT"):
                        new = 1.0
                    elif action == "SABOTAGE":
                        new = cur * 1.25
                    elif action == "FAKE_TASK":
                        new = cur * 1.10
                    else:
                        new = cur
                else:
                    new = cur  # no new info → keep current estimate

                self.second_order_beliefs[subject] = self._clamp(new)

    # ── LLM-based action generation ──────────────────────────────────

    def generate_actor_step(self, game_state=None, context_log=None):
        """Construct the role-specific prompt, call the LLM, parse JSON.

        Returns a dict with the action and the current mental-state matrix.
        """
        role = getattr(self.player, "identity",
                       getattr(self.player, "role", "Crewmate"))
        if role.lower() in ("impostor", "imposter"):
            role = "Impostor"
        else:
            role = "Crewmate"

        name = getattr(self.player, "name",
                       getattr(self.player, "color", "Unknown"))

        # Build system prompt: base from neutral_prompts + ToM overlay
        if role == "Crewmate":
            system_prompt = (CREWMATE_PROMPT.format(name=name)
                             + CREWMATE_TOM_OVERLAY)
        else:
            system_prompt = (IMPOSTOR_PROMPT.format(name=name)
                             + IMPOSTOR_TOM_OVERLAY)

        user_prompt = _format_context_log(self.player, game_state,
                                          context_log)

        # Call LLM with retry on parse failure
        for attempt in range(MAX_RETRIES):
            raw = call_llm(system_prompt, user_prompt)
            parsed = _parse_llm_json(raw)
            if parsed is not None:
                # Inject the current deterministic matrices
                if role == "Crewmate":
                    parsed["suspicion_matrix"] = self.suspicion_matrix
                else:
                    parsed["second_order_beliefs"] = self.second_order_beliefs
                return parsed
            print(f"[actor] JSON parse failed for {name} ({role}), "
                  f"attempt {attempt + 1}/{MAX_RETRIES}. Retrying…")

        result = _default_action(role)
        if role == "Crewmate":
            result["suspicion_matrix"] = self.suspicion_matrix
        else:
            result["second_order_beliefs"] = self.second_order_beliefs
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Context Formatting
# ═══════════════════════════════════════════════════════════════════════════

def _format_context_log(player, game_state=None, context_log=None) -> str:
    """Build the user-message context from the player's recent history."""
    parts: List[str] = []

    presence_log = getattr(player, "verified_presence_log", [])
    if presence_log:
        parts.append("## Recent History (last 5 turns)")
        for entry in presence_log[-5:]:
            ts = entry.get("timestep", "?")
            room = entry.get("room", "?")
            seen = entry.get("players_seen", [])
            if seen:
                parts.append(f"  T{ts}: {room} — saw {', '.join(seen)}")
            else:
                parts.append(f"  T{ts}: {room} — no one else present")

    current_room = getattr(player, "location", None)
    if current_room:
        parts.append(f"\nCurrent room: {current_room}")

    if game_state:
        visible = game_state.get("visible_players", [])
        if visible:
            parts.append(f"Players visible right now: {', '.join(visible)}")
        dead_bodies = game_state.get("dead_bodies", [])
        if dead_bodies:
            parts.append(
                f"⚠️ DEAD BODIES in room: {', '.join(dead_bodies)}")

    if context_log:
        parts.append("\n## Event Log (recent)")
        for entry in context_log[-5:]:
            parts.append(f"  • {entry}")

    obs = getattr(player, "observation_history", [])
    if obs:
        parts.append("\n## Observations")
        for o in obs[-5:]:
            parts.append(f"  • {o}")

    return "\n".join(parts) if parts else "No context available."


# ═══════════════════════════════════════════════════════════════════════════
# LLM Placeholder
# ═══════════════════════════════════════════════════════════════════════════

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Placeholder — swap for real LLM provider later."""
    is_impostor = ("You are an Impostor" in system_prompt
                   or "SECOND-ORDER BELIEF" in system_prompt)
    if is_impostor:
        return json.dumps({
            "role": "Impostor",
            "second_order_beliefs": {},
            "thought": "Blue saw me near the body. Must eliminate Blue.",
            "action_type": "KILL",
            "target": "Blue",
        })
    return json.dumps({
        "role": "Crewmate",
        "suspicion_matrix": {},
        "thought": "Red is suspicious. Staying with Green for safety.",
        "action_type": "MOVE",
        "target": "Admin",
    })


# ═══════════════════════════════════════════════════════════════════════════
# JSON Parsing Helpers
# ═══════════════════════════════════════════════════════════════════════════

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

MAX_RETRIES = 2


def _parse_llm_json(raw: str) -> Optional[Dict]:
    """Extract a JSON dict from LLM output (direct → fenced → greedy)."""
    for extract in (
        lambda r: json.loads(r.strip()),
        lambda r: json.loads(_JSON_BLOCK_RE.search(r).group(1))
                  if _JSON_BLOCK_RE.search(r) else None,
        lambda r: json.loads(_JSON_OBJECT_RE.search(r).group(0))
                  if _JSON_OBJECT_RE.search(r) else None,
    ):
        try:
            obj = extract(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _default_action(role: str) -> Dict[str, Any]:
    """Safe fallback when JSON parsing fails."""
    if role == "Crewmate":
        return {
            "role": "Crewmate",
            "suspicion_matrix": {},
            "thought": "Unable to reason — staying put.",
            "action_type": "STAY",
            "target": None,
        }
    return {
        "role": "Impostor",
        "second_order_beliefs": {},
        "thought": "Unable to reason — faking a task.",
        "action_type": "FAKE_TASK",
        "target": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Verification Block
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    class _DummyPlayer:
        def __init__(self, name, role, color, location="Cafeteria"):
            self.name = f"{name}: {color}"
            self.identity = role
            self.role = role
            self.color = color
            self.location = location
            self.observation_history: List[str] = []
            self.verified_presence_log: List[Dict] = []

    # --- Setup ---
    blue = _DummyPlayer("Player 1", "Crewmate", "blue")
    blue.verified_presence_log = [
        {"timestep": 0, "room": "Cafeteria",
         "players_seen": ["Player 2: red", "Player 3: green"]},
        {"timestep": 1, "room": "Admin",
         "players_seen": ["Player 2: red"]},
    ]
    blue.observation_history = [
        "Player 2: red moved from Cafeteria to Admin",
    ]

    red = _DummyPlayer("Player 2", "Impostor", "red")
    red.verified_presence_log = [
        {"timestep": 0, "room": "Cafeteria",
         "players_seen": ["Player 1: blue", "Player 3: green"]},
        {"timestep": 1, "room": "Admin",
         "players_seen": ["Player 1: blue"]},
    ]

    green = _DummyPlayer("Player 3", "Crewmate", "green")
    all_players = [blue, red, green]

    # --- Test belief updates ---
    actor_crew = ActorModule(blue, all_players)
    actor_imp = ActorModule(red, all_players)

    print("=" * 60)
    print("INITIAL BELIEFS")
    print("=" * 60)
    print(f"Crewmate suspicion: {actor_crew.suspicion_matrix}")
    print(f"Impostor threat:    {actor_imp.second_order_beliefs}")

    # Simulate observations
    obs_log = [
        {"subject": "Player 2: red", "action": "KILL",
         "witnesses": ["Player 1: blue"]},
        {"subject": "Player 3: green", "action": "VISUAL_TASK",
         "witnesses": ["Player 1: blue"]},
    ]
    actor_crew.update_beliefs(obs_log)

    imp_obs_log = [
        {"subject": "Player 2: red", "action": "KILL",
         "witnesses": ["Player 1: blue"]},
    ]
    actor_imp.update_beliefs(imp_obs_log)

    print()
    print("=" * 60)
    print("AFTER OBSERVATIONS")
    print("=" * 60)
    print(f"Crewmate suspicion: {actor_crew.suspicion_matrix}")
    print(f"Impostor threat:    {actor_imp.second_order_beliefs}")

    # --- Test generate_actor_step ---
    game_state = {"visible_players": ["Player 3: green"],
                  "dead_bodies": []}
    context_log = ["Round 5 started.", "No sabotage active."]

    print()
    print("=" * 60)
    print("CREWMATE ACTION")
    print("=" * 60)
    crew_result = actor_crew.generate_actor_step(game_state, context_log)
    print(json.dumps(crew_result, indent=2))

    print()
    print("=" * 60)
    print("IMPOSTOR ACTION")
    print("=" * 60)
    imp_result = actor_imp.generate_actor_step(game_state, context_log)
    print(json.dumps(imp_result, indent=2))

    # --- Assertions ---
    print()
    print("=" * 60)
    print("ASSERTIONS")
    print("=" * 60)

    # Belief updates
    assert actor_crew.suspicion_matrix["Player 2: red"] == 1.0, \
        "KILL should set suspicion to 1.0"
    assert actor_crew.suspicion_matrix["Player 3: green"] == 0.45, \
        "VISUAL_TASK should reduce by ×0.90 (0.5 → 0.45)"

    # Crewmate output keys
    crew_keys = {"role", "suspicion_matrix", "thought",
                 "action_type", "target"}
    assert crew_keys.issubset(crew_result.keys()), \
        f"Missing: {crew_keys - crew_result.keys()}"
    assert crew_result["role"] == "Crewmate"
    assert isinstance(crew_result["suspicion_matrix"], dict)

    # Impostor output keys
    imp_keys = {"role", "second_order_beliefs", "thought",
                "action_type", "target"}
    assert imp_keys.issubset(imp_result.keys()), \
        f"Missing: {imp_keys - imp_result.keys()}"
    assert imp_result["role"] == "Impostor"
    assert isinstance(imp_result["second_order_beliefs"], dict)

    print("✅ All assertions passed.")
