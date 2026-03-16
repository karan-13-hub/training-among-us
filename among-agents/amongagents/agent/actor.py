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

    A trained ``BeliefModel`` can be injected via :meth:`set_belief_model`.
    When set, the neural model is used for belief updates; otherwise the
    hard-coded rules above remain active (backward compatible).
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

        # Ordered list of other-player names (needed for tensor conversion)
        self._all_player_names = [p.name for p in all_players if p.name != player.name]

        # Optional trained belief model (injected at RL training time)
        self._belief_model  = None   # training.belief_model.BeliefModel
        self._belief_device = None

        # Cached hidden-state pool from the last LoRA forward pass.
        # Set by the RolloutCollector (or any inference wrapper) after each
        # LLM call so that update_beliefs() can feed it into the BeliefModel.
        self._last_hidden_pool = None  # Optional[torch.Tensor] shape [hidden_size]

    def set_belief_model(self, belief_model, device="cuda"):
        """
        Inject a trained :class:`training.belief_model.BeliefModel`.

        Once set, :meth:`update_beliefs` will use the neural network output
        to **replace** the rule-computed suspicion / second-order belief
        values so that :meth:`format_belief_for_prompt` shows model beliefs.

        Rules still run first so that hard-evidence events (KILL/VENT) are
        always reflected; the model then refines the full distribution.

        Parameters
        ----------
        belief_model : BeliefModel
        device : str
            Device where the model lives ("cuda" / "cpu").
        """
        import torch
        self._belief_model  = belief_model.eval()
        self._belief_device = torch.device(device)

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

        **Phase 1 — Hard Rules** (always run):
          1. KILL / VENT       → 1.0  (Hard Evidence)
          2. SABOTAGE          → ×1.25
          3. FAKE_TASK         → ×1.10
          4. VISUAL_TASK       → ×0.90 (Soft Clear)
          5. COMPLETE_TASK     → ×0.90

        **Phase 2 — BeliefModel override** (when a trained model is set):
        The model's predicted belief vector replaces the rule-computed values
        so that :meth:`format_belief_for_prompt` always shows model beliefs.
        Hard-evidence events (KILL/VENT = 1.0) are re-applied as a floor
        so the model cannot suppress a directly-witnessed kill.
        """
        # ── Phase 1: deterministic rules ───────────────────────────────
        _hard_evidence: dict = {}   # track direct witnesses (floor for model)

        for event in observation_log:
            subject  = event["subject"]
            action   = event["action"]
            witnesses = event.get("witnesses", [])

            if subject == self.player.name:
                continue

            # ── CREWMATE: first-order suspicion ──
            if self.player.role == "Crewmate":
                cur = self.suspicion_matrix.get(subject, 0.5)

                if action in ("KILL", "VENT"):
                    new = 1.0
                    _hard_evidence[subject] = 1.0
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

                if (event["subject"] == self.player.name
                        and subject in witnesses):
                    if action in ("KILL", "VENT"):
                        new = 1.0
                        _hard_evidence[subject] = 1.0
                    elif action == "SABOTAGE":
                        new = cur * 1.25
                    elif action == "FAKE_TASK":
                        new = cur * 1.10
                    else:
                        new = cur
                else:
                    new = cur

                self.second_order_beliefs[subject] = self._clamp(new)

        # ── Phase 2: BeliefModel override ──────────────────────────────
        # If a trained model is attached and we have a cached hidden pool,
        # run the forward pass and OVERWRITE the belief dicts so the prompt
        # shows model beliefs instead of (or in addition to) rule beliefs.
        if self._belief_model is not None and self._last_hidden_pool is not None:
            self._apply_belief_model(_hard_evidence)

    def _apply_belief_model(self, hard_evidence: Optional[Dict[str, float]] = None):
        """
        Run the BeliefModel forward pass using ``_last_hidden_pool`` and
        **overwrite** ``suspicion_matrix`` / ``second_order_beliefs`` with the
        model’s predicted values.

        Hard-evidence floors (KILL/VENT = 1.0) are re-applied after the model
        output so the model cannot suppress directly-witnessed crimes.

        Parameters
        ----------
        hard_evidence : dict or None
            Mapping of player-name → floor value (1.0) for directly-witnessed
            kill/vent events that must remain at 1.0 regardless of model output.
        """
        import torch
        from training.belief_model import beliefs_to_tensor

        if self._belief_model is None or self._last_hidden_pool is None:
            return

        hard_evidence = hard_evidence or {}
        is_impostor = self.player.role in ("Impostor", "Imposter")
        beliefs = self.second_order_beliefs if is_impostor else self.suspicion_matrix

        B   = len(self._all_player_names)
        D   = self._belief_model.config.belief_dim

        # Current belief vector (model’s prior)
        bvec = beliefs_to_tensor(beliefs, self._all_player_names, D)\
                   .unsqueeze(0).to(self._belief_device)   # [1, D]

        # Role embedding (0 = Crewmate, 1 = Impostor)
        role_id = torch.tensor(
            [1 if is_impostor else 0],
            dtype=torch.long, device=self._belief_device
        )

        # Hidden pool — ensure float32 on correct device
        hp = self._last_hidden_pool.float().to(self._belief_device)
        if hp.dim() == 1:
            hp = hp.unsqueeze(0)   # [1, H]

        with torch.no_grad():
            pred = self._belief_model(hp, bvec, role_id)   # [1, D]
        pred = pred.squeeze(0).cpu()  # [D]

        # Map model output back to the belief dict
        for i, name in enumerate(self._all_player_names):
            if i >= D:
                break
            score = float(self._clamp(pred[i].item()))
            # Re-apply hard-evidence floor
            score = max(score, hard_evidence.get(name, 0.0))
            if is_impostor:
                self.second_order_beliefs[name] = score
            else:
                self.suspicion_matrix[name] = score

    def update_beliefs_from_model(self, hidden_pool):
        """
        Explicitly update beliefs using the BeliefModel with a freshly-computed
        ``hidden_pool``.

        This is the primary API used by the :class:`RolloutCollector` (or any
        RL inference wrapper) after each LLM forward pass where the hidden states
        are available.  Calling this stores the hidden pool and immediately runs
        :meth:`_apply_belief_model`.

        Parameters
        ----------
        hidden_pool : torch.Tensor
            Pooled hidden states from the policy backbone, shape [hidden_size]
            or [1, hidden_size].
        """
        self._last_hidden_pool = hidden_pool
        if self._belief_model is not None:
            self._apply_belief_model()


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

    # ── Belief → Prompt Formatter ─────────────────────────────────────────

    def format_belief_for_prompt(
        self,
        top_k: int = 3,
        threshold_high: float = 0.70,
        threshold_low: float = 0.30,
    ) -> str:
        """
        Render the current belief state as a concise advisory block for
        inclusion in LLM action-selection and speech prompts.

        For **Crewmates** this exposes the first-order suspicion matrix so
        the agent can:
          - speak strategically (accuse high-suspicion players, vouch for low)
          - choose actions that prioritise self-preservation (avoid being alone
            with high-suspicion players)

        For **Impostors** this exposes the second-order threat model so the
        agent can:
          - prioritise killing witnesses before they call a meeting
          - frame low-threat players as they are trusted by the group

        Parameters
        ----------
        top_k : int
            Maximum number of players to list in each category.
        threshold_high : float
            Belief score above which a player is flagged as HIGH suspicion /
            HIGH threat.
        threshold_low : float
            Belief score below which a player is flagged as SAFE / TRUSTED.

        Returns
        -------
        str  — a formatted block ready to splice into a prompt, or "" if the
               belief dict is empty.
        """
        role = getattr(self.player, "identity",
                       getattr(self.player, "role", "Crewmate"))
        is_impostor = role.lower() in ("impostor", "imposter")

        if is_impostor:
            beliefs = self.second_order_beliefs
        else:
            beliefs = self.suspicion_matrix

        if not beliefs:
            return ""

        # Sort by score descending
        ranked = sorted(beliefs.items(), key=lambda kv: kv[1], reverse=True)

        high = [(n, s) for n, s in ranked if s >= threshold_high][:top_k]
        mid  = [(n, s) for n, s in ranked
                if threshold_low <= s < threshold_high][:top_k]
        low  = [(n, s) for n, s in ranked if s < threshold_low][:top_k]

        lines = []

        if is_impostor:
            lines.append("## 🧠 SECOND-ORDER BELIEF (who suspects YOU?)")
            lines.append(
                "These are YOUR internal threat estimates. "
                "Higher = that player suspects you more. "
                "Use this to decide kill priority and speech strategy."
            )

            if high:
                lines.append("\n**⚠️ HIGH THREAT — may expose you (prioritise eliminating or silencing):**")
                for name, score in high:
                    lines.append(f"  • {name}: {score:.2f} — they may accuse you next meeting")

            if mid:
                lines.append("\n**😐 MODERATE THREAT — uncertain, stay vague around them:**")
                for name, score in mid:
                    lines.append(f"  • {name}: {score:.2f}")

            if low:
                lines.append("\n**✅ LOW THREAT — trust you (safe to keep alive as alibi cover):**")
                for name, score in low:
                    lines.append(f"  • {name}: {score:.2f} — a good frame target or useful shield")

            lines.append(
                "\n**STRATEGY HINT:**"
                "\n  • In DISCUSSION: deflect accusations away from yourself toward neutral/medium players."
                "\n  • On KILL decisions: HIGH THREAT players are your biggest risk — eliminate before next meeting."
                "\n  • LOW THREAT players can be framed — mention you 'saw them acting suspicious'."
            )

        else:  # Crewmate
            lines.append("## 🧠 SUSPICION MATRIX (your internal read on other players)")
            lines.append(
                "These are YOUR belief estimates — not shared publicly. "
                "Higher = you believe that player is the Impostor. "
                "Use this to decide who to accuse, who to trust, and where to go."
            )

            if high:
                lines.append("\n**🔴 HIGH SUSPICION — strong impostor signal:**")
                for name, score in high:
                    lines.append(f"  • {name}: {score:.2f} — avoid being alone with them; accuse in meetings")

            if mid:
                lines.append("\n**🟡 MEDIUM SUSPICION — needs more observation:**")
                for name, score in mid:
                    lines.append(f"  • {name}: {score:.2f} — watch carefully; don't fully trust")

            if low:
                lines.append("\n**🟢 LOW SUSPICION — likely safe:**")
                for name, score in low:
                    lines.append(f"  • {name}: {score:.2f} — can move/work near them without extra risk")

            lines.append(
                "\n**STRATEGY HINT:**"
                "\n  • In DISCUSSION: name your top suspect and cite WHY (what you observed)."
                "\n  • When MOVING: prefer rooms with LOW suspicion players; avoid HIGH suspicion ones when alone."
                "\n  • In VOTING: weight your vote toward your highest-suspicion player IF you have evidence."
            )

        return "\n".join(lines)


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
