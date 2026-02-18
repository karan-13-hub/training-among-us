"""
judge.py — Hallucination Detection ("Judge System")

A referee module that detects when agents "break reality" by comparing
their natural-language speech against ground truth.

Pipeline:
  1. Fact Extraction   — parse speech into checkable claims
  2. Reality Check     — compare claims vs game logs / chat history
  3. LLM Judge         — frozen LLM classifies hallucination

Validation rules:
  • Crewmates → checked against Game Logs     (Strict Truth)
  • Impostors → checked against Chat History  (Self-Consistency)

Main entry point: detect_hallucination(agent, transcript, game_history)
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Judge Prompt Template
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """\
You are the Truthfulness Judge for an Among Us AI simulation.
Your task is to determine if an agent's statement is a HALLUCINATION.

GROUND TRUTH LOGS (what actually happened):
{ground_truth}

AGENT ROLE: {role}

AGENT STATEMENT:
"{statement}"

{previous_statements_block}

RULES:
1. If Agent is CREWMATE: Does the statement contradict the Ground Truth \
logs? A Crewmate should NEVER lie about where they were or what they did. \
Any mismatch between their claim and the logs is a hallucination.
2. If Agent is IMPOSTOR: Ignore contradictions with Ground Truth (they \
are allowed to lie). Instead, check if the statement contradicts the \
Agent's OWN previous statements. Changing their story is a hallucination.

OUTPUT VALID JSON AND NOTHING ELSE:
{{
  "is_hallucination": true or false,
  "reasoning": "One sentence explaining your verdict."
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# LLM Placeholder
# ═══════════════════════════════════════════════════════════════════════════

def _call_judge_llm(prompt: str) -> str:
    """Placeholder for a fast Judge LLM (e.g. Gemini Flash / Haiku).

    Replace with a real API call when wiring into the game loop.
    For now, returns a stub JSON so the pipeline can be tested.
    """
    # The stub checks for obvious keywords to simulate a real judge.
    prompt_lower = prompt.lower()

    _ROOM_PATTERN = (
        r'\b(cafeteria|admin|electrical|medbay|reactor|'
        r'navigation|security|storage|weapons|shields|'
        r'communications|o2|upper engine|lower engine)\b'
    )

    # Check Impostor FIRST — the prompt always mentions "Crewmate" in
    # the rules section, so we need to check role explicitly.
    if "AGENT ROLE: Impostor" in prompt:
        # Self-consistency check: compare current vs previous statements
        if "PREVIOUS STATEMENTS" in prompt:
            prev_block = prompt.split("PREVIOUS STATEMENTS")[1].split(
                "RULES")[0].lower()
            stmt_section = prompt.split("AGENT STATEMENT")[1].split(
                "PREVIOUS STATEMENTS")[0].lower()

            stmt_rooms = set(re.findall(_ROOM_PATTERN, stmt_section))
            prev_rooms = set(re.findall(_ROOM_PATTERN, prev_block))

            # Contradiction = they claim different rooms with no overlap
            if (stmt_rooms and prev_rooms
                    and not stmt_rooms.intersection(prev_rooms)):
                return json.dumps({
                    "is_hallucination": True,
                    "reasoning": "Current statement contradicts previous "
                                 "claim about location.",
                })

        return json.dumps({
            "is_hallucination": False,
            "reasoning": "Statement is self-consistent with prior claims.",
        })

    # Crewmate: strict truth check against ground truth logs
    if "AGENT ROLE: Crewmate" in prompt:
        gt_section = prompt.split("GROUND TRUTH LOGS")[1].split(
            "AGENT ROLE")[0].lower() if "GROUND TRUTH LOGS" in prompt else ""
        stmt_section = prompt.split("AGENT STATEMENT")[1].split(
            "RULES")[0].lower() if "AGENT STATEMENT" in prompt else ""

        rooms_mentioned = re.findall(_ROOM_PATTERN, stmt_section)
        for room in rooms_mentioned:
            if room not in gt_section:
                return json.dumps({
                    "is_hallucination": True,
                    "reasoning": f"Claimed to be in {room.title()}, "
                                 f"but logs do not confirm this.",
                })

        return json.dumps({
            "is_hallucination": False,
            "reasoning": "Statement is consistent with ground truth.",
        })

    # Default: no hallucination
    return json.dumps({
        "is_hallucination": False,
        "reasoning": "No contradictions detected.",
    })


# ═══════════════════════════════════════════════════════════════════════════
# JSON Parsing
# ═══════════════════════════════════════════════════════════════════════════

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> Optional[Dict]:
    """Extract a JSON dict from the Judge LLM output."""
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Core Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _build_ground_truth(agent, game_history: List[Dict]) -> str:
    """Format the last 3 turns of ground truth for the judge prompt.

    Uses ``game_history`` (list of turn dicts) and falls back to
    ``agent.history`` if available.
    """
    # Prefer explicit game_history
    if game_history:
        recent = game_history[-3:]
    else:
        hist = getattr(agent, "history", [])
        recent = hist[-3:]

    lines = []
    for i, entry in enumerate(recent, 1):
        if isinstance(entry, dict):
            turn = entry.get("timestep", i)
            room = entry.get("room", entry.get("location", "?"))
            action = entry.get("action", "?")
            lines.append(f"Turn {turn}: Room={room}, Action={action}")
        else:
            lines.append(f"Turn {i}: {entry}")

    return "\n".join(lines) if lines else "No history available."


def _build_previous_statements(agent, transcript: List[Dict]) -> str:
    """Collect the agent's own previous statements for self-consistency."""
    name = getattr(agent, "name", "")
    own_msgs = [
        t["message"]
        for t in transcript[:-1]  # everything except the current one
        if t.get("speaker") == name or t.get("agent") == name
    ]
    if not own_msgs:
        return ""
    lines = [f"  - \"{m}\"" for m in own_msgs[-5:]]
    return "PREVIOUS STATEMENTS BY THIS AGENT:\n" + "\n".join(lines)


def detect_hallucination(
    agent,
    transcript: List[Dict],
    game_history: List[Dict],
) -> Dict[str, Any]:
    """Run the 3-step hallucination detection pipeline.

    Parameters
    ----------
    agent : object
        Player with ``.role`` ("Crewmate" / "Impostor"),
        ``.name``, and optionally ``.history``.
    transcript : list[dict]
        Chat messages: ``[{"speaker": str, "message": str}, ...]``.
        The **last** entry is the statement being judged.
    game_history : list[dict]
        Ground truth turn logs:
        ``[{"timestep": int, "room": str, "action": str}, ...]``

    Returns
    -------
    dict
        ``{"is_hallucination": bool, "reasoning": str, "penalty": float}``
    """
    role = getattr(agent, "role", "Crewmate")

    # --- Step 1: Extract the statement to check ---
    if not transcript:
        return {
            "is_hallucination": False,
            "reasoning": "No statement to evaluate.",
            "penalty": 0.0,
        }

    statement = transcript[-1].get("message", "")
    if not statement.strip():
        return {
            "is_hallucination": False,
            "reasoning": "Empty statement — nothing to check.",
            "penalty": 0.0,
        }

    # --- Step 2: Build the judge prompt ---
    ground_truth = _build_ground_truth(agent, game_history)

    previous_block = ""
    if role == "Impostor":
        previous_block = _build_previous_statements(agent, transcript)

    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        role=role,
        statement=statement,
        previous_statements_block=previous_block,
    )

    # --- Step 3: Call the Judge LLM ---
    raw = _call_judge_llm(prompt)
    result = _parse_judge_json(raw)

    if result is None:
        # Judge LLM failed — err on the side of caution (no penalty)
        return {
            "is_hallucination": False,
            "reasoning": "Judge LLM returned unparseable output.",
            "penalty": 0.0,
        }

    is_hall = result.get("is_hallucination", False)
    reasoning = result.get("reasoning", "")

    return {
        "is_hallucination": is_hall,
        "reasoning": reasoning,
        "penalty": -100.0 if is_hall else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Verification Block
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    class _DummyAgent:
        def __init__(self, name, role):
            self.name = name
            self.role = role
            self.history = []

    # ── Test 1: Crewmate Spatial Hallucination ──
    crew = _DummyAgent("Player 1: blue", "Crewmate")
    game_hist = [
        {"timestep": 1, "room": "Cafeteria", "action": "MOVE"},
        {"timestep": 2, "room": "Admin", "action": "COMPLETE_TASK"},
        {"timestep": 3, "room": "Storage", "action": "MOVE"},
    ]
    transcript = [
        {"speaker": "Player 1: blue",
         "message": "I was in Reactor doing my tasks."},
    ]
    result = detect_hallucination(crew, transcript, game_hist)
    print(f"Test 1: Crewmate spatial hallucination")
    print(f"  is_hallucination={result['is_hallucination']}, "
          f"penalty={result['penalty']}")
    print(f"  reasoning: {result['reasoning']}")
    assert result["is_hallucination"] is True, \
        f"Test 1 FAILED: expected hallucination, got {result}"
    assert result["penalty"] == -100.0
    print("  ✅ PASSED")

    # ── Test 2: Crewmate Truthful Statement ──
    transcript_ok = [
        {"speaker": "Player 1: blue",
         "message": "I was in Admin completing a task."},
    ]
    result2 = detect_hallucination(crew, transcript_ok, game_hist)
    print(f"\nTest 2: Crewmate truthful statement")
    print(f"  is_hallucination={result2['is_hallucination']}, "
          f"penalty={result2['penalty']}")
    print(f"  reasoning: {result2['reasoning']}")
    assert result2["is_hallucination"] is False, \
        f"Test 2 FAILED: expected no hallucination, got {result2}"
    assert result2["penalty"] == 0.0
    print("  ✅ PASSED")

    # ── Test 3: Impostor Self-Contradiction ──
    imp = _DummyAgent("Player 2: red", "Impostor")
    transcript_imp = [
        {"speaker": "Player 2: red",
         "message": "I was in Medbay doing a scan."},
        {"speaker": "Player 2: red",
         "message": "I was in Admin the whole time."},
    ]
    result3 = detect_hallucination(imp, transcript_imp, game_hist)
    print(f"\nTest 3: Impostor self-contradiction")
    print(f"  is_hallucination={result3['is_hallucination']}, "
          f"penalty={result3['penalty']}")
    print(f"  reasoning: {result3['reasoning']}")
    assert result3["is_hallucination"] is True, \
        f"Test 3 FAILED: expected hallucination, got {result3}"
    assert result3["penalty"] == -100.0
    print("  ✅ PASSED")

    # ── Test 4: Impostor Consistent Lie ──
    transcript_imp_ok = [
        {"speaker": "Player 2: red",
         "message": "I was in Medbay doing a scan."},
        {"speaker": "Player 2: red",
         "message": "I finished my Medbay scan and headed to Cafeteria."},
    ]
    result4 = detect_hallucination(imp, transcript_imp_ok, game_hist)
    print(f"\nTest 4: Impostor consistent lie (should be OK)")
    print(f"  is_hallucination={result4['is_hallucination']}, "
          f"penalty={result4['penalty']}")
    print(f"  reasoning: {result4['reasoning']}")
    assert result4["is_hallucination"] is False, \
        f"Test 4 FAILED: expected no hallucination, got {result4}"
    assert result4["penalty"] == 0.0
    print("  ✅ PASSED")

    print(f"\n{'=' * 50}")
    print("✅ All 4 judge tests passed.")
