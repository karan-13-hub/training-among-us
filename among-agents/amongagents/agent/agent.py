import ast
import json
import os
import random
import re
from datetime import datetime
from typing import Any, List, Dict, Tuple
import aiohttp
import time
import numpy as np
import requests
import asyncio
from amongagents.agent.neutral_prompts import *

# Set Flask environment variable to True by default
if "FLASK" not in os.environ:
    os.environ["FLASK"] = "True"

# Suppress verbose logs from vllm and httpx
import logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global dictionary to store futures for human actions, keyed by game_id
human_action_futures: Dict[int, asyncio.Future] = {}

class Agent:
    def __init__(self, player):
        self.player = player

    def respond(self, message):
        return "..."

    def choose_action(self):
        return None


class LLMAgent(Agent):
    def __init__(self, player, tools, game_index, agent_config, list_of_impostors):
        super().__init__(player)
        if player.identity == "Crewmate":
            system_prompt = CREWMATE_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += PERSONALITY_PROMPT.format(
                    personality=CrewmatePersonalities[player.personality]
                )
            system_prompt += CREWMATE_EXAMPLE
            model = random.choice(agent_config["CREWMATE_LLM_CHOICES"])
        elif player.identity == "Impostor":
            system_prompt = IMPOSTOR_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += PERSONALITY_PROMPT.format(
                    personality=ImpostorPersonalities[player.personality]
                )
            system_prompt += IMPOSTOR_EXAMPLE
            system_prompt += f"List of impostors: {list_of_impostors}"
            model = random.choice(agent_config["IMPOSTOR_LLM_CHOICES"])

        self.system_prompt = system_prompt
        self.model = model
        self.temperature = 0.7
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "XXXX"
        self.summarization = "No thought process has been made."
        self.processed_memory = "No world state info yet."
        self.previous_location = None  # Track previous location to prevent loops
        self.chat_history = []
        self.meeting_role = None # Persistent role during a single meeting
        
        # Impostor Deception Ledger
        self.kill_location = None       # Actual room where the kill happened
        self.public_alibi = None        # The room the Impostor will CLAIM to have been in
        self.kill_victim = None         # Name of the victim
        self.tools = tools
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.getenv("EXPERIMENT_PATH") + "/agent-logs.json"
        self.compact_log_path = os.getenv("EXPERIMENT_PATH") + "/agent-logs-compact.json"
        self.game_index = game_index

    def log_interaction(self, sysprompt, prompt, original_response, step):
        """
        Helper method to store model interactions in properly nested JSON format.
        Handles deep nesting and properly parses all string-formatted dictionaries.

        Args:
            prompt (str): The input prompt containing dictionary-like strings
            response (str): The model response containing bracketed sections
            step (str): The game step number
        """

        def parse_dict_string(s):
            if isinstance(s, str):
                # Replace any single quotes with double quotes for valid JSON
                s = s.replace("'", '"')
                s = s.replace('"', '"')
                # Properly escape newlines for JSON
                s = s.replace("\\n", "\\\\n")
                try:
                    # Try parsing as JSON first
                    try:
                        return json.loads(s)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try ast.literal_eval
                        return ast.literal_eval(s)
                except:
                    # If parsing fails, keep original string
                    return s
            return s

        def extract_action(text):
            """Extract action from response text."""
            if "[Action]" in text:
                action_parts = text.split("[Action]")
                thought = action_parts[0].strip()
                action = action_parts[1].strip()
                return {"thought": thought, "action": action}
            return text

        # Parse the prompt
        if isinstance(prompt, str):
            try:
                prompt = parse_dict_string(prompt)
            except:
                pass
        if isinstance(original_response, str):
            sections = {}
            current_section = None
            current_content = []

            for line in original_response.split("\n"):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    if current_section:
                        sections[current_section] = " ".join(current_content).strip()
                        current_content = []
                    current_section = line[1:-1]  # Remove brackets
                elif line and current_section:
                    current_content.append(line)

            if current_section and current_content:
                sections[current_section] = " ".join(current_content).strip()

            new_response = sections if sections else original_response

            # Parse any dictionary strings in the response sections and handle [Action]
            if isinstance(new_response, dict):
                for key, value in new_response.items():
                    if isinstance(value, str):
                        new_response[key] = extract_action(value)
                    else:
                        new_response[key] = parse_dict_string(value)

        # Create the interaction object with proper nesting
        interaction = {
            'game_index': 'Game ' + str(self.game_index),
            'step': step,
            "timestamp": str(datetime.now()),
            "player": {"name": self.player.name, "identity": self.player.identity, "personality": self.player.personality, "model": self.model, "location": self.player.location},
            "interaction": {"system_prompt": sysprompt, "prompt": prompt, "response": new_response, "full_response": original_response},
        }

        # Write to file with minimal whitespace but still readable
        with open(self.log_path, "a") as f:
            json.dump(interaction, f, indent=2, separators=(",", ": "))
            f.write("\n")
            f.flush()
        with open(self.compact_log_path, "a") as f:
            json.dump(interaction, f, separators=(",", ": "))
            f.write("\n")
            f.flush()

        print(".", end="", flush=True)

    async def send_request(self, messages):
        """Send a POST request to OpenRouter API with the provided messages."""
        # Prompt size safety: truncate the user message if it would exceed the context budget.
        # This prevents 400 errors from vLLM/API when prompts exceed max_model_len,
        # and ensures late-turn players (like Player 5) actually get to participate
        # instead of being "silenced" by context starvation.
        max_input_tokens = 12000  # Conservative budget (leaves room for max_tokens=2048 in response)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 characters
        
        if estimated_tokens > max_input_tokens:
            excess_chars = (estimated_tokens - max_input_tokens) * 4
            user_msg = messages[-1]["content"]
            # Smart truncation: try to preserve the Available Actions section at the end
            # (which is critical for the agent to know what it can do)
            actions_marker = user_msg.rfind("Available actions:")
            if actions_marker > 0 and actions_marker > len(user_msg) - excess_chars:
                # Cut from the middle (observation/action history) but keep the tail
                actions_section = user_msg[actions_marker:]
                truncated_head = user_msg[:len(user_msg) - excess_chars - len(actions_section)]
                messages[-1]["content"] = truncated_head + "\n\n[...context truncated for length...]\n\n" + actions_section
            else:
                messages[-1]["content"] = user_msg[:len(user_msg) - excess_chars] + "\n\n[...context truncated for length. Respond based on available information...]"
            print(f"[WARNING] Prompt truncated for {self.player.name}: ~{estimated_tokens} est. tokens → ~{max_input_tokens}")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0,
            "max_tokens": 3072,  # Increased from 2048 to reduce truncation of action lines
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(10):
                try:
                    async with session.post(self.api_url, headers=headers, data=json.dumps(payload)) as response:
                        if response is None:
                            print(f"API request failed: response is None for {self.model}.")
                            continue
                        if response.status == 200:
                            data = await response.json()
                            if "choices" not in data:
                                print(f"API request failed: 'choices' key not in response for {self.model}.")
                                continue
                            if not data["choices"]:
                                print(f"API request failed: 'choices' key is empty in response for {self.model}.")
                                continue
                            # Warn if response was truncated due to token limit
                            finish_reason = data["choices"][0].get("finish_reason", "")
                            content = data["choices"][0]["message"]["content"]
                            if finish_reason == "length":
                                print(f"[WARNING] {self.player.name} response truncated (hit max_tokens).")
                                # Check if the response has an [Action] tag — if not, the truncation killed the action
                                if content and "[Action]" not in content:
                                    print(f"[CRITICAL] {self.player.name} response truncated BEFORE [Action] — action will be inferred from thinking or defaulted.")
                            return content
                except Exception as e:
                    print(f"[API ERROR] Request failed for {self.model} (Attempt {attempt + 1}/10): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"[API ERROR] All attempts failed for {self.model}. Returning None.")
        return None

    def respond(self, message):
        all_info = self.player.all_info_prompt()
        prompt = f"{all_info}\n{message}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.send_request(messages)

    def _assign_meeting_role(self) -> str:
        """Dynamically assign a discussion role based on the player's CURRENT observation history.
        
        Called EVERY discussion round (not just once per meeting) so that roles adapt
        as new information emerges. E.g., a Bystander in Round 1 who gets accused in
        Round 1 discussion becomes a Defender in Round 2.
        
        Priority stack (highest priority first — order matters!):
          1. Accused + Witnessed crime → Counter-Attacker (defend by attacking the real killer)
          2. Accused                   → Defender (survival is #1 instinct)
          3. Witnessed Kill/Vent       → Prosecutor (present hard evidence)
          4. Has location data         → Detective (ask questions, cross-reference)
          5. No strong evidence        → Bystander (listen, evaluate, vouch)
        
        For Impostors:
          - If accused → Defender (must deflect)
          - Otherwise  → Detective or Bystander (blend in, don't over-accuse)
        """
        obs_history = self.player.observation_history
        is_impostor = self.player.identity == "Impostor"
        
        # --- Evidence scan ---
        
        # Check if this player has eyewitness evidence of a crime
        has_witnessed_crime = False
        for obs in obs_history:
            obs_upper = obs.upper()
            if "[CONFIRMED EYEWITNESS]" in obs_upper:
                has_witnessed_crime = True
                break
            if ("KILL" in obs_upper or "VENT" in obs_upper) and "SAW" in obs_upper:
                has_witnessed_crime = True
                break
        
        # Check if this player is being accused by others in discussion
        is_accused = False
        player_name_lower = self.player.name.lower()
        for obs in obs_history:
            obs_lower = obs.lower()
            # Look for accusations in discussion round observations or round summaries
            if ("discussion round" in obs_lower or "said:" in obs_lower or 
                "summary" in obs_lower):
                if player_name_lower in obs_lower and (
                    "suspicious" in obs_lower or "impostor" in obs_lower or
                    "killed" in obs_lower or "vote" in obs_lower or
                    "lying" in obs_lower or "liar" in obs_lower or
                    "accuse" in obs_lower or "sus" in obs_lower or
                    "eject" in obs_lower
                ):
                    is_accused = True
                    break
        
        # Check if player has general location/movement data (weaker evidence)
        has_location_data = any(
            "moved" in obs.lower() or "entered" in obs.lower() or "location" in obs.lower()
            for obs in obs_history
        )
        
        # --- Role assignment (priority stack) ---
        
        if is_impostor:
            # Impostor role strategy: blend in, never Prosecutor
            if is_accused:
                return "Defender"
            elif has_witnessed_crime:
                # Impostor "witnessed" something — useful for framing
                # but Prosecutor is too aggressive and draws attention
                return "Detective"
            else:
                import random
                return random.choice(["Detective", "Bystander"])
        else:
            # Crewmate role assignment — priority order is critical:
            # Accused+Witness is the most important combo (being framed by the killer you caught)
            if is_accused and has_witnessed_crime:
                return "Counter-Attacker"
            elif is_accused:
                return "Defender"
            elif has_witnessed_crime:
                return "Prosecutor"
            elif has_location_data:
                return "Detective"
            else:
                return "Bystander"

    def _normalize_response(self, response: str) -> str:
        """Normalize LLM response to ensure it always has a proper [Action] section.
        
        Fixes common LLM format violations:
        1. [SPEAK: "msg"] or [VOTE Player X] used as section headers → [Action] SPEAK/VOTE
        2. Bare SPEAK: "..." or VOTE ... at end of response without [Action] wrapper
        3. Truncated responses where [Action] never appeared — extract intent from thinking
        4. Responses that are None or empty
        """
        # Handle None/empty responses gracefully
        if not response or not isinstance(response, str):
            return ""
        
        if "[Action]" in response:
            return response  # Already correct format
        
        ACTION_PREFIXES = ("SPEAK", "VOTE", "MOVE", "KILL", "COMPLETE", "CALL", "REPORT", "VENT", "SABOTAGE")
        lines = response.split("\n")
        normalized_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Pattern 1: [SPEAK: "msg"] or [VOTE Player X] as section header
            if stripped.startswith("[") and stripped.endswith("]"):
                inner = stripped[1:-1]  # Remove brackets
                if inner.upper().startswith(ACTION_PREFIXES):
                    # Convert [SPEAK: "msg"] → [Action] SPEAK: "msg"
                    normalized_lines.append(f"[Action] {inner}")
                    continue
            normalized_lines.append(line)
        
        result = "\n".join(normalized_lines)
        
        # Pattern 2: No [Action] found yet — search backwards for a bare action line
        if "[Action]" not in result:
            # Scan ALL lines from end to find an action prefix line
            # (multi-line SPEAK messages mean the action line may not be the last line)
            for i in range(len(normalized_lines) - 1, -1, -1):
                stripped = normalized_lines[i].strip()
                if not stripped:
                    continue
                if stripped.upper().startswith(ACTION_PREFIXES):
                    normalized_lines[i] = f"[Action] {stripped}"
                    result = "\n".join(normalized_lines)
                    break
        
        # Pattern 3: TRUNCATION RECOVERY — response was cut off before [Action].
        # Search the thinking/reasoning text for action-like phrases that indicate intent.
        # This catches cases where max_tokens truncated the response mid-thought.
        if "[Action]" not in result:
            import re
            response_upper = response.upper()
            
            # Look for "I should/will/must [ACTION]" patterns in the thinking
            intent_patterns = [
                # "I should MOVE from X to Y" or "I will MOVE to X"
                r'I (?:SHOULD|WILL|MUST|NEED TO|AM GOING TO)\s+(MOVE\s+(?:FROM\s+\w[\w\s]*?\s+)?TO\s+\w[\w\s]*?)(?:\.|,|\n|$)',
                # "I should COMPLETE TASK" or "I will complete my task"
                r'I (?:SHOULD|WILL|MUST|NEED TO)\s+(COMPLETE\s+(?:FAKE\s+)?TASK(?:\s*-\s*[\w\s]+)?)',
                # "I should KILL Player X"
                r'I (?:SHOULD|WILL|MUST|NEED TO)\s+(KILL\s+PLAYER\s+\d+[:\s]*\w+)',
                # "I should REPORT DEAD BODY" or "I should CALL MEETING"
                r'I (?:SHOULD|WILL|MUST|NEED TO)\s+((?:REPORT\s+DEAD\s+BODY|CALL\s+MEETING)[\w\s]*)',
                # "I should VOTE for Player X" or "I will vote Player X"
                r'I (?:SHOULD|WILL|MUST|NEED TO)\s+(?:VOTE\s+(?:FOR\s+)?)(PLAYER\s+\d+[:\s]*\w+)',
                # "I should VOTE SKIP"
                r'I (?:SHOULD|WILL|MUST|NEED TO)\s+(VOTE\s+SKIP)',
            ]
            
            for pattern in intent_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    extracted_action = match.group(1).strip().rstrip('.')
                    # For VOTE patterns that only captured the target
                    if pattern.find('VOTE') >= 0 and not extracted_action.upper().startswith('VOTE'):
                        extracted_action = f"VOTE {extracted_action}"
                    result += f"\n[Action] {extracted_action}"
                    print(f"[TRUNCATION RECOVERY] {self.player.name}: Extracted action intent from thinking: '{extracted_action}'")
                    break
        
        return result

    def _extract_memory(self, response: str) -> str:
        """Extract the condensed memory from the LLM response."""
        if not isinstance(response, str):
            return self.processed_memory

        # Robust regex to find [World State Ledger] or [Condensed Memory] tag (legacy support), handling:
        # - Case insensitivity
        # - Optional asterisks (**[World State Ledger]**)
        # - Optional colons or whitespace
        import re
        memory_tag_pattern = r"\**\[\s*(World\s+State\s+Ledger|Condensed\s+Memory)\s*\]\**:?"
        
        match = re.search(memory_tag_pattern, response, re.IGNORECASE)
        
        if match:
            try:
                # Extract text starting after the tag
                start_idx = match.end()
                content = response[start_idx:]
                
                # Find the end of the memory block
                # Memory usually ends at the next tag like [Thinking Process] or [Action]
                # Also handle case variations and optional asterisks
                next_tag_pattern = r"\**\[\s*(Thinking\s+Process|Action|Thought)\s*\]\**"
                next_match = re.search(next_tag_pattern, content, re.IGNORECASE)
                
                if next_match:
                    memory_content = content[:next_match.start()].strip()
                else:
                    # If no next tag, assume rest of string is memory
                    memory_content = content.strip()
                
                if memory_content:
                    return memory_content
            except Exception as e:
                print(f"[WARNING] Failed to extract memory for {self.player.name}: {e}")
        
        # If extraction failed or tag missing, keep previous memory to avoid amnesia
        return self.processed_memory

    # ═══════════════════════════════════════════════════════════════════════
    # SPEAKING SCORE HEURISTIC (Post-Generation Validation)
    # Scores LLM speech on a 0–20 scale. Negative scores = hallucination.
    # Applied after generation, before the speech is allowed into the game.
    # ═══════════════════════════════════════════════════════════════════════

    # Room name regex fragment (sorted longest-first for correct alternation)
    _ROOM_PATTERN = "(?:" + "|".join(sorted([
        "cafeteria", "weapons", "navigation", "o2", "shields",
        "communications", "storage", "admin", "electrical",
        "lower engine", "security", "reactor", "upper engine", "medbay"
    ], key=len, reverse=True)) + ")"

    _COLOR_PATTERN = "(?:" + "|".join([
        "red", "blue", "green", "pink", "orange", "yellow",
        "black", "white", "purple", "brown", "cyan", "lime"
    ]) + ")"

    def _compute_valid_truths(self) -> dict:
        """Step 1 of the Speaking Score Heuristic: the 'Reality Check'.

        Pre-computes what this agent ACTUALLY knows from ground-truth game data:
        - Which rooms did I visit?
        - Who did I see in each room?
        - Did I witness a kill or vent?

        This creates the Line-of-Sight (LOS) truth table against which speech
        candidates are scored. The agent can ONLY make claims about rooms in
        rooms_visited; everything else is outside their epistemic boundary.
        """
        truths = {
            "saw_kill": False,
            "saw_vent": False,
            "kill_witness_details": [],
            "vent_witness_details": [],
            "rooms_visited": set(),          # rooms I was physically in (lowercase)
            "players_seen_per_room": {},     # room (lower) → {player names (lower)}
            "timestep_rooms": {},            # timestep → room (lower)
            "is_impostor": self.player.identity == "Impostor",
            "kill_location": getattr(self, 'kill_location', None),
            "kill_victim": getattr(self, 'kill_victim', None),
            "public_alibi": getattr(self, 'public_alibi', None),
        }

        # Scan observation history for witnessed crimes
        for obs in self.player.observation_history:
            obs_upper = obs.upper()
            is_eyewitness = (
                "[CONFIRMED EYEWITNESS]" in obs_upper or
                ("SAW" in obs_upper and ("KILL" in obs_upper or "VENT" in obs_upper))
            )
            if is_eyewitness:
                if "KILL" in obs_upper:
                    truths["saw_kill"] = True
                    truths["kill_witness_details"].append(obs)
                if "VENT" in obs_upper:
                    truths["saw_vent"] = True
                    truths["vent_witness_details"].append(obs)

        # Scan verified presence log for rooms visited and players seen
        for entry in self.player.verified_presence_log:
            ts = entry["timestep"]
            room = entry["room"].lower()
            others = entry["players_seen"]
            truths["rooms_visited"].add(room)
            truths["timestep_rooms"][ts] = room
            if room not in truths["players_seen_per_room"]:
                truths["players_seen_per_room"][room] = set()
            for p in others:
                truths["players_seen_per_room"][room].add(p.lower())

        # Impostors: add alibi room to rooms_visited (intentional deception is allowed)
        if truths["is_impostor"] and truths["public_alibi"]:
            truths["rooms_visited"].add(truths["public_alibi"].lower())

        return truths

    def _extract_thought_from_response(self, response: str) -> str:
        """Extract the THOUGHT block from a CoT-formatted LLM response.

        Returns the internal reasoning (for debug logging) or "" if not found.
        """
        if not response:
            return ""
        import re
        # THOUGHT: ... (everything until SPEAK: or [Action] or end)
        match = re.search(
            r'THOUGHT\s*:\s*(.*?)(?=\nSPEAK\s*:|\n\[Action\]|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        return match.group(1).strip() if match else ""

    def _extract_speech_from_response(self, response: str) -> str:
        """Extract the SPEAK message content from a raw LLM response.

        Supports two formats:
          1. CoT format:  THOUGHT: ...  SPEAK: "public dialogue"
          2. Legacy format: [Action] SPEAK: "public dialogue"
        """
        if not response:
            return ""
        import re

        # ─── CoT format: SPEAK: "..." (standalone, after a THOUGHT block) ───
        # Quoted with double-quotes
        match = re.search(
            r'(?:^|\n)\s*SPEAK\s*:\s*"(.*?)"',
            response, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # Quoted with single-quotes
        match = re.search(
            r"(?:^|\n)\s*SPEAK\s*:\s*'(.*?)'",
            response, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # Unquoted SPEAK: everything to end-of-line
        match = re.search(
            r'(?:^|\n)\s*SPEAK\s*:\s*(.+)',
            response, re.IGNORECASE
        )
        if match:
            text = match.group(1).strip().strip('"').strip("'")
            if text:
                return text

        # ─── Legacy [Action] SPEAK: "..." format ───
        match = re.search(
            r'\[Action\]\s*SPEAK\s*:?\s*"(.*?)"',
            response, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        match = re.search(
            r"\[Action\]\s*SPEAK\s*:?\s*'(.*?)'",
            response, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # No quotes — take everything after SPEAK: to end of line
        match = re.search(
            r'\[Action\]\s*SPEAK\s*:?\s*(.+)',
            response, re.IGNORECASE
        )
        if match:
            return match.group(1).strip().strip('"').strip("'")
        return ""

    def _score_speech(self, speech: str, valid_truths: dict) -> tuple:
        """Step 2 of the Speaking Score Heuristic: score a speech candidate.

        Scoring Table (0–20 positive, negative = hallucination):
          A. Hard Evidence:  Kill Witness +20, Vent Witness +18, Body Camping +15,
                             Hard Alibi +12, Path Contradiction +10
          B. Soft Evidence:  Task Logic +8, Spatial Logic +8, Soft Alibi +6,
                             Sighting +5, Direct Defense +10
          C. Noise/Fluff:    Uncertainty +2, Skip Vote +1, Agreement +1
          D. Hallucination:  X-Ray Vision -100, Meta-Gaming -50,
                             Self-Incrimination -50, Spatial Non-Sequitur -20

        Returns (total_score, [(condition, points, detail), ...]).
        Negative total → speech MUST be rejected and regenerated.
        """
        import re
        score = 0
        breakdown = []
        speech_lower = speech.lower()
        # Normalize whitespace for multi-line speeches
        speech_norm = " ".join(speech.split())
        speech_norm_lower = speech_norm.lower()
        rp = self._ROOM_PATTERN
        cp = self._COLOR_PATTERN

        # ═══════════════════════════════════════════
        # D. HALLUCINATION FILTER (Strict Penalties)
        # ═══════════════════════════════════════════

        # D1. Meta-Gaming (-50): Referencing game mechanics instead of speaking naturally
        meta_patterns = [
            r'\bverified presence log\b',
            r'\bgame engine\b',
            r'\bsystem log\b',
            r'\bmemory stream\b',
            r'\btimestep\s*\d+\b',
            r'\bT\d+\b',          # T0, T1, T2 — game notation, not natural speech
            r'\bobservation history\b',
            r'\baction history\b',
            r'\bpresence log\b',
        ]
        for pat in meta_patterns:
            if re.search(pat, speech, re.IGNORECASE):
                score -= 50
                matched = re.search(pat, speech, re.IGNORECASE).group(0)
                breakdown.append(("META-GAMING", -50, f"Referenced game mechanic: '{matched}'"))
                break

        # D2. Self-Incrimination (-50): Impostor revealing their own crime
        if valid_truths["is_impostor"] and valid_truths.get("kill_location"):
            incrim_pats = [
                r'\bi killed\b', r'\bi did kill\b', r'\bi murdered\b',
                r'\bi vented\b', r'\bi used (?:the )?vent\b',
            ]
            for pat in incrim_pats:
                if re.search(pat, speech, re.IGNORECASE):
                    score -= 50
                    breakdown.append(("SELF-INCRIMINATION", -50, f"Confession detected"))
                    break
            # Revealing actual kill location (not alibi)
            kill_loc = valid_truths["kill_location"].lower()
            alibi = (valid_truths.get("public_alibi") or "").lower()
            if kill_loc and alibi and kill_loc != alibi:
                if re.search(r'\bi was (?:in|at) ' + re.escape(kill_loc) + r'\b', speech_norm_lower):
                    score -= 50
                    breakdown.append(("SELF-INCRIMINATION", -50, f"Revealed kill location: {kill_loc}"))

        # D3. Spatial Non-Sequitur (-20): "I was in Room A, so you weren't in Room B"
        #     Being in one room gives ZERO information about a different room.
        spatial_pat = (
            r'i was (?:in|at) (' + rp + r')'
            r'.*?(?:so|therefore|thus|which means|that means|this means)'
            r'.*?(?:you|they|he|she|player\s*\d+[\w\s:]*?)\s+'
            r'(?:weren\'t|wasn\'t|couldn\'t|could not|were not|was not|can\'t|cannot)\s+'
            r'(?:have been\s+)?(?:in|at)\s+(' + rp + r')'
        )
        spatial_match = re.search(spatial_pat, speech_norm_lower)
        if spatial_match:
            room_a = spatial_match.group(1).strip()
            room_b = spatial_match.group(2).strip()
            if room_a != room_b:
                score -= 20
                breakdown.append(("SPATIAL NON-SEQUITUR", -20,
                                  f"In '{room_a}' → claimed knowledge of '{room_b}'"))

        # D4. X-Ray Vision (-100): Claiming to see events/players in rooms where LOS=False
        #     Crewmates: full check on all first-person location/observation claims
        #     Impostors: alibi room is exempt (intentional deception), but other rooms checked
        rooms_visited = valid_truths["rooms_visited"]

        if not valid_truths["is_impostor"]:
            # --- Crewmate: check "I was in [room]" claims ---
            for match in re.finditer(r'\bi was (?:in|at) (' + rp + r')', speech_norm_lower):
                claimed_room = match.group(1).strip()
                if claimed_room not in rooms_visited:
                    score -= 100
                    breakdown.append(("X-RAY VISION", -100,
                                      f"Claimed to be in '{claimed_room}' (never visited)"))
                    break

        # --- All agents: check "I saw [player/event] in [room]" observation claims ---
        xray_obs_pats = [
            r'\bi (?:saw|noticed|watched|witnessed) .+? (?:in|at) (' + rp + r')',
            r'\bwhen i was (?:in|at) (' + rp + r')',
            r'\bin (' + rp + r'),?\s+i (?:saw|noticed|watched|witnessed)',
        ]
        for pat in xray_obs_pats:
            for match in re.finditer(pat, speech_norm_lower):
                claimed_room = match.group(1).strip()
                if claimed_room not in rooms_visited:
                    score -= 100
                    breakdown.append(("X-RAY VISION", -100,
                                      f"Claimed observation in '{claimed_room}' (never visited)"))
                    break
            if any(b[0] == "X-RAY VISION" for b in breakdown):
                break

        # --- All agents: check "[Player] was NOT in [room]" denial claims ---
        #     You can only deny someone's presence in a room you personally visited.
        not_in_pat = (
            r'(?:player\s*\d+[\w\s:]*?|' + cp + r')\s+'
            r'(?:was not|wasn\'t|were not|weren\'t|couldn\'t have been|could not have been)\s+'
            r'(?:in|at)\s+(' + rp + r')'
        )
        for match in re.finditer(not_in_pat, speech_norm_lower):
            denied_room = match.group(1).strip()
            if denied_room not in rooms_visited:
                score -= 100
                breakdown.append(("X-RAY VISION", -100,
                                  f"Denied player presence in '{denied_room}' (never visited)"))
                break

        # ═══════════════════════════════════════════
        # A. HARD EVIDENCE (High-impact positive)
        # ═══════════════════════════════════════════

        # Kill Witness (+20): Agent saw a kill AND references it in speech
        if valid_truths["saw_kill"]:
            kill_kws = [r'\bkill', r'\bmurder', r'\bstab', r'\battack']
            if any(re.search(kw, speech_norm_lower) for kw in kill_kws):
                # Verify the claim matches LOS (agent was in the room)
                score += 20
                breakdown.append(("KILL WITNESS", +20, "Referenced witnessed kill"))

        # Vent Witness (+18): Agent saw a vent AND references it
        if valid_truths["saw_vent"]:
            if re.search(r'\bvent(?:ed|ing)?\b', speech_norm_lower):
                score += 18
                breakdown.append(("VENT WITNESS", +18, "Referenced witnessed vent"))

        # Hard Alibi (+12): "I was with [player] in [room]" — verified by presence log
        alibi_match = re.search(
            r'\bi was with ([\w\s:]+?) (?:in|at) (' + rp + r')', speech_norm_lower
        )
        if alibi_match:
            claimed_player = alibi_match.group(1).strip()
            claimed_room = alibi_match.group(2).strip()
            if claimed_room in rooms_visited:
                seen = valid_truths["players_seen_per_room"].get(claimed_room, set())
                if any(claimed_player in p for p in seen):
                    score += 12
                    breakdown.append(("HARD ALIBI", +12,
                                      f"Verified: with '{claimed_player}' in '{claimed_room}'"))

        # Path Contradiction (+10): Pointing out impossible room-to-room travel
        contradiction_kws = [
            "how did you get from", "rooms aren't connected", "rooms aren't adjacent",
            "not adjacent", "that's impossible", "couldn't get from",
            "did you vent", "those rooms"
        ]
        if any(kw in speech_norm_lower for kw in contradiction_kws):
            score += 10
            breakdown.append(("PATH CONTRADICTION", +10, "Questioned impossible travel"))

        # ═══════════════════════════════════════════
        # B. SOFT EVIDENCE (Medium-impact positive)
        # ═══════════════════════════════════════════

        # Task Logic (+8): Referencing task bar evidence
        task_kws = ["task bar didn't", "task bar did not", "faking task", "fake task",
                     "bar didn't go up", "bar didn't move", "bar didn't increase"]
        if any(kw in speech_norm_lower for kw in task_kws):
            score += 8
            breakdown.append(("TASK LOGIC", +8, "Referenced task bar evidence"))

        # Spatial Logic (+8): Pointing out distance/time impossibility
        spatial_logic_kws = ["couldn't get from", "can't get from", "too far",
                              "rooms apart", "not enough time"]
        if any(kw in speech_norm_lower for kw in spatial_logic_kws):
            # Don't double-count with PATH CONTRADICTION
            if not any(b[0] == "PATH CONTRADICTION" for b in breakdown):
                score += 8
                breakdown.append(("SPATIAL LOGIC", +8, "Referenced spatial impossibility"))

        # Direct Defense (+10): Offering visual task proof
        defense_kws = ["watch me do", "visual task", "medbay scan", "asteroids",
                        "watch me complete", "i can prove"]
        if any(kw in speech_norm_lower for kw in defense_kws):
            score += 10
            breakdown.append(("DIRECT DEFENSE", +10, "Offered visual proof"))

        # Soft Alibi / Sighting (+5): Reporting seeing a player somewhere
        if re.search(r'\bi saw [\w\s:]+ (?:in|at|near|heading|going)', speech_norm_lower):
            # Only count if it's not already covered by KILL/VENT WITNESS
            if not valid_truths["saw_kill"] and not valid_truths["saw_vent"]:
                score += 5
                breakdown.append(("SIGHTING", +5, "Reported seeing a player"))

        # ═══════════════════════════════════════════
        # C. NOISE & FLUFF (Low-impact fallback)
        # ═══════════════════════════════════════════
        has_substance = any(s > 0 for _, s, *_ in breakdown)
        if not has_substance:
            if "skip" in speech_norm_lower or "don't have enough" in speech_norm_lower:
                score += 1
                breakdown.append(("SKIP VOTE", +1, "Suggested skipping"))
            elif "i agree" in speech_norm_lower or "i think so too" in speech_norm_lower:
                score += 1
                breakdown.append(("AGREEMENT", +1, "Agreed with another player"))
            elif any(kw in speech_norm_lower for kw in
                     ["didn't see", "don't know", "no information", "no evidence",
                      "nothing suspicious", "i have no"]):
                score += 2
                breakdown.append(("UNCERTAINTY", +2, "Expressed lack of information"))
            else:
                score += 2
                breakdown.append(("GENERAL", +2, "Unclassified speech"))

        return score, breakdown

    def _compute_kill_risk_matrix(self, available_actions):
        """Compute per-target kill risk scores for the Impostor.

        Returns a list of dicts sorted by risk_score (lowest first), or None
        if no KILL actions are available.  Each entry contains:
          target, witnesses, exposure, vent_escape, risk_score
        """
        kill_targets = [a for a in available_actions if a.name == "KILL"]
        if not kill_targets:
            return None

        vent_available = any(a.name == "VENT" for a in available_actions)
        num_targets = len(kill_targets)
        witnesses = num_targets - 1  # killing one leaves the rest

        # Exposure score: how often has each crewmate been co-located with us?
        presence_log = getattr(self.player, 'verified_presence_log', [])
        co_location_counts = {}  # player_name -> co-located timestep count
        for entry in presence_log:
            for p in entry["players_seen"]:
                co_location_counts[p] = co_location_counts.get(p, 0) + 1
        total_entries = max(len(presence_log), 1)

        results = []
        for kill_action in kill_targets:
            target_name = kill_action.other_player.name
            # Exposure: fraction of timesteps target has seen us (0.0-1.0)
            exposure = min(1.0, co_location_counts.get(target_name, 0) / total_entries)
            # Witness risk: capped at 1.0
            witness_risk = min(1.0, witnesses * 0.35)
            # No vent escape adds flat risk
            escape = 0.0 if vent_available else 0.25
            # Composite risk score
            risk = min(1.0, witness_risk + 0.4 * exposure + escape)
            results.append({
                "target": target_name,
                "witnesses": witnesses,
                "exposure": round(exposure, 2),
                "vent_escape": vent_available,
                "risk_score": round(risk, 2),
            })

        return sorted(results, key=lambda r: r["risk_score"])

    def _compose_action_prompt(self, timestep: int, available_actions: List[Any]) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Composes the system and user messages for action selection."""
        all_info = self.player.all_info_prompt()
        
        # Determine phase based on available actions
        action_names = [a.name for a in available_actions]
        action_reprs = [repr(a).upper() for a in available_actions]
        
        # --- GHOST LOGIC: Simplified prompt for dead players ---
        if not self.player.is_alive:
            from amongagents.agent.neutral_prompts import GHOST_SYSTEM_PROMPT, GHOST_EXAMPLE
            
            # Build a simple task list for the ghost
            incomplete_tasks = [t for t in self.player.tasks if not t.check_completion()]
            task_list = ", ".join([f"{t.name} (at {t.location})" for t in incomplete_tasks]) if incomplete_tasks else "ALL TASKS COMPLETE"
            
            ghost_system = GHOST_SYSTEM_PROMPT.format(name=self.player.name)
            ghost_system += GHOST_EXAMPLE
            
            # --- Ghost State Injection (user turn 1) ---
            ghost_state_injection = (
                f"[STATE INJECTION] You are {self.player.name}. You are DEAD (GHOST). "
                f"Current room: {self.player.location}. "
                f"Remaining tasks: {task_list}. "
                f"Confirm you have read this state."
            )

            # --- Ghost Fake Acknowledgment (assistant turn) ---
            ghost_ack = (
                f"I have read the state. I am {self.player.name}. "
                f"I am DEAD. I am at {self.player.location}. "
                f"My remaining tasks: {task_list}."
            )

            # --- Ghost Action Request (user turn 2) ---
            ghost_action_request = f"""Good. You are a GHOST. Here is your situation:

- You CANNOT speak in meetings, vote, call meetings, or interact with living players.
- You CANNOT be seen by anyone. You are invisible.
- You CAN walk through walls to ANY room in one step.
- Your ONLY purpose: complete your remaining tasks to help your team win.

Available actions:
{chr(10).join(f'{i+1}. {repr(a)}' for i, a in enumerate(available_actions))}

INSTRUCTIONS: If COMPLETE TASK is available, choose it. Otherwise, MOVE to the room with your nearest task.
Do NOT generate safety checks, suspicion analysis, or observation logic. You are a ghost — just do tasks."""

            messages = [
                {"role": "system", "content": ghost_system},
                {"role": "user", "content": ghost_state_injection},
                {"role": "assistant", "content": ghost_ack},
                {"role": "user", "content": ghost_action_request},
            ]
            return messages, action_names, action_reprs
        
        if "SPEAK" in action_names and "MOVE" not in action_names:
            # === STAGED DEBATE SYSTEM ===
            # Meetings proceed through structured stages:
            #   Stage 0 (Testimony): Share facts only — where were you, who did you see
            #   Stage 1 (Accusation/Defense): Compare testimonies, find contradictions
            #   Stage 2 (Final Arguments): Summarize evidence, state voting intent
            from amongagents.agent.neutral_prompts import MEETING_STAGES, MEETING_PHASE_RULES, DISCUSSION_ROLES, SPEAKING_STYLES
            import random
            
            meeting_stage = getattr(self.player, 'current_meeting_stage', 0)
            stage_info = MEETING_STAGES.get(min(meeting_stage, 2), MEETING_STAGES[2])
            
            # Build phase context: stage instruction first
            phase_context = f"\n\n{stage_info['instruction']}\n"
            
            # Dynamic Role Assignment starting from Stage 1 (Accusation/Defense)
            # Re-evaluated EVERY round (not cached) so roles adapt as new info emerges.
            # E.g., a Bystander in Round 1 who gets accused becomes a Defender in Round 2.
            if meeting_stage >= 1:
                self.meeting_role = self._assign_meeting_role()
                role_desc = DISCUSSION_ROLES[self.meeting_role]
                # Anti-Parrot: assign a unique speaking style based on player index.
                # This ensures two Prosecutors don't produce identical phrasing.
                # Use player name hash for deterministic-per-player but varied assignment.
                style_idx = hash(self.player.name) % len(SPEAKING_STYLES)
                voice_style = SPEAKING_STYLES[style_idx]
                phase_context += f"\n## YOUR DISCUSSION ROLE: {self.meeting_role}\n{role_desc}{voice_style}\n"
            
            # General meeting rules (fact-checking, contradiction detection, etc.)
            phase_context += MEETING_PHASE_RULES
            
            # --- Impostor Deception Ledger: inject alibi constraint during meetings ---
            if self.player.identity == "Impostor":
                # Fake Task Alibi: always show the Impostor their cover story tasks
                incomplete_fake_tasks = [t for t in self.player.tasks if not t.check_completion()]
                completed_fake_tasks = [t for t in self.player.tasks if t.check_completion()]
                if incomplete_fake_tasks or completed_fake_tasks:
                    phase_context += f"\n\n## YOUR FAKE TASK ALIBI (PRIVATE — use these to sound like a real Crewmate) ##"
                    if completed_fake_tasks:
                        phase_context += f"\nTasks you 'completed': " + ", ".join(f"{t.name} at {t.location}" for t in completed_fake_tasks)
                    if incomplete_fake_tasks:
                        phase_context += f"\nTasks you're 'working on': " + ", ".join(f"{t.name} at {t.location}" for t in incomplete_fake_tasks)
                    example_task = incomplete_fake_tasks[0] if incomplete_fake_tasks else completed_fake_tasks[0]
                    phase_context += f"\nWhen asked what you were doing, reference these BY NAME. Example: 'I was doing {example_task.name} in {example_task.location}.'"
                
                # Kill-specific deception details + Pre-calculated Lie Menu
                if self.public_alibi and self.kill_location:
                    phase_context += f"\n\n## 🎭 YOUR DECEPTION LEDGER (PRIVATE -- never reveal this) ##"
                    phase_context += f"\n- You killed {self.kill_victim} at {self.kill_location}."
                    phase_context += f"\n- YOUR PUBLIC ALIBI: You will CLAIM you were in **{self.public_alibi}** at the time of the kill."
                    phase_context += f"\n- FORBIDDEN WORDS: Do NOT say 'I killed', 'I was in {self.kill_location}', or anything that reveals the true kill location."
                    phase_context += f"\n- If someone accuses you, deflect: 'I was in {self.public_alibi} doing tasks. Check with anyone who saw me there.'"
                    
                    # === PRE-CALCULATED LIE MENU ===
                    # Give the Impostor 3 ready-made lies so they don't hallucinate or go passive.
                    # These are calculated from the game state to be plausible.
                    
                    # Lie 1: Safe Alibi (already computed, enhance with task name)
                    fake_task_name = incomplete_fake_tasks[0].name if incomplete_fake_tasks else (
                        completed_fake_tasks[0].name if completed_fake_tasks else "Fix Wiring"
                    )
                    safe_alibi_lie = f"'I was in {self.public_alibi} doing {fake_task_name}. I didn't see anything unusual.'"
                    
                    # Lie 2: Frame Job — pick a living crewmate to accuse
                    # Choose someone the Impostor has NOT been seen with (harder to disprove)
                    import random as _lie_rng
                    presence_log = getattr(self.player, 'verified_presence_log', [])
                    seen_players = set()
                    for entry in presence_log:
                        for p in entry["players_seen"]:
                            seen_players.add(p.lower())
                    
                    # Pick a living non-impostor player name from observations to frame
                    living_player_names = []
                    for obs in self.player.observation_history:
                        # Look for player names mentioned in observations
                        import re as _lie_re
                        for match in _lie_re.finditer(r'Player \d+: \w+', obs):
                            pname = match.group(0)
                            if pname.lower() != self.player.name.lower() and pname.lower() != (self.kill_victim or "").lower():
                                living_player_names.append(pname)
                    
                    # Deduplicate and pick a frame target
                    living_player_names = list(set(living_player_names))
                    if living_player_names:
                        frame_target = _lie_rng.choice(living_player_names)
                        frame_lie = f"'I saw {frame_target} heading toward {self.kill_location} right before the meeting. They were acting suspicious — moving fast, not doing tasks.'"
                    else:
                        frame_lie = f"'Someone was near {self.kill_location} acting suspiciously, but I couldn't see who clearly.'"
                    
                    # Lie 3: Witness Lie — claim victim was alive somewhere recently
                    from amongagents.agent.neutral_prompts import CONNECTION_INFO
                    adjacent_to_kill = []
                    for line in CONNECTION_INFO.split('\n'):
                        if line.startswith(self.kill_location + ' →'):
                            rooms_part = line.split('→')[1].strip()
                            adjacent_to_kill = [r.strip() for r in rooms_part.split(',')]
                            break
                    witness_room = _lie_rng.choice(adjacent_to_kill) if adjacent_to_kill else "Cafeteria"
                    witness_lie = f"'I saw {self.kill_victim} alive near {witness_room} not long ago. The kill must have just happened — whoever did it might still be nearby.'"
                    
                    phase_context += f"\n\n## 🗣️ YOUR LIE MENU (pick one or combine — DO NOT be passive) ##"
                    phase_context += f"\nYou MUST actively participate in discussion. 'I was doing my tasks' is TOO PASSIVE and makes you look suspicious."
                    phase_context += f"\n**Choose from these pre-calculated safe lies:**"
                    phase_context += f"\n  1. SAFE ALIBI: {safe_alibi_lie}"
                    phase_context += f"\n  2. FRAME JOB: {frame_lie}"
                    phase_context += f"\n  3. WITNESS LIE: {witness_lie}"
                    phase_context += f"\n\n**STRATEGY**: In Round 1 (Testimony), use Lie #1 or #3 to establish your story."
                    phase_context += f"\n In Round 2+, use Lie #2 to redirect suspicion. Combine lies for maximum effect."
                    phase_context += f"\n NEVER just say 'I was doing my tasks' with nothing else. That is the #1 way Impostors get caught."
            
            # ═══════════════════════════════════════════════════════════════
            # HARD MEMORY vs SOCIAL MEMORY: Strictly separated information sources.
            # This is the core hallucination prevention mechanism.
            # HARD MEMORY = things the player physically SAW (ground truth).
            # SOCIAL MEMORY = things other players SAID (may be lies).
            # The LLM is FORBIDDEN from claiming to have SEEN something that
            # is only in SOCIAL MEMORY.
            # ═══════════════════════════════════════════════════════════════
            hard_mem = self.player.memory.hard_memory_prompt()
            social_mem = self.player.memory.social_memory_prompt()
            consistency = self.player.memory.consistency_prompt()
            
            if hard_mem or social_mem:
                phase_context += "\n\n" + hard_mem + social_mem

            # Consistency check: show the agent its own previous claims
            if consistency:
                phase_context += consistency

            # Also inject the location-based MEMORY STREAM for alibi cross-referencing
            # (this gives timestamped room visits which complement the event-based memory)
            presence_log = getattr(self.player, 'verified_presence_log', [])
            if presence_log:
                phase_context += "\n## YOUR LOCATION HISTORY (rooms YOU personally visited):\n"
                phase_context += "⚠️ This ONLY covers rooms YOU were in. You have NO information about rooms you did not visit.\n"
                for entry in presence_log[-10:]:  # Last 10 timesteps
                    ts = entry["timestep"]
                    room = entry["room"]
                    others = entry["players_seen"]
                    if others:
                        phase_context += f"  T{ts}: I was at {room} and saw {', '.join(others)}\n"
                    else:
                        phase_context += f"  T{ts}: I was at {room} — no one else was there\n"
                phase_context += "\n**HOW TO USE YOUR MEMORY:**\n"
                phase_context += "- If Player X claims 'I was in Electrical at T2' AND you remember being in Electrical at T2 and did NOT see Player X → Player X is LYING.\n"
                phase_context += "- If Player X claims 'I was in Admin' AND you remember being in Admin WITH Player X → Player X is telling the truth (vouch for them).\n"
                phase_context += "- If Player X claims 'I was in Admin' but you were in Cafeteria at that time → You have NO information about Admin. Do NOT claim they are lying.\n"
                phase_context += "- Moving between NON-ADJACENT rooms in 1 timestep is IMPOSSIBLE without venting (check the Room Adjacency Map).\n"
                
                # ═══════════════════════════════════════════════════════════════
                # TRUTH CHECK: Pre-computed contradiction analysis.
                # The LLM struggles to cross-reference claims vs Memory Stream on
                # its own. We pre-compute what the agent CAN verify and present
                # it as a structured block so the LLM doesn't say "I can't confirm."
                # ═══════════════════════════════════════════════════════════════
                if meeting_stage >= 1:
                    # Build a lookup: room → set of players seen at each timestep
                    my_sightings = {}  # (timestep, room) → set of player names
                    for entry in presence_log:
                        key = (entry["timestep"], entry["room"].lower())
                        my_sightings[key] = {p.lower() for p in entry["players_seen"]}
                    
                    # Scan observation history for claims made by other players
                    import re as _truth_re
                    claim_pattern = _truth_re.compile(
                        r'(Player \d+: \w+) said:.*?(?:I was (?:in|at) (\w[\w\s]*?)(?:\.|,| at| doing| and|$))',
                        _truth_re.IGNORECASE
                    )
                    
                    truth_checks = []
                    for obs in self.player.observation_history:
                        m = claim_pattern.search(obs)
                        if m:
                            claimer = m.group(1)
                            claimed_room = m.group(2).strip().lower()
                            # Check: was I in that room? Did I see the claimer?
                            for (ts, room), seen_set in my_sightings.items():
                                if room == claimed_room:
                                    if claimer.lower() in seen_set or any(claimer.lower() in s for s in seen_set):
                                        truth_checks.append(f"✅ CONFIRMED: {claimer} claims {claimed_room} — I WAS there and I SAW them.")
                                    else:
                                        truth_checks.append(f"❌ LIE DETECTED: {claimer} claims {claimed_room} at T{ts} — I WAS in {claimed_room} at T{ts} and did NOT see them!")
                                    break
                    
                    # Also check: do I know where each player actually was?
                    # This helps the agent make positive claims about what they DID see.
                    phase_context += "\n\n## 🔍 TRUTH CHECK (pre-computed from YOUR Memory Stream):\n"
                    phase_context += "Use this to verify or refute what others claim. This is YOUR evidence.\n"
                    
                    # List who I personally saw and where
                    phase_context += "\n**What YOU personally witnessed:**\n"
                    for entry in presence_log:
                        ts = entry["timestep"]
                        room = entry["room"]
                        others = entry["players_seen"]
                        if others:
                            phase_context += f"  T{ts}: YOU were in {room} with {', '.join(others)}\n"
                    
                    if truth_checks:
                        phase_context += "\n**Claim Verification Results:**\n"
                        for check in truth_checks:
                            phase_context += f"  {check}\n"
                    
                    phase_context += "\n⚠️ If someone claims a room YOU were in and you did NOT see them → they are LYING. Call it out!\n"
                    phase_context += "⚠️ If someone claims a room you were NOT in → you have no information. Say 'I wasn't there.'\n"

            # ═══════════════════════════════════════════════════════════════
            # RECENCY-BIASED VERIFIED HISTORY + CHAIN-OF-THOUGHT ENFORCEMENT
            # Placed LAST in phase_context (closest to the LLM's completion
            # point) so the model cannot ignore it due to "lost in the middle"
            # effects.  The THOUGHT → SPEAK format forces the agent to READ
            # its own history and reason about it BEFORE producing speech.
            # ═══════════════════════════════════════════════════════════════
            verified_hist = self.player.memory.verified_history_prompt()
            if verified_hist:
                phase_context += "\n\n" + verified_hist

            # Knowledge rule (role-aware)
            if self.player.identity == "Impostor":
                phase_context += (
                    "\n## KNOWLEDGE RULES ##\n"
                    "You CANNOT lie about your own VERIFIED HISTORY above — the engine recorded it.\n"
                    "However, you ARE the Impostor. In your THOUGHT, acknowledge the truth (kills, vents, real locations).\n"
                    "Then in your SPEAK, craft a believable lie that does NOT contradict any verifiable public fact.\n"
                )
            else:
                phase_context += (
                    "\n## KNOWLEDGE RULES ##\n"
                    "You CANNOT lie about your own VERIFIED HISTORY above — the engine recorded it.\n"
                    "Your SPEAK must be consistent with your VERIFIED HISTORY. If another player's claim "
                    "contradicts your history, you MUST call it out.\n"
                )

            # Chain-of-Thought format instruction (the recency anchor)
            phase_context += (
                "\n\n══════════════════════════════════════════════\n"
                "  CRITICAL INSTRUCTION — READ BEFORE RESPONDING\n"
                "══════════════════════════════════════════════\n"
                "You MUST generate your response in TWO parts.\n\n"
                "**Part 1 — THOUGHT** (private, never shown to others):\n"
                "  1. READ your VERIFIED LOCATION HISTORY above.\n"
                "  2. State where YOU were during the key events.\n"
                "  3. Compare this to what others said. Are they lying?\n"
            )
            if self.player.identity == "Impostor":
                phase_context += (
                    "  4. Decide on your lie. Pick from your LIE MENU or craft one.\n"
                    "  5. Make sure the lie does NOT contradict any room you publicly visited.\n"
                )
            else:
                phase_context += (
                    "  4. Decide what useful information you can share from your HARD MEMORY.\n"
                    "  5. If you have nothing new, build on previous statements (consistency).\n"
                )
            phase_context += (
                "\n**Part 2 — SPEAK** (public, everyone hears this):\n"
                "  Generate your public dialogue based on your thought process.\n\n"
                "**RESPONSE FORMAT (you MUST follow this exactly):**\n"
                "THOUGHT: [your internal reasoning — reference your verified history]\n"
                "SPEAK: \"[your public statement]\"\n\n"
                "Example:\n"
                "THOUGHT: My history shows I was in Weapons at T6 and T7. Black claims I was in O2. "
                "That contradicts my history — Black is lying or confused.\n"
                "SPEAK: \"Black, you are mistaken. I was in Weapons the entire time doing my task.\"\n"
                "══════════════════════════════════════════════\n"
            )

        elif "VOTE" in action_names:
            # Reset meeting role after voting
            self.meeting_role = None
            
            # === BUILD CONSTRAINED VOTING PROMPT ===
            # Explicit active/deceased lists + ONLY valid names to prevent nan/hallucination
            vote_targets = [a for a in available_actions if a.name == "VOTE"]
            valid_target_names = [a.other_player.name for a in vote_targets]
            
            phase_context = "\n\n## ═══════════════════════════════════════ ##\n"
            phase_context += "## ⚠️  VOTING PHASE — CHOOSE ONE PLAYER  ⚠️ ##\n"
            phase_context += "## ═══════════════════════════════════════ ##\n\n"
            
            # Active vs Deceased roster (grounding the LLM on who can be voted for)
            phase_context += f"ACTIVE PLAYERS (can be voted for): {', '.join(valid_target_names)}\n"
            phase_context += f"YOU ({self.player.name}) are voting. You CANNOT vote for yourself.\n\n"
            
            phase_context += "**INSTRUCTIONS:**\n"
            phase_context += "1. Review the discussion that just happened.\n"
            phase_context += "2. Review YOUR secret knowledge below.\n"
            phase_context += "3. In your [Thinking Process], you MUST state your reasoning BEFORE voting.\n"
            phase_context += "4. Pick ONE player from the ACTIVE PLAYERS list above, or SKIP.\n"
            phase_context += f"5. Your action MUST be exactly: VOTE [name] — where [name] is one of: {', '.join(valid_target_names)}\n"
            phase_context += "   OR: VOTE SKIP (if you don't want to vote anyone out)\n\n"
            
            # === EVIDENCE THRESHOLD: Anti-Bandwagon Logic ===
            # Check if the player has any HARD evidence (witnessed crime, confirmed contradiction)
            has_hard_evidence = False
            evidence_target = None
            for obs in self.player.observation_history:
                obs_upper = obs.upper()
                if "[CONFIRMED EYEWITNESS]" in obs_upper:
                    has_hard_evidence = True
                    break
                if "KILL" in obs_upper and "SAW" in obs_upper:
                    has_hard_evidence = True
                    break
                if "VENT" in obs_upper and "SAW" in obs_upper:
                    has_hard_evidence = True
                    break
            
            if not has_hard_evidence and self.player.identity == "Crewmate":
                phase_context += "## ⚠️ EVIDENCE CHECK — READ THIS BEFORE VOTING ##\n"
                phase_context += "You have NO hard evidence (you did NOT witness a kill or vent).\n"
                phase_context += "**STRONG RECOMMENDATION: VOTE SKIP** unless:\n"
                phase_context += "  - Someone else presented a CONFIRMED eyewitness account during discussion\n"
                phase_context += "  - You found a CONFIRMED CONTRADICTION (someone lied about a room you were in)\n"
                phase_context += "  - A player refused to answer direct questions or had no alibi\n"
                phase_context += "Voting out an innocent Crewmate HELPS the Impostor. When in doubt, SKIP.\n"
                phase_context += "Do NOT bandwagon: just because others accuse someone doesn't make it true.\n\n"
            
            # === DISCUSSION SUMMARY: what was said ===
            # Extract the last discussion summaries so the agent remembers claims
            discussion_summaries = []
            for obs in self.player.observation_history:
                if "Discussion Summary" in obs or "said:" in obs:
                    discussion_summaries.append(obs)
            if discussion_summaries:
                phase_context += "## DISCUSSION RECAP (what was said before this vote):\n"
                for summary in discussion_summaries[-12:]:  # Last 12 relevant lines
                    phase_context += f"  {summary}\n"
                phase_context += "\n"
            
            # === VERIFIED HISTORY FOR VOTING ===
            verified_hist = self.player.memory.verified_history_prompt()
            if verified_hist:
                phase_context += verified_hist

            # === STRUCTURED MEMORY FOR VOTING ===
            # Uses the same HARD MEMORY / SOCIAL MEMORY separation as discussion.
            # HARD MEMORY = eyewitness evidence that overrides all discussion claims.
            # SOCIAL MEMORY = what others said during discussion — may be lies.
            hard_mem = self.player.memory.hard_memory_prompt()
            social_mem = self.player.memory.social_memory_prompt()

            if hard_mem:
                phase_context += "## 🚨 YOUR HARD MEMORY — THIS OVERRIDES EVERYTHING ELSE 🚨 ##\n"
                phase_context += "These are events you personally WITNESSED. They are 100% FACT.\n"
                phase_context += "Vote based on THIS evidence, NOT on what other players said:\n"
                phase_context += hard_mem
                phase_context += "⚠️ Even if EVERY other player disagrees with you, YOUR EYES are more reliable than their words.\n\n"

            if social_mem:
                phase_context += social_mem

            # Also inject location-based MEMORY STREAM for voting cross-reference
            presence_log = getattr(self.player, 'verified_presence_log', [])
            if presence_log:
                phase_context += "## YOUR LOCATION HISTORY (rooms YOU visited — only these are verifiable):\n"
                for entry in presence_log[-8:]:
                    ts = entry["timestep"]
                    room = entry["room"]
                    others = entry["players_seen"]
                    if others:
                        phase_context += f"  T{ts}: I was at {room} and saw {', '.join(others)}\n"
                    else:
                        phase_context += f"  T{ts}: I was at {room} — no one else was there\n"
                phase_context += "\n"
            
            # Impostor voting guidance: vote strategically, not for yourself
            if self.player.identity == "Impostor":
                phase_context += "## IMPOSTOR VOTING STRATEGY ##\n"
                phase_context += "- NEVER vote for yourself or another Impostor.\n"
                phase_context += "- Vote for whichever Crewmate is most suspicious to the group (to blend in).\n"
                phase_context += "- If the group consensus targets a Crewmate, vote with the group to look normal.\n\n"
            
            phase_context += f"FINAL REMINDER: Your [Action] line MUST be one of:\n"
            for tn in valid_target_names:
                phase_context += f"  VOTE {tn}\n"
            phase_context += "  VOTE SKIP\n"
            phase_context += "\nDo NOT output any other action type. Do NOT output MOVE, SPEAK, COMPLETE TASK, or any player name not listed above."
        else:
            # Task phase
            phase_context = f"\n\n## ⚠️ TASK PHASE ⚠️\nYou are currently at {self.player.location}."
            if self.previous_location and self.previous_location != self.player.location:
                phase_context += f" You just moved from {self.previous_location}."
            
            # ALIVE STATUS ENFORCEMENT: Prevent LLMs from hallucinating ghost state
            # when they have no tasks left. This is a direct fix for the Game 3 bug
            # where Player 2 (Lime) thought they were a ghost while still alive.
            if self.player.is_alive:
                phase_context += "\n⚠️ REMINDER: You are ALIVE. You are a living player in this game. You are NOT a ghost. Do NOT think or act as if you are dead."
                # Check if all tasks are complete
                all_tasks_done = all(t.check_completion() for t in self.player.tasks) if len(self.player.tasks) > 0 else False
                if all_tasks_done and self.player.identity == "Crewmate":
                    phase_context += "\n🎯 All your tasks are DONE. Your new role: WATCHDOG. Follow other players, stay in groups, and call a meeting if you see anything suspicious."
            
            phase_context += "\n\n## CRITICAL TASK RULES:\n1. Maintain your [World State Ledger] to track occupancy and trust.\n2. Follow the Safety -> Observation -> Goal decision hierarchy.\n3. Before choosing MOVE, verify you haven't already reached your destination."
            
            # Multi-turn task commitment: if a task is in progress, warn about leaving
            in_progress_tasks = [t for t in self.player.tasks 
                                 if not t.check_completion() and t.duration < t.max_duration]
            if in_progress_tasks:
                task = in_progress_tasks[0]
                phase_context += f"\n\n⏳ **TASK IN PROGRESS**: You are working on '{task.name}' ({task.duration} turn{'s' if task.duration > 1 else ''} remaining). You are LOCKED to this room until the task finishes. Choose COMPLETE TASK to continue. Your progress is saved if you are forced to leave (body report / sabotage)."
            
            # Add room connection info so agents can distinguish normal movement from venting
            from amongagents.agent.neutral_prompts import CONNECTION_INFO
            phase_context += f"\n\n{CONNECTION_INFO}"
            
            # ═══ GLOBAL EMERGENCY: Critical sabotage overrides all task behavior ═══
            # When O2/Reactor is sabotaged, crewmates MUST drop everything and run to fix it.
            active_sabs = getattr(self.player, 'active_sabotages', set())
            critical_sab = None
            for sab in ("OXYGEN", "REACTOR"):
                if sab in active_sabs:
                    critical_sab = sab
                    break
            
            if critical_sab and self.player.identity == "Crewmate":
                from amongagents.envs.action import FixSabotage
                fix_room = FixSabotage.FIX_LOCATIONS.get(critical_sab, "unknown")
                fix_available = "FIX SABOTAGE" in action_names
                
                phase_context += f"\n\n🚨🚨🚨 **EMERGENCY ACTIVE — {critical_sab} SABOTAGE** 🚨🚨🚨"
                phase_context += f"\nThe ship's {critical_sab} system has been sabotaged! If not fixed, EVERYONE DIES."
                phase_context += f"\n**You CANNOT do tasks right now.** All task actions have been disabled."
                if fix_available:
                    phase_context += f"\n✅ You are at {fix_room}! Choose **FIX SABOTAGE** to save the crew!"
                else:
                    phase_context += f"\n⚠️ You MUST run to **{fix_room}** immediately and choose FIX SABOTAGE."
                    phase_context += f"\nDrop EVERYTHING. Do NOT do tasks. Do NOT investigate. RUN to {fix_room} NOW."
            elif critical_sab and self.player.identity == "Impostor":
                phase_context += f"\n\n🎭 **YOUR SABOTAGE IS ACTIVE**: {critical_sab} is sabotaged! Crewmates are panicking and running to fix it."
                phase_context += f"\nThis is your chance — crewmates are scattered and distracted. Find one alone and KILL them."
                phase_context += f"\nDo NOT fix the sabotage yourself (unless you need to look innocent)."
            
            # Task Commitment for Crewmates: if COMPLETE TASK is available, strongly push it
            # (Only applies when NO critical sabotage is active)
            if self.player.identity == "Crewmate" and not critical_sab:
                task_available = "COMPLETE TASK" in action_names
                has_witnessed = self.player.has_witnessed_crime()
                if task_available and not has_witnessed:
                    phase_context += "\n\n✅ **TASK COMMITMENT**: You are at a room with YOUR task available! You MUST choose COMPLETE TASK. Completing tasks is how Crewmates WIN. Do NOT leave to follow someone suspicious or wander. Finish your task FIRST, then move on. The only exception is if you have witnessed a KILL or VENT (you haven't)."
                elif task_available and has_witnessed:
                    phase_context += "\n\n✅ **TASK AVAILABLE**: You have a task here, but you also have witnessed a crime. You may choose to COMPLETE TASK or REPORT/MOVE to Cafeteria -- use your judgment."
            
            # Kill Urgency + Vent Escape Routes for Impostors
            # Dynamic aggression: kill desire scales with task bar completion percentage
            if self.player.identity == "Impostor":
                # Count ACTUAL kill targets (= crewmates in the room)
                kill_targets = [a for a in available_actions if a.name == "KILL"]
                kill_available = len(kill_targets) > 0
                num_targets = len(kill_targets)
                vent_available = "VENT" in action_names
                
                # Build the target list explicitly so the LLM doesn't hallucinate ghost targets
                target_names = [a.other_player.name for a in kill_targets]
                
                # Show vent escape routes from current room
                vent_info = ""
                if vent_available:
                    vent_destinations = [repr(a) for a in available_actions if a.name == "VENT"]
                    vent_info = f" After killing, you can ESCAPE via vent: {', '.join(vent_destinations)}."
                
                # --- Dynamic Kill Desire: 3-tier Desperation Scaling ---
                # Extract task completion % from room context (injected by create_location_message)
                import re as _re_kill
                _task_bar_match = _re_kill.search(r'Global Task Bar:\s*(\d+)%', all_info)
                task_pct = int(_task_bar_match.group(1)) if _task_bar_match else 0
                
                # Determine aggression tier (matches the user's Desperation Scaling spec)
                if task_pct >= 80:
                    kill_urgency_tier = "CRITICAL"  # PANIC PHASE: Kill immediately, ignore witnesses
                    phase_context += f"\n\n🚨🚨 **PANIC PHASE — TASK BAR AT {task_pct}%** 🚨🚨"
                    phase_context += f"\nCRITICAL: YOU ARE ABOUT TO LOSE. The Crewmates are almost done with tasks."
                    phase_context += f"\nKILL IMMEDIATELY. IGNORE WITNESSES. This is a SUICIDE RUN — if you don't kill NOW, you lose the game."
                    phase_context += f"\nDo NOT waste time on sabotage, fake tasks, or positioning. Find the nearest Crewmate and KILL them."
                elif task_pct >= 50:
                    kill_urgency_tier = "HIGH"       # AGGRESSIVE PHASE: Take risks, accept witnesses
                    phase_context += f"\n\n⚠️ **AGGRESSIVE PHASE — TASK BAR AT {task_pct}%**"
                    phase_context += f"\nThe crew is WINNING. You must take risks. Killing with 1 witness is acceptable if you can blame them."
                    phase_context += f"\nUse sabotage to scatter Crewmates, then kill a straggler. Do NOT wait for the 'perfect' opportunity."
                else:
                    kill_urgency_tier = "NORMAL"     # STEALTH PHASE: Play it safe
                    phase_context += f"\n\n🎭 **STEALTH PHASE — TASK BAR AT {task_pct}%**"
                    phase_context += f"\nPlay it safe. Look for isolated targets. Fake tasks to build an alibi."
                    phase_context += f"\nDo NOT kill with witnesses unless you can vent away immediately."
                
                # ═══ KILL RISK MATRIX: per-crewmate risk assessment ═══
                risk_matrix = self._compute_kill_risk_matrix(available_actions)
                if risk_matrix:
                    phase_context += "\n\n## 🎯 KILL RISK MATRIX (per-crewmate risk assessment)"
                    phase_context += "\n| Target | Witnesses | Exposure | Vent? | Risk Score |"
                    phase_context += "\n|--------|-----------|----------|-------|------------|"
                    for entry in risk_matrix:
                        risk_label = ("🟢 LOW" if entry["risk_score"] < 0.3
                                      else "🟡 MED" if entry["risk_score"] < 0.6
                                      else "🔴 HIGH")
                        vent_str = "✅" if entry["vent_escape"] else "❌"
                        phase_context += (
                            f"\n| {entry['target']} | {entry['witnesses']} | "
                            f"{entry['exposure']:.0%} | {vent_str} | "
                            f"{entry['risk_score']:.2f} {risk_label} |")
                    phase_context += "\n\n**STRATEGY**: Pick the target with the LOWEST risk score."
                    phase_context += " Risk > 0.6 = do NOT kill. Risk < 0.3 = safe to kill."

                if kill_available:
                    # Accurate witness count: total targets = total crewmates in room.
                    # If you kill one, the others are witnesses.
                    witnesses = num_targets - 1  # killing 1 leaves (num_targets - 1) alive witnesses
                    
                    if witnesses == 0:
                        # IDEAL: truly alone with ONE crewmate
                        phase_context += f"\n\n🔪 **KILL OPPORTUNITY**: You are ALONE with {target_names[0]} — NO witnesses! This is your chance.{vent_info}"
                        phase_context += f"\nVALID KILL TARGETS: {', '.join(target_names)}. You can ONLY kill players in this list."
                        if kill_urgency_tier in ("HIGH", "CRITICAL"):
                            phase_context += f"\n🚨 **TASK BAR AT {task_pct}%** — You are LOSING the game. You MUST kill NOW. There is no time to wait for a 'better' opportunity."
                    else:
                        # RISKY: other crewmates will see the kill
                        if kill_urgency_tier == "CRITICAL":
                            # Maximum aggression: kill even with witnesses
                            phase_context += f"\n\n🚨🔪 **DESPERATE KILL — TASK BAR AT {task_pct}%**: The Crewmates are about to WIN. There are {num_targets} players here: {', '.join(target_names)}."
                            phase_context += f" Yes, {witnesses} will witness it — but if you DON'T kill, you LOSE ANYWAY. Kill NOW and vent immediately.{vent_info}"
                            phase_context += f"\nVALID KILL TARGETS: {', '.join(target_names)}. You can ONLY kill players in this list."
                        elif kill_urgency_tier == "HIGH" and vent_available:
                            # High aggression: kill with witnesses if vent escape exists
                            phase_context += f"\n\n⚠️🔪 **RISKY KILL — TASK BAR AT {task_pct}%**: There are {num_targets} Crewmates here: {', '.join(target_names)}."
                            phase_context += f" Normally this would be too risky, but the task bar is dangerously high. Kill and VENT immediately to escape.{vent_info}"
                            phase_context += f"\nVALID KILL TARGETS: {', '.join(target_names)}. You can ONLY kill players in this list."
                        else:
                            # Standard risk assessment
                            phase_context += f"\n\n⚠️ **KILL RISK**: There are {num_targets} Crewmates in this room: {', '.join(target_names)}."
                            phase_context += f" If you kill one, the other {witnesses} will be WITNESSES and will report you immediately."
                            phase_context += f"\n**RECOMMENDATION**: Do NOT kill here. Move to a room with only 1 Crewmate, or use SABOTAGE to scatter them first."
                            if vent_info:
                                phase_context += f"\n(If you still want to try:{vent_info})"
                            phase_context += f"\nVALID KILL TARGETS (if you choose to risk it): {', '.join(target_names)}. You can ONLY kill players in this list."
                else:
                    # No kill targets in this room — urge the Impostor to find one
                    if kill_urgency_tier == "CRITICAL":
                        phase_context += f"\n\n🚨🚨 **EMERGENCY — TASK BAR AT {task_pct}%**: The Crewmates are about to finish ALL tasks. You WILL LOSE if you don't kill IMMEDIATELY. MOVE to the nearest low-traffic room (Electrical, Navigation, Shields, Storage) RIGHT NOW. Do NOT fake tasks, do NOT sabotage — FIND someone alone and KILL them. This is your LAST CHANCE."
                    elif kill_urgency_tier == "HIGH":
                        phase_context += f"\n\n🚨 **KILL URGENCY — TASK BAR AT {task_pct}%**: The Crewmates are winning. You MUST find an isolated Crewmate and kill them NOW. Move to low-traffic rooms (Electrical, Navigation, Shields) where someone might be alone. After killing, use VENT to escape. Every turn you waste is a turn closer to LOSING."
                    elif timestep >= 5:
                        phase_context += f"\n\n⚠️ **KILL URGENCY**: It is timestep {timestep} and you have not killed yet. The Crewmates are completing tasks. You MUST seek out isolated Crewmates and KILL them. Move to low-traffic rooms (Electrical, Navigation, Shields) where you can find someone alone. Remember: after killing, use VENT to escape to a different room."
                
                # Sabotage Priority: check task bar and crew grouping
                sabotage_available = "SABOTAGE" in action_names
                if sabotage_available and not kill_available:
                    if kill_urgency_tier in ("HIGH", "CRITICAL"):
                        phase_context += f"\n\n🔧 **SABOTAGE vs KILL**: The task bar is at {task_pct}%. Sabotage can buy time, but KILLING is more important. If there is ANY chance to reach a Crewmate alone, MOVE toward them instead of sabotaging. Only sabotage if all Crewmates are grouped and you cannot isolate anyone."
                    elif task_pct >= 50:
                        phase_context += "\n\n🔧 **SABOTAGE URGENT**: The task bar is above 50%! Crewmates are winning the task race. If you cannot KILL right now, use SABOTAGE to disrupt their progress. SABOTAGE OXYGEN or REACTOR to force them away from their tasks."
                    elif num_targets == 0:
                        phase_context += "\n\n🔧 **SABOTAGE OPPORTUNITY**: No Crewmates nearby to kill. Use SABOTAGE OXYGEN or REACTOR to scatter them, then pick off a straggler who ends up alone."

        # --- Emergency Prompt Injection ---
        emergency_context = ""
        
        # Detect dead body via action_reprs OR room context in all_info
        body_in_action_reprs = any("REPORT DEAD BODY" in r for r in action_reprs)
        body_in_room_context = "DEAD BODY" in all_info.upper()
        can_report = any("CALL MEETING" in name or "REPORT DEAD BODY" in r for name, r in zip(action_names, action_reprs))
        
        if body_in_action_reprs or (body_in_room_context and can_report):
            emergency_context += "\n\n⚠️ EMERGENCY: There is a DEAD BODY in your room! Your ONLY legal actions are: REPORT DEAD BODY or CALL MEETING. All other actions (MOVE, COMPLETE TASK, etc.) are FORBIDDEN until you report."
        elif body_in_room_context and not can_report:
            # Body is visible but reporting/meeting is not available (e.g. button limit reached, body already reported)
            emergency_context += "\n\n⚠️ WARNING: There is a DEAD BODY in your room, but REPORT DEAD BODY and CALL MEETING are NOT available right now (the emergency button limit may have been reached). You MUST pick an action from the Available actions list. Move to Cafeteria if the emergency button is still usable, or continue with your tasks. Do NOT output REPORT DEAD BODY or CALL MEETING -- they are not in your available actions."
        
        is_impostor = self.player.identity == "Impostor"
        if not is_impostor:
            # IMPORTANT: Only use the player's actual observation_history (ground truth from game engine)
            # to detect witnessed crimes. Do NOT use self.processed_memory (LLM's self-maintained ledger)
            # because stale mentions of "kill" in memory cause infinite "go to Cafeteria" loops.
            has_witnessed = self.player.has_witnessed_crime()
            if has_witnessed:
                if can_report:
                    emergency_context += "\n\n⚠️ URGENT: You have witnessed someone VENTING or KILLING! That person is the IMPOSTOR. You should CALL AN EMERGENCY MEETING now (available at Cafeteria or near body)."
                else:
                    if self.player.location != "Cafeteria":
                        emergency_context += "\n\n⚠️ URGENT: You have evidence of the Impostor! You should MOVE to the Cafeteria to use the emergency button as soon as possible. Do NOT try to CALL MEETING until you reach the Cafeteria."
                    else:
                        # Already at Cafeteria but can't call meeting (button limit reached) -- stop looping
                        emergency_context += "\n\n⚠️ NOTE: You have evidence of the Impostor but the emergency button is unavailable. Focus on completing your tasks and staying safe. Do NOT keep trying to call a meeting."
        
        if emergency_context:
            phase_context += emergency_context

        # Determine phase name for logging purposes
        phase = "Meeting phase - DISCUSSION" if "SPEAK" in action_names and "MOVE" not in action_names \
                else "Meeting phase - VOTING" if "VOTE" in action_names \
                else "Task phase"

        # Identity reinforcement: mark self as "YOU" in the living players list
        # so the LLM never forgets which player it is
        player_name = self.player.name
        all_info = all_info.replace(
            f"Living Players in {self.player.location}: {player_name}",
            f"Living Players in {self.player.location}: YOU ({player_name})"
        ).replace(
            f", {player_name}",
            f", YOU ({player_name})"
        )
        
        # Add explicit identity + location reminder at the start of user message.
        identity_reminder = f"REMINDER: You are {player_name}. YOUR CURRENT ROOM IS: {self.player.location}. Use 'I/me/my' when referring to yourself. Never say '{player_name}' in the third person.\n\n"
        
        # ═══════════════════════════════════════════════════════════════
        # STRUCTURED MEMORY STATE: JSON block injected into the prompt.
        # This is the "Player Memory State" — the ONLY source of truth
        # the agent should use for reasoning. Replaces free-form text
        # recall with machine-verified structured data.
        # ═══════════════════════════════════════════════════════════════
        import json as _json
        memory_state = self.player.get_memory_state_json()
        memory_json_str = _json.dumps(memory_state, indent=2)
        
        state_block = f"[PLAYER MEMORY STATE — Ground Truth, DO NOT contradict this]\n{memory_json_str}\n"

        # ─── TASK COMMITMENT + CRISIS DISPATCH (from MemoryState) ───
        commitment_block = self.player.memory.commitment_prompt()
        crisis_block = self.player.memory.crisis_prompt()
        
        # ─── DANGER SCORE (Crewmate self-preservation) ───
        # Injected during task phase for Crewmates to override task-first behavior
        danger_block = ""
        if phase == "Task phase" and self.player.identity == "Crewmate" and self.player.is_alive:
            danger = self.player.get_danger_score()
            if danger >= 60:
                danger_block = f"\n⚠️ DANGER LEVEL: {danger}/100 — HIGH. You are unsafe. Seek a group or move to a populated room immediately. Do NOT stay alone.\n"
            elif danger >= 30:
                danger_block = f"\n⚠️ DANGER LEVEL: {danger}/100 — MODERATE. Be alert. Complete your task quickly and move to a safer area.\n"
        
        # ─── ALIBI LEDGER (Crewmate meeting precision) ───
        # Forces Crewmates to construct alibis from verified memory, not imagination
        alibi_block = ""
        is_discussion = "SPEAK" in action_names and "MOVE" not in action_names
        if is_discussion and self.player.identity == "Crewmate":
            alibi = self.player.get_alibi_ledger()
            if alibi:
                alibi_block = "\n[YOUR ALIBI — construct your testimony from ONLY these facts]\n"
                for a in alibi:
                    witnesses = ", ".join(a["witnesses"]) if a["witnesses"] else "no one"
                    alibi_block += f"  T{a['turn']}: I was at {a['location']} (saw: {witnesses})\n"
                alibi_block += "RULE: If a claim you want to make is NOT in this alibi, do NOT say it. You will be caught lying.\n"
        
        # ─── CONTRADICTION CHECKER (auto-detect lies from other players) ───
        # During Stage 1+ of meetings, cross-reference other players' claims
        contradiction_block = ""
        meeting_stage = getattr(self.player, 'current_meeting_stage', 0)
        if is_discussion and meeting_stage >= 1:
            # Extract claims from meeting notes in observation_history
            import re as _re_contra
            claims = []
            for obs in self.player.observation_history:
                if "said:" in obs:
                    # Try to extract location claims: "Player X said: I was in [Room]"
                    match = _re_contra.search(r'(Player \d+: \w+) said:.*?(?:I was (?:in|at) )(\w[\w\s]*?)(?:\.|,|$)', obs)
                    if match:
                        claimer = match.group(1)
                        claimed_loc = match.group(2).strip()
                        # Try to extract a turn reference
                        turn_match = _re_contra.search(r'(?:T|turn|timestep)\s*(\d+)', obs, _re_contra.IGNORECASE)
                        turn = int(turn_match.group(1)) if turn_match else None
                        if turn is not None:
                            claims.append({"player": claimer, "claimed_location": claimed_loc, "turn": turn})
            
            if claims:
                results = self.player.check_contradictions(claims)
                if results:
                    contradiction_block = "\n[CONTRADICTION CHECK — engine-verified cross-reference]\n"
                    for r in results:
                        if r["type"] == "HARD_LIE":
                            contradiction_block += f"  ❌ HARD LIE: {r['player']} claims {r['claim']} — BUT {r['evidence']}\n"
                        elif r["type"] == "CONFIRMED":
                            contradiction_block += f"  ✅ CONFIRMED: {r['player']} claims {r['claim']} — {r['evidence']}\n"
        
        # ─── PHANTOM ALIBI (Impostor meeting deception) ───
        # Give the Impostor their pre-built fake history for consistent lies
        phantom_block = ""
        if is_discussion and self.player.identity == "Impostor" and hasattr(self.player, 'fake_memory'):
            fake_alibi = self.player.get_fake_alibi_for_meeting()
            if isinstance(fake_alibi, list) and fake_alibi:
                phantom_block = "\n[YOUR FAKE ALIBI — use this story during the meeting, NEVER reveal the truth]\n"
                for entry in fake_alibi:
                    phantom_block += f"  {entry}\n"
                phantom_block += "RULE: Your statements must match this fake timeline. Do NOT improvise a different story.\n"
        
        # ─── THIRD IMPOSTOR STRATEGY (exploit Crewmate mistakes) ───
        third_impostor_block = ""
        if is_discussion and self.player.identity == "Impostor" and meeting_stage >= 1:
            # Scan discussion for crewmates accusing other crewmates
            wrong_accusations = []
            for obs in self.player.observation_history:
                if "said:" in obs and ("suspicious" in obs.lower() or "vote" in obs.lower()):
                    # Check if the accused is NOT an impostor (from the Impostor's perspective)
                    import re as _re_third
                    match = _re_third.search(r'(Player \d+: \w+) said:.*?(?:vote|voting|suspicious).*?(Player \d+: \w+)', obs, _re_third.IGNORECASE)
                    if match:
                        accuser = match.group(1)
                        accused = match.group(2)
                        if accuser.lower() != self.player.name.lower() and accused.lower() != self.player.name.lower():
                            wrong_accusations.append((accuser, accused))
            
            if wrong_accusations:
                third_impostor_block = "\n[THIRD IMPOSTOR OPPORTUNITY — a Crewmate is doing your job for you]\n"
                for accuser, accused in wrong_accusations[:2]:
                    third_impostor_block += f"  {accuser} is accusing {accused}. SUPPORT this accusation to get a free kill via vote.\n"
                third_impostor_block += "STRATEGY: Agree with the wrong accusation. Let Crewmates eliminate each other.\n"

        # ═══════════════════════════════════════════════════════════════
        # MULTI-TURN STATE ACKNOWLEDGMENT
        # Split into 4 messages: system, state-injection (user),
        # fake-ack (assistant), action-request (user).
        # The fake assistant turn forces the LLM to treat the memory
        # state as already-accepted context, drastically improving
        # adherence to the structured JSON.
        # ═══════════════════════════════════════════════════════════════

        # --- Message 2: State Injection (user) ---
        state_injection = (
            f"{identity_reminder}"
            f"[STATE INJECTION] Here is your current memory state. This is engine-verified ground truth.\n"
            f"{state_block}\n"
            f"Confirm you have read this state."
        )

        # --- Message 3: Fake Acknowledgment (assistant) ---
        # Deterministic string built from memory_state fields. NOT an LLM call.
        status_str = memory_state["my_identity"]["status"].upper()
        vis = memory_state["current_perception"]["visible_players"]
        vis_str = ", ".join(vis) if vis else "no one"
        bodies = memory_state["current_perception"]["dead_bodies"]
        bodies_str = f" Dead bodies here: {', '.join(bodies)}." if bodies else ""

        ack_parts = [
            f"I have read the state.",
            f"I am {player_name}.",
            f"I am at {self.player.location}. Status: {status_str}.{bodies_str}",
            f"Visible players: {vis_str}.",
        ]

        # Task commitment acknowledgment (task phase)
        if self.player.memory.task_commitment >= 0.8 and not is_discussion:
            ack_parts.append("Task commitment: HIGH — I must finish my current task.")
        
        # Crisis role acknowledgment
        if self.player.memory.crisis_role == "CRISIS_RESPONDER":
            ack_parts.append("Crisis role: RESPONDER — I must fix the sabotage.")
        elif self.player.memory.crisis_role == "IGNORE_ALARM":
            ack_parts.append("Crisis role: IGNORE — others are closer, I continue tasks.")

        # Meeting-specific acknowledgment
        if is_discussion:
            if self.player.identity == "Crewmate" and alibi_block:
                alibi_entries = self.player.get_alibi_ledger()
                if alibi_entries:
                    last = alibi_entries[-1]
                    ack_parts.append(f"My last known position: T{last['turn']} at {last['location']}.")
            elif self.player.identity == "Impostor" and phantom_block:
                fake_alibi = self.player.get_fake_alibi_for_meeting()
                if isinstance(fake_alibi, list) and fake_alibi:
                    ack_parts.append(f"My cover story: {fake_alibi[-1]}.")

        fake_ack = " ".join(ack_parts)

        # --- Message 4: Action Request (user) ---
        # For discussion phase, the CoT format instruction is already inside
        # phase_context (at the very bottom, for recency bias).  We override
        # the generic "Return your output…" with a reminder to use THOUGHT/SPEAK.
        if is_discussion:
            format_reminder = (
                "\n\nRemember: respond in the THOUGHT / SPEAK format. "
                "THOUGHT first (private reasoning referencing your VERIFIED HISTORY), "
                "then SPEAK (public dialogue)."
            )
        else:
            format_reminder = (
                "\n\nReturn your output following the exact format "
                "specified in the system instructions."
            )

        action_request = (
            f"Good. Now here is the full game context.\n\n"
            f"{all_info}\n\n"
            f"[World State Ledger]\n{self.processed_memory}\n\n"
            f"Phase: {phase}.{phase_context}"
            f"{commitment_block}"
            f"{crisis_block}"
            f"{danger_block}"
            f"{alibi_block}"
            f"{contradiction_block}"
            f"{phantom_block}"
            f"{third_impostor_block}"
            f"{format_reminder}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state_injection},
            {"role": "assistant", "content": fake_ack},
            {"role": "user", "content": action_request},
        ]
        return messages, action_names, action_reprs

    async def choose_action(self, timestep: int):
        available_actions = self.player.get_available_actions()
        
        # 0. Compose the prompt messages
        messages, action_names, action_reprs = self._compose_action_prompt(timestep, available_actions)
        
        # Construct full prompt for logging
        all_info = self.player.all_info_prompt()
        phase = "Meeting phase - DISCUSSION" if "SPEAK" in action_names and "MOVE" not in action_names \
                else "Meeting phase - VOTING" if "VOTE" in action_names \
                else "Task phase"
        full_prompt = {
            "Summarization": self.summarization,
            "All Info": all_info,
            "Memory": self.processed_memory,
            "Phase": phase,
            "Current Location": self.player.location,
            "Previous Location": self.previous_location,
        }

        # 1. Get LLM response (adjust temperature per phase)
        # Discussion keeps default temp (0.7) to encourage varied speech and prevent
        # "mode collapse" where all agents copy each other's phrasing (the "LIAR ALERT" parrot effect).
        # Voting gets slightly lower temp for more deterministic logic.
        original_temp = self.temperature
        if "VOTE" in action_names:
            self.temperature = 0.5  # Lower temp for voting: improve logical consistency
        
        try:
            response = await self.send_request(messages)
        except Exception as e:
            print(f"[ERROR] Error in choose_action for {self.model}: {e}")
            import traceback
            traceback.print_exc()
            fallback = available_actions[0] if available_actions else None
            self.log_interaction(
                sysprompt=self.system_prompt,
                prompt=full_prompt,
                original_response=f"[ERROR]\n{e}\n\n[Resolved Action]\n{repr(fallback) if fallback else 'None'}",
                step=timestep,
            )
            return fallback
        finally:
            self.temperature = original_temp  # Always restore original temperature
        
        # ───────────────────────────────────────────────────────────────────
        # CHAIN-OF-THOUGHT EXTRACTION (Discussion Phase)
        # The new prompt forces a  THOUGHT: … / SPEAK: "…"  format.
        # We log the THOUGHT for debug, then normalize the response so the
        # downstream parser (which expects [Action] SPEAK: "…") can handle it.
        # ───────────────────────────────────────────────────────────────────
        is_discussion_phase = "SPEAK" in action_names and "MOVE" not in action_names
        if is_discussion_phase and response:
            thought = self._extract_thought_from_response(response)
            if thought:
                print(f"[COT] {self.player.name} THOUGHT: {thought[:200]}{'…' if len(thought) > 200 else ''}")

            # Normalize CoT output → legacy [Action] format
            # If the LLM followed the CoT format (SPEAK: "…" without [Action]),
            # wrap it so the existing regex parsers downstream can match it.
            import re as _cot_re
            if _cot_re.search(r'(?:^|\n)\s*SPEAK\s*:', response, _cot_re.IGNORECASE):
                if '[Action]' not in response:
                    _speak_pat = _cot_re.compile(
                        r'(^|\n)(\s*)(SPEAK)\s*:', _cot_re.IGNORECASE)
                    response = _speak_pat.sub(
                        r'\1\2[Action] SPEAK:', response, count=1)

        # ───────────────────────────────────────────────────────────────────
        # SPEAKING SCORE HEURISTIC: Generate → Score → Select/Regenerate
        # Applied only during discussion phase (SPEAK available, MOVE not).
        # Validates the LLM's speech against ground-truth before allowing it.
        # ───────────────────────────────────────────────────────────────────
        if is_discussion_phase and self.player.is_alive and response:
            valid_truths = self._compute_valid_truths()
            speech = self._extract_speech_from_response(response)
            
            if speech:
                best_response = response
                best_score, breakdown = self._score_speech(speech, valid_truths)
                score_summary = [(b[0], b[1]) for b in breakdown]
                print(f"[SPEECH SCORE] {self.player.name}: {best_score} | {score_summary}")
                
                if best_score < 0:
                    # Speech contains hallucination — regenerate up to 2 more times
                    # (total: 3 distinct thoughts, as per the heuristic algorithm)
                    rejection_reasons = [b[0] for b in breakdown if b[1] < 0]
                    
                    for attempt in range(2):
                        # Build corrective prompt based on what went wrong
                        corrections = []
                        if "X-RAY VISION" in rejection_reasons:
                            corrections.append(
                                "You can ONLY discuss rooms you PERSONALLY visited "
                                "and players you PERSONALLY saw. If you were not in a room, "
                                "you cannot claim to know who was or wasn't there."
                            )
                        if "META-GAMING" in rejection_reasons:
                            corrections.append(
                                "Speak naturally like a real person. Do NOT reference "
                                "'logs', 'timesteps', 'T0', 'T1', or any game mechanics."
                            )
                        if "SELF-INCRIMINATION" in rejection_reasons:
                            corrections.append(
                                "Protect your cover. Talk about your tasks and what you "
                                "observed. Do NOT reveal anything about your actual actions."
                            )
                        if "SPATIAL NON-SEQUITUR" in rejection_reasons:
                            corrections.append(
                                "Being in one room tells you NOTHING about a different room. "
                                "Only make claims about rooms you were physically in."
                            )
                        
                        correction_text = " ".join(corrections) if corrections else (
                            "Stick to what you actually saw in your current room."
                        )
                        
                        # Create fresh copy of messages with correction appended
                        regen_messages = [dict(m) for m in messages]
                        regen_messages[-1] = dict(regen_messages[-1])
                        regen_messages[-1]["content"] += (
                            f"\n\n⚠️ SPEECH CORRECTION (attempt {attempt + 2}/3): "
                            f"Your previous speech was rejected. {correction_text} "
                            f"Generate a new speech based ONLY on your firsthand experience."
                        )
                        
                        retry_response = await self.send_request(regen_messages)
                        if retry_response:
                            retry_speech = self._extract_speech_from_response(retry_response)
                            if retry_speech:
                                retry_score, retry_breakdown = self._score_speech(
                                    retry_speech, valid_truths
                                )
                                retry_summary = [(b[0], b[1]) for b in retry_breakdown]
                                print(f"[SPEECH SCORE] {self.player.name} (retry {attempt + 1}): "
                                      f"{retry_score} | {retry_summary}")
                                
                                if retry_score > best_score:
                                    best_score = retry_score
                                    best_response = retry_response
                                
                                if retry_score >= 0:
                                    break  # Good enough — use this one
                    
                    if best_score < 0:
                        # All 3 attempts scored negative — force safe fallback
                        print(f"[SPEECH FALLBACK] {self.player.name}: All attempts scored "
                              f"negative ({best_score}). Forcing safe speech.")
                        if valid_truths["is_impostor"]:
                            best_response = (
                                '[Action] SPEAK: "I was doing my tasks. '
                                'I didn\'t see anything unusual. Has anyone else found anything?"'
                            )
                        else:
                            best_response = (
                                '[Action] SPEAK: "I was doing my tasks. '
                                'I don\'t have any direct evidence to share right now."'
                            )
                    
                    response = best_response

        output_action = self._normalize_response(response)

        # Extract and update memory
        new_memory = self._extract_memory(response)
        if new_memory != self.processed_memory:
            self.processed_memory = new_memory

        # NOTE: We log the raw response here, but will append the resolved action below
        # after parsing completes, so evaluations show what actually executed.
        raw_response_for_log = response if response else ""

        # Update previous location before action is executed
        self.previous_location = self.player.location

        # Debug: log what actions are available
        print(f"[DEBUG] {self.player.name} available: {action_names}, phase: {phase}")

        # Helper: log interaction with resolved action appended
        def _log_with_resolved(resolved_action_str):
            """Log the raw LLM response with the engine-resolved action appended.
            
            The [Resolved Action] tag goes on its own line so the section parser
            (which checks line.startswith('[') and line.endswith(']')) can extract it.
            """
            annotated_response = raw_response_for_log
            if resolved_action_str:
                annotated_response += f"\n\n[Resolved Action]\n{resolved_action_str}"
            self.log_interaction(
                sysprompt=self.system_prompt,
                prompt=full_prompt,
                original_response=annotated_response,
                step=timestep,
            )

        # Attempt to find the action in available_actions
        selected_action = None
        
        # 1. Try exact repr match
        for action in available_actions:
            if repr(action) == output_action:
                selected_action = action
                print(f"[DEBUG] Exact match found: {action}")
                break
        
        # Guard against empty/None response (e.g. LLM failure)
        if not output_action:
             print(f"[WARNING] No response from LLM for {self.model}. Falling back.")
             output_action = ""

        # 2. Strict regex match
        if selected_action is None:
            import re
            regex = r"(\[Thinking Process\]:?[\s\S]*?)?(\[Action\]:?)\s*([A-Z]+(?:\s+[A-Z]+)*)\b(?:[:\s]+\"?([^\"]*)\"?)?"
            match = re.search(regex, output_action, re.DOTALL | re.MULTILINE)
            
            if match:
                action_type = match.group(3).strip()
                action_message = match.group(4).strip() if match.group(4) else ""
                print(f"[DEBUG] Regex match: type='{action_type}', location/msg='{action_message}'")

                # Try to match action_type to available actions
                for action in available_actions:
                    # Alias matching for CALL MEETING / REPORT DEAD BODY
                    match_name = action.name.upper()
                    target_name = action_type.upper()
                    
                    names_match = (match_name == target_name) or \
                                  (match_name == "CALL MEETING" and target_name == "REPORT DEAD BODY") or \
                                  (match_name == "REPORT DEAD BODY" and target_name == "CALL MEETING")

                    if names_match:
                        # For SPEAK actions, attach the message
                        if action.name == "SPEAK":
                            action.provide_message(action_message if action_message else "...")
                            selected_action = action
                            break
                        # For VOTE/KILL, ensure the target matches if specified
                        elif action.name in ["VOTE", "KILL"]:
                            if hasattr(action, 'other_player') and action.other_player.name.upper() in action_message.upper():
                                selected_action = action
                                break
                            elif not action_message: # If no message, assume it's a generic vote/kill if only one option
                                selected_action = action
                                break
                        else:
                            selected_action = action
                            # For MOVE/VENT, check destination
                            if action.name in ["MOVE", "VENT"] and hasattr(action, "new_location"):
                                if action.new_location.upper() in action_message.upper():
                                    break # Perfect match
                                else:
                                    selected_action = None # Keep looking
                            else:
                                break

        # 3. Flexible CALL MEETING / REPORT DEAD BODY matching (if regex or loop didn't find specific target)
        if selected_action is None and ("CALL MEETING" in output_action or "REPORT DEAD BODY" in output_action):
             for action in available_actions:
                  if action.name == "CALL MEETING":
                       selected_action = action
                       break

        # 4. Partial match (Fuzzy)
        if selected_action is None:
            for action in available_actions:
                if repr(action) in output_action:
                    selected_action = action
                    break

        # 5. Keyword fallback for SPEAK (if they just output a quoted message)
        if selected_action is None and "SPEAK" in action_names:
            if '"' in output_action:
                for action in available_actions:
                    if action.name == "SPEAK":
                        msg_match = re.search(r'"([^"]*)"', output_action)
                        if msg_match:
                            action.provide_message(msg_match.group(1))
                            selected_action = action
                            break
            # Check if output starts with SPEAK or contains SPEAK despite failed regex
            elif output_action.startswith("SPEAK") or "SPEAK:" in output_action:
                for action in available_actions:
                    if action.name == "SPEAK":
                        selected_action = action
                        break

        # 6. Special handling for "KILL" (fuzzy match target if regex failed)
        if selected_action is None and "KILL" in output_action:
             for action in available_actions:
                 if action.name == "KILL" and action.other_player.name in output_action:
                     selected_action = action
                     break

        # 7. Flexible VOTE matching (case-insensitive, handles partial color names)
        if selected_action is None and "VOTE" in action_names:
            output_upper_vote = output_action.upper() if output_action else ""
            
            # Check for SKIP first
            if "SKIP" in output_upper_vote:
                # Record as SKIP — this will be handled in agent_step
                print(f"[VOTE] {self.player.name} voted to SKIP.")
                _log_with_resolved("VOTE SKIP")
                return None  # Returning None triggers the SKIP handler in agent_step
            
            # Try to match any valid VOTE target
            if "VOTE" in output_upper_vote or any(a.name == "VOTE" for a in available_actions):
                best_vote_match = None
                for action in available_actions:
                    if action.name == "VOTE":
                        target_name = action.other_player.name.upper()
                        # Full name match (e.g., "Player 2: pink")
                        if target_name in output_upper_vote:
                            best_vote_match = action
                            break
                        # Color-only match (e.g., "pink", "orange")
                        target_color = action.other_player.color.upper() if hasattr(action.other_player, 'color') else ""
                        if target_color and target_color in output_upper_vote:
                            best_vote_match = action
                            break
                        # Player number match (e.g., "Player 2")
                        # Extract player number from name like "Player 2: pink"
                        import re as _re
                        num_match = _re.search(r'Player\s*(\d+)', action.other_player.name, _re.IGNORECASE)
                        if num_match:
                            player_num = num_match.group(0).upper()
                            if player_num in output_upper_vote:
                                best_vote_match = action
                                break
                
                if best_vote_match:
                    selected_action = best_vote_match
                    print(f"[VOTE] {self.player.name} → {best_vote_match.other_player.name} (flexible match)")
                    # (logging is handled in the `if selected_action:` block below)

        if selected_action:
            # If it's a SPEAK action, ensure message is attached (if we parsed it earlier)
            if selected_action.name == "SPEAK":
                # Re-extract message for safety if not already set or default
                if not hasattr(selected_action, 'message') or selected_action.message == "...":
                        # Extract message from various formats: "SPEAK: ...", "SPEAK to ...", "SPEAK ..."
                        if "SPEAK: " in output_action:
                            message = output_action.split("SPEAK: ", 1)[1]
                        elif "SPEAK:" in output_action:
                            message = output_action.split("SPEAK:", 1)[1].strip()
                        else:
                            # Just take everything after "SPEAK " if it starts with SPEAK
                            if output_action.startswith("SPEAK "):
                                message = output_action[6:].strip()
                            else:
                                message = "..." # Default fallthrough

                        # Clean up quotes if present
                        if message and message.startswith('"') and message.endswith('"'):
                            message = message[1:-1]
                        elif message and message.startswith("'") and message.endswith("'"):
                            message = message[1:-1]
                        
                        selected_action.message = message if message else "..."
            
            # --- Impostor Deception Ledger: auto-generate alibi after KILL ---
            if selected_action.name == "KILL" and self.player.identity == "Impostor":
                self.kill_location = self.player.location
                self.kill_victim = getattr(selected_action, 'other_player', None)
                self.kill_victim = self.kill_victim.name if self.kill_victim else "unknown"
                # Pick an alibi room: a vent-connected room (best escape story) or a random adjacent room
                import random as _rng
                from amongagents.agent.neutral_prompts import CONNECTION_INFO
                # Parse adjacent rooms from CONNECTION_INFO for current location
                alibi_candidates = []
                for line in CONNECTION_INFO.split('\n'):
                    if line.startswith(self.player.location + ' →'):
                        rooms_part = line.split('→')[1].strip()
                        alibi_candidates = [r.strip() for r in rooms_part.split(',')]
                        break
                # Remove the actual kill location and Cafeteria (too risky to claim)
                alibi_candidates = [r for r in alibi_candidates if r != self.player.location and r != "Cafeteria"]
                if alibi_candidates:
                    self.public_alibi = _rng.choice(alibi_candidates)
                else:
                    self.public_alibi = "Medbay"  # Safe fallback
                print(f"[DECEPTION] {self.player.name} killed {self.kill_victim} at {self.kill_location}. PUBLIC ALIBI: {self.public_alibi}")
            
            # --- Impostor Speech Sanitizer: prevent self-incrimination ---
            if selected_action.name == "SPEAK" and self.player.identity == "Impostor" and self.kill_location:
                msg = getattr(selected_action, 'message', '')
                if msg:
                    msg_lower = msg.lower()
                    # Check for accidental confession: mentioning "I killed" or revealing true kill location
                    self_incriminating = False
                    if "i killed" in msg_lower or "i did kill" in msg_lower or "i murdered" in msg_lower:
                        self_incriminating = True
                    # Check if they're revealing the actual kill location as their own location
                    if self.kill_location.lower() in msg_lower and f"i was in {self.kill_location.lower()}" in msg_lower:
                        self_incriminating = True
                    
                    if self_incriminating:
                        print(f"[DECEPTION FILTER] {self.player.name} tried to self-incriminate! Rewriting speech.")
                        # Replace with alibi-safe speech
                        selected_action.message = f"I was in {self.public_alibi} doing my tasks. I didn't see anything suspicious. Has anyone else seen anything?"
            
            # ─── THOUGHT-ACTION ALIGNMENT VALIDATOR ───
            # If the LLM's thinking explicitly says "stay" / "remain" / "must stay"
            # but the resolved action is MOVE, this is a thought-action misalignment.
            # Override: pick the best non-MOVE action (prefer COMPLETE TASK).
            if selected_action.name == "MOVE" and self.player.is_alive:
                import re as _re_align
                # Extract the thinking portion (everything before [Action])
                thinking_text = output_action.split("[Action]")[0] if "[Action]" in output_action else output_action
                stay_patterns = [
                    r'\bmust\s+stay\b', r'\bshould\s+stay\b', r'\bneed\s+to\s+stay\b',
                    r'\bstay\s+(?:in|here|and|to)\b', r'\bremain\s+(?:in|here)\b',
                    r'\bfinish\s+(?:my|the|this)\s+task\b', r'\bcomplete\s+(?:my|the|this)\s+task\b',
                    r'\bdon\'?t\s+move\b', r'\bshould\s+not\s+move\b', r'\bshouldn\'?t\s+move\b',
                ]
                thought_says_stay = any(_re_align.search(p, thinking_text, _re_align.IGNORECASE) for p in stay_patterns)
                
                if thought_says_stay:
                    # Find best replacement: prefer COMPLETE TASK, then any non-MOVE action
                    replacement = None
                    for a in available_actions:
                        if a.name == "COMPLETE TASK":
                            replacement = a
                            break
                    if replacement is None:
                        for a in available_actions:
                            if a.name != "MOVE":
                                replacement = a
                                break
                    if replacement:
                        print(f"[ALIGNMENT] {self.player.name}: Thought says 'stay' but action was MOVE → overriding to {replacement.name}")
                        selected_action = replacement

            _log_with_resolved(repr(selected_action))
            return selected_action
        
        # --- Smart Fallback: try to salvage the intended action ---
        # The LLM output didn't match any action via exact/regex/partial.
        # Before defaulting to available_actions[0], try to extract the intended
        # destination or target from the output and find the closest valid action.
        
        output_upper = output_action.upper() if output_action else ""
        
        # Try to find a MOVE/VENT with a matching destination
        if "MOVE" in output_upper or "VENT" in output_upper:
            for action in available_actions:
                if action.name in ("MOVE", "VENT") and hasattr(action, "new_location"):
                    if action.new_location.upper() in output_upper:
                        print(f"[SMART FALLBACK] {self.player.name}: LLM wanted '{output_action[:80]}...' → matched destination '{action.new_location}' via {action.name}")
                        _log_with_resolved(repr(action))
                        return action
        
        # Try to find a KILL with ANY valid target (LLM wanted to kill but named wrong target)
        # This handles the "Ghost Player" case: LLM says "KILL Player 5: red" but Player 5
        # doesn't exist. If there IS a valid kill target, pick the first one.
        if "KILL" in output_upper:
            kill_actions = [a for a in available_actions if a.name == "KILL"]
            if kill_actions:
                # Only auto-select if there's exactly 1 target (unambiguous intent)
                if len(kill_actions) == 1:
                    print(f"[SMART FALLBACK] {self.player.name}: LLM hallucinated kill target → redirected to only valid target '{kill_actions[0].other_player.name}'")
                    _log_with_resolved(repr(kill_actions[0]))
                    return kill_actions[0]
                else:
                    # Multiple targets: can't guess which one the LLM meant, don't kill
                    print(f"[WARNING] {self.player.name}: LLM wanted KILL but named invalid target. {len(kill_actions)} valid targets available — ambiguous, falling through.")
        
        # Try to find a COMPLETE TASK matching a task name
        if "COMPLETE" in output_upper or "TASK" in output_upper:
            for action in available_actions:
                if "COMPLETE" in action.name:
                    print(f"[SMART FALLBACK] {self.player.name}: LLM wanted task → matched '{repr(action)}'")
                    _log_with_resolved(repr(action))
                    return action
        
        # VOTE-specific smart fallback: if in voting phase and all parsing failed,
        # try aggressive fuzzy matching or default to SKIP instead of a random action.
        vote_actions = [a for a in available_actions if a.name == "VOTE"]
        if vote_actions:
            # Check for SKIP intent
            if "SKIP" in output_upper or "ABSTAIN" in output_upper or "NO ONE" in output_upper or "NO VOTE" in output_upper:
                print(f"[SMART FALLBACK] {self.player.name}: LLM intended to skip vote.")
                _log_with_resolved("VOTE SKIP")
                return None  # Triggers SKIP handler in agent_step
            
            # Try aggressive single-word matching against vote target colors/names
            output_words = output_upper.split()
            for action in vote_actions:
                color = action.other_player.color.upper() if hasattr(action.other_player, 'color') else ""
                if color and color in output_words:
                    print(f"[SMART FALLBACK] {self.player.name}: matched vote target by color '{color}' → {action.other_player.name}")
                    _log_with_resolved(repr(action))
                    return action
            
            # If only 1 vote target exists, just pick it (unambiguous)
            if len(vote_actions) == 1:
                print(f"[SMART FALLBACK] {self.player.name}: only 1 vote target available → {vote_actions[0].other_player.name}")
                _log_with_resolved(repr(vote_actions[0]))
                return vote_actions[0]
            
            # All else failed in voting: default to SKIP rather than a random vote
            print(f"[WARNING] {self.player.name}: could not parse vote target from '{output_action[:80]}'. Recording as SKIP.")
            _log_with_resolved("VOTE SKIP")
            return None  # Triggers SKIP handler in agent_step
        
        # Final default: fall back to first available action
        if available_actions:
            print(f"[WARNING] {self.player.name}: Invalid action, no smart match found. Falling back to {available_actions[0]}")
            _log_with_resolved(repr(available_actions[0]))
            return available_actions[0]
        
        # Empty available_actions: this should be extremely rare.
        # Return a dummy action that agent_step's NO-SKIP enforcer will handle.
        print(f"[WARNING] {self.player.name}: No available actions at all.")
        _log_with_resolved("NO ACTIONS AVAILABLE")
        return None

    def choose_observation_location(self, map):
        if isinstance(map, (list, tuple)):
            return random.choice(map)
        else:
            # For sets, dicts, or other non-sequence types
            return random.choice(list(map))


class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def choose_action(self):
        available_actions = self.player.get_available_actions()
        action = np.random.choice(available_actions)
        if action.name == "speak":
            message = "Hello, I am a crewmate."
            action.provide_message(message)
        return action

    def choose_observation_location(self, map):
        return random.sample(map, 1)[0]


class HumanAgent(Agent):
    def __init__(self, player, tools=None, game_index=0, agent_config=None, list_of_impostors=None):
        super().__init__(player)
        self.model = "homosapiens/brain-1.0"
        self.tools = tools
        self.game_index = game_index
        self.summarization = "No thought process has been made."
        self.processed_memory = "No memory has been processed."
        self.log_path = os.getenv("EXPERIMENT_PATH") + "/agent-logs.json"
        self.compact_log_path = os.getenv("EXPERIMENT_PATH") + "/agent-logs-compact.json"
        self.current_available_actions = []
        self.current_step = 0
        self.max_steps = 50  # Default value, will be updated from game config
        self.action_future = None  # Store the future as an instance variable
        self.condensed_memory = ""  # Store the condensed memory (scratchpad) between turns
    
    def update_max_steps(self, max_steps):
        """Update the max_steps value from the game config."""
        self.max_steps = max_steps

    async def choose_action(self, timestep: int):
        """
        Chooses an action, either via web interface (if FLASK_ENABLED=True)
        or command line (if FLASK_ENABLED=False).
        """
        use_flask = os.getenv("FLASK_ENABLED", "True") == "True"
        all_info = self.player.all_info_prompt()
        self.current_available_actions = self.player.get_available_actions()
        self.current_step = timestep

        if use_flask:
            # --- Web Interface Logic ---            
            action_prompt = "Waiting for human action via web interface.\nAvailable actions:\n" + "\n".join([f"{i+1}: {str(action)}" for i, action in enumerate(self.current_available_actions)])
            full_prompt = {
                "All Info": all_info,
                "Available Actions": action_prompt,
                "Current Step": f"{timestep}/{self.max_steps}",
                "Current Player": self.player.name
            }

            loop = asyncio.get_event_loop()
            self.action_future = loop.create_future()  # Store in instance variable
            
            # Use game_id from the server instead of game_index
            # The game_id is passed to the HumanAgent when it's created
            game_id = getattr(self, 'game_id', self.game_index)
            human_action_futures[game_id] = self.action_future
            
            print(f"[Agent] Created future for game {game_id}")
            print(f"[Agent] Available futures: {list(human_action_futures.keys())}")

            print(f"\n[Game {game_id}] Human player {self.player.name}'s turn. Waiting for action via web interface...")
            print(f"Available actions: {[str(a) for a in self.current_available_actions]}")

            try:
                chosen_action_data = await self.action_future
                action_idx = chosen_action_data.get("action_index")
                action_message = chosen_action_data.get("message")
                condensed_memory = chosen_action_data.get("condensed_memory", "")
                thinking_process = chosen_action_data.get("thinking_process", "")

                # Update the condensed memory if provided
                if condensed_memory:
                    self.condensed_memory = condensed_memory

                if action_idx is None or action_idx < 0 or action_idx >= len(self.current_available_actions):
                    print(f"[Game {game_id}] Invalid action index received: {action_idx}. Defaulting to first action.")
                    selected_action = self.current_available_actions[0]
                else:
                    selected_action = self.current_available_actions[action_idx]

                # Format the response log to match LLMAgent format
                response_log = ""
                if self.condensed_memory:
                    response_log += f"[Condensed Memory]\n{self.condensed_memory}\n\n"
                if thinking_process:
                    response_log += f"[Thinking Process]\n{thinking_process}\n\n"
                
                response_log += f"[Action] {str(selected_action)}"
                
                # Check if action requires a message (e.g., SPEAK)
                # Use str() and check for attributes robustly
                is_speak_action = False
                if hasattr(selected_action, 'name'): # Check attribute exists
                    is_speak_action = selected_action.name == "SPEAK"
                elif "SPEAK" in str(selected_action): # Fallback to string check
                    is_speak_action = True
                
                if is_speak_action and action_message:
                    if hasattr(selected_action, 'provide_message'):
                        selected_action.provide_message(action_message)
                    elif hasattr(selected_action, 'message'): # Fallback to setting attribute
                        selected_action.message = action_message
                    response_log += f" {action_message}"

                # Update the prompt to not include "Waiting for human action via web interface"
                full_prompt = {
                    "All Info": all_info,
                    "Available Actions": "\n".join([f"{i+1}: {str(action)}" for i, action in enumerate(self.current_available_actions)]),
                    "Current Step": f"{timestep}/{self.max_steps}",
                    "Current Player": self.player.name
                }

                self.log_interaction(sysprompt="Human Agent (Web)", prompt=full_prompt,
                                     original_response=response_log,
                                     step=timestep)
                
                # Clear the future and actions only after successful action selection
                if game_id in human_action_futures:
                    print(f"[Agent] Deleting future for game {game_id} after successful action")
                    del human_action_futures[game_id]
                self.current_available_actions = []
                self.action_future = None
                
                return selected_action

            except asyncio.CancelledError:
                print(f"[Game {game_id}] Human action cancelled.")
                # Clean up on cancellation
                if game_id in human_action_futures:
                    print(f"[Agent] Deleting future for game {game_id} after cancellation")
                    del human_action_futures[game_id]
                self.current_available_actions = []
                self.action_future = None
                raise
        else:
            # --- Command Line Interface Logic ---            
            action_prompt = "Available actions:\n" + "\n".join([f"{i+1}: {str(action)}" for i, action in enumerate(self.current_available_actions)])
            full_prompt = {
                "All Info": all_info,
                "Available Actions": action_prompt
            }
            
            print(f"\n--- [Game {self.game_index}] Player: {self.player.name} ({self.player.identity if self.player.identity else 'Role Unknown'}) ---")
            print(all_info)
            print("\nChoose an action:")
            for i, action in enumerate(self.current_available_actions):
                print(f"{i+1}: {str(action)}")
            print("(Enter 0 to stop game)")
                
            stop_triggered = False
            valid_input = False
            selected_action = None
            action_idx_chosen = -1

            while (not stop_triggered) and (not valid_input):
                try:
                    user_input = input("> ")
                    action_idx_chosen = int(user_input)
                    if action_idx_chosen == 0:
                        stop_triggered = True
                    elif action_idx_chosen < 1 or action_idx_chosen > len(self.current_available_actions):
                        print(f"Invalid input. Please enter a number between 1 and {len(self.current_available_actions)} (or 0 to stop).")
                    else:
                        valid_input = True
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue
                    
            if stop_triggered:
                print("Stopping game as requested by user.")
                # How to signal stop? Raise exception? Return specific value?
                # For now, raise an exception that the game loop might catch.
                raise KeyboardInterrupt("Game stopped by user via CLI.")
                
            selected_action = self.current_available_actions[action_idx_chosen - 1]
            response_log = f"[Action] {str(selected_action)}"
            
            # Check if action requires a message using string check
            is_speak_action = False
            if hasattr(selected_action, 'name'):
                 is_speak_action = selected_action.name == "SPEAK"
            elif "SPEAK" in str(selected_action):
                 is_speak_action = True

            if is_speak_action:
                print("Enter your message:")
                action_message = input("> ")
                if hasattr(selected_action, 'provide_message'):
                     selected_action.provide_message(action_message)
                elif hasattr(selected_action, 'message'):
                     selected_action.message = action_message
                else:
                     print("Warning: Could not set message for SPEAK action.")
                response_log += f" {action_message}"
            
            self.log_interaction(sysprompt="Human Agent (CLI)", prompt=full_prompt, 
                                 original_response=response_log, 
                                 step=timestep)
        
            self.current_available_actions = [] # Clear actions after use
            return selected_action # Return synchronously within async def

    def get_current_state_for_web(self) -> Dict[str, Any]:
        """
        Returns the necessary state for the web UI when it's the human's turn.
        Uses string checks for action properties.
        """
        available_actions_web = []
        for action in self.current_available_actions:
            action_str = str(action)
            requires_message = False
            if hasattr(action, 'name'):
                 requires_message = action.name == "SPEAK"
            elif "SPEAK" in action_str:
                 requires_message = True
                 
            available_actions_web.append({
                "name": action_str,
                "requires_message": requires_message
            })
            
        return {
            "is_human_turn": True,
            "player_name": self.player.name,
            "player_info": self.player.all_info_prompt(),
            "available_actions": available_actions_web,
            "current_step": f"{self.current_step}/{self.max_steps}",
            "current_player": self.player.name,
            "condensed_memory": self.condensed_memory  # Include the condensed memory in the state
        }

    def respond(self, message):
        print(message)
        response = input()
        return response

    def choose_observation_location(self, map):
        map_list = list(map)
        print("Please select the room you wish to observe:")
        for i, room in enumerate(map_list):
            print(f"{i}: {room}")
        while True:
            try:
                index = int(input())
                if index < 0 or index >= len(map_list):
                    print(f"Invalid input. Please enter a number between 0 and {len(map_list) - 1}.")
                else:
                    return map_list[index]
            except:
                print("Invalid input. Please enter a number.")

    def log_interaction(self, sysprompt, prompt, original_response, step):
        """
        Helper method to store model interactions in properly nested JSON format.
        Handles deep nesting and properly parses all string-formatted dictionaries.
        Correctly separates Memory, Thinking, and Action sections.
        """
        sections = {}

        # Clean the original response slightly for easier parsing
        response_text = original_response.strip()

        # Use regex to find sections robustly, ignoring case for tags
        action_match = re.search(r"\[Action\](.*)", response_text, re.DOTALL | re.IGNORECASE)
        memory_match = re.search(r"\[Condensed Memory\](.*?)(\[(Thinking Process|Action)\]|$)", response_text, re.DOTALL | re.IGNORECASE)
        thinking_match = re.search(r"\[Thinking Process\](.*?)(\[(Condensed Memory|Action)\]|$)", response_text, re.DOTALL | re.IGNORECASE)

        # Initialize keys to ensure they exist, defaulting to empty string
        sections["Condensed Memory"] = ""
        sections["Thinking Process"] = ""

        # Extract content based on matches, overwriting defaults if found
        if memory_match:
            sections["Condensed Memory"] = memory_match.group(1).strip()

        if thinking_match:
            sections["Thinking Process"] = thinking_match.group(1).strip()

        if action_match:
            action_text = action_match.group(1).strip()
            # Remove leading number format like "1. "
            action_text_cleaned = re.sub(r"^\d+\.\s*", "", action_text).strip()

            # Assign the full cleaned action string directly, regardless of message presence
            if action_text_cleaned:
                sections["Action"] = action_text_cleaned
            # If action_text_cleaned is empty after stripping number, don't add Action section

        # Handle cases where tags might be missing or text exists outside tags
        # (This logic might need refinement depending on expected variations)
        # For now, prioritize explicitly tagged sections.

        # Create the interaction object with proper nesting
        interaction = {
            'game_index': 'Game ' + str(self.game_index),
            'step': step,
            "timestamp": str(datetime.now()),
            "player": {"name": self.player.name, "identity": self.player.identity, "personality": self.player.personality, "model": self.model, "location": self.player.location},
            "interaction": {"system_prompt": sysprompt, "prompt": prompt, "response": sections, "full_response": original_response},
        }

        # Ensure log directories exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.compact_log_path), exist_ok=True)

        # Write to file with minimal whitespace but still readable
        try:
            with open(self.log_path, "a") as f:
                json.dump(interaction, f, indent=2, separators=(",", ": "))
                f.write("\n")
                f.flush()
            with open(self.compact_log_path, "a") as f:
                json.dump(interaction, f, separators=(",", ":"))
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}") # Add error logging

        print(".", end="", flush=True)

class LLMHumanAgent(HumanAgent, LLMAgent):
    def __init__(self, player, tools=None, game_index=0, agent_config=None, list_of_impostors=None):
        super().__init__(player, tools, game_index, agent_config, list_of_impostors)

    async def choose_action(self, timestep):
        return await HumanAgent.choose_action(self, timestep)

    def respond(self, message):
        return HumanAgent.respond(self, message)
        
    def log_interaction(self, sysprompt, prompt, original_response, step):
        return HumanAgent.log_interaction(self, sysprompt, prompt, original_response, step)