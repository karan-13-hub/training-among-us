from amongagents.envs.action import (
    COMMON_ACTIONS,
    CREWMATE_ACTIONS,
    IMPOSTER_ACTIONS,
    CompleteTask,
)

PLAYER_COLORS = [
    "red",
    "blue",
    "green",
    "pink",
    "orange",
    "yellow",
    "black",
    "white",
    "purple",
    "brown",
    "cyan",
    "lime",
]


class MemoryState:
    """Persistent structured memory for each agent.

    Separates *verified observations* (things the player physically witnessed)
    from *social claims* (things other players said).  Carries over between
    timesteps so the agent never loses track of its own history.
    """

    def __init__(self, player_name: str, role: str):
        # Identity
        self.identity = (role, player_name)

        # Location trace: [(room, timestep), …]
        self.location_history: list = []

        # Ground-truth visual observations the player SAW themselves.
        # Each entry: {tick, event, type: "VISUAL", location}
        self.verified_observations: list = []

        # Claims made by OTHER players (hearsay — may be lies).
        # Each entry: {tick, actor, claim, type: "HEARSAY"}
        self.social_log: list = []

        # Active goal: "TASK_EXECUTION" | "CRISIS_RESPONSE" | "INVESTIGATION" | "DEAD"
        self.current_intent: str = "TASK_EXECUTION"

        # 0.0 → free to move.  ≥ 0.8 → MUST finish the current task first.
        self.task_commitment: float = 0.0

        # Crisis dispatch role set by the engine each sabotage tick.
        # "CRISIS_RESPONDER" | "IGNORE_ALARM" | None
        self.crisis_role: str = None

        # ═══ CONSISTENCY CHECK: Track the agent's own claims ═══
        # Prevents self-contradiction across meeting rounds.
        # last_statement: the most recent public claim this agent made.
        # own_claims: rolling list of all public claims for self-audit.
        self.last_statement: str = ""
        self.own_claims: list = []  # [{tick, claim}]

    # ──────────────────────────────────────────────
    # Public API consumed by the game engine
    # ──────────────────────────────────────────────
    def update_location(self, room: str, timestep: int,
                        action_taken: str = ""):
        """Append a new location entry with the action the player took.

        This creates the VERIFIED HISTORY — an immutable, engine-recorded
        log that the LLM cannot contradict during meetings.
        """
        self.location_history.append((room, timestep, action_taken))

    def add_verified(self, timestep: int, event: str, location: str,
                     event_type: str = "VISUAL"):
        """Record something the player physically witnessed."""
        self.verified_observations.append({
            "tick": timestep,
            "event": event,
            "type": event_type,
            "location": location,
        })

    def add_hearsay(self, timestep: int, actor: str, claim: str):
        """Record a claim made by another player (discussion speech)."""
        self.social_log.append({
            "tick": timestep,
            "actor": actor,
            "claim": claim,
            "type": "HEARSAY",
        })

    def update_memory(self, observation: str, timestep: int,
                      player_location: str):
        """Classify a raw observation string and route it to the correct
        memory store.  Called by the engine after each event dispatch.

        Classification rules:
          * ``[CONFIRMED EYEWITNESS]`` or kill/vent SAW keywords → VISUAL
          * ``said:`` from another player → HEARSAY (social_log)
          * ``[Discussion Round …]`` → HEARSAY
          * Everything else (room context, system notes) → VISUAL
        """
        obs_upper = observation.upper()

        # Eyewitness crime evidence → always VISUAL
        if "[CONFIRMED EYEWITNESS]" in obs_upper:
            self.add_verified(timestep, observation, player_location,
                              "VISUAL_CRIME")
            return

        if ("KILL" in obs_upper or "VENT" in obs_upper) and "SAW" in obs_upper:
            self.add_verified(timestep, observation, player_location,
                              "VISUAL_CRIME")
            return

        # Speech from another player → HEARSAY
        if "said:" in observation:
            import re as _re_mem
            actor_match = _re_mem.search(
                r'(Player \d+: \w+) said:', observation)
            actor = actor_match.group(1) if actor_match else "unknown"
            self.add_hearsay(timestep, actor, observation)
            return

        if observation.startswith("[Discussion Round"):
            import re as _re_mem2
            actor_match = _re_mem2.search(
                r'(Player \d+: \w+) said:', observation)
            actor = actor_match.group(1) if actor_match else "unknown"
            self.add_hearsay(timestep, actor, observation)
            return

        # Everything else (room changes, sabotage alerts, system notes) → VISUAL
        self.add_verified(timestep, observation, player_location)

    def update_task_commitment(self, player_location: str, tasks: list):
        """Recalculate task_commitment based on proximity to active task.

        * Player is in the same room as an in-progress task → 1.0
        * Player is in the same room as a pending task → 0.9
        * Otherwise → decay toward 0.0
        """
        for t in tasks:
            if t.check_completion():
                continue
            if t.location == player_location:
                if t.duration < t.max_duration:
                    # In-progress multi-turn task here
                    self.task_commitment = 1.0
                    self.current_intent = "TASK_EXECUTION"
                    return
                else:
                    # Pending task here (not started yet)
                    self.task_commitment = 0.9
                    self.current_intent = "TASK_EXECUTION"
                    return
        # No relevant task in room — decay
        self.task_commitment = max(0.0, self.task_commitment - 0.3)

    def record_own_statement(self, timestep: int, statement: str):
        """Record a public claim this agent made during a meeting.

        Called by the engine after the agent speaks so that later
        prompts can warn the agent not to contradict itself.
        """
        self.last_statement = statement
        self.own_claims.append({"tick": timestep, "claim": statement})
        # Keep only the last 8 claims to avoid bloat
        if len(self.own_claims) > 8:
            self.own_claims = self.own_claims[-8:]

    def set_dead(self):
        """Mark memory as dead — locks intent permanently."""
        self.current_intent = "DEAD"
        self.task_commitment = 0.0

    # ──────────────────────────────────────────────
    # Prompt helpers
    # ──────────────────────────────────────────────
    def verified_history_prompt(self, max_entries: int = 10) -> str:
        """Format the engine-recorded location + action history.

        This is an IMMUTABLE log the LLM cannot contradict.
        It replaces the LLM's self-reported "I was in X" with
        engine-verified facts.
        """
        if not self.location_history:
            return ""
        text = "## YOUR VERIFIED HISTORY (IMMUTABLE FACTS — logged by the engine) ##\n"
        text += ("The system has logged your exact movements. "
                 "You CANNOT lie about or misremember these locations.\n")
        for entry in self.location_history[-max_entries:]:
            room = entry[0]
            ts = entry[1]
            action = entry[2] if len(entry) > 2 and entry[2] else "—"
            text += f"  TIMESTEP {ts}: {room} ({action})\n"
        text += ("CONSTRAINT: If you claim to be in a room different "
                 "from this list, you are hallucinating. "
                 "Stick to these facts.\n\n")
        return text

    def hard_memory_prompt(self, max_entries: int = 12) -> str:
        """Format verified observations for the LLM prompt."""
        if not self.verified_observations:
            return ""
        text = "## YOUR HARD MEMORY (Facts you SAW — 100% reliable) ##\n"
        text += "These are events YOU personally witnessed. You can state them as FACT.\n"
        for entry in self.verified_observations[-max_entries:]:
            marker = "🔴" if entry["type"] == "VISUAL_CRIME" else "•"
            text += f"  {marker} T{entry['tick']} [{entry['location']}]: {entry['event']}\n"
        return text + "\n"

    def social_memory_prompt(self, max_entries: int = 10) -> str:
        """Format hearsay for the LLM prompt."""
        if not self.social_log:
            return ""
        text = "## YOUR SOCIAL MEMORY (What others SAID — may be lies) ##\n"
        text += ("CONSTRAINT: You CANNOT claim to have SEEN an event "
                 "unless it is in your HARD MEMORY above.\n"
                 "If it is only here, you MUST say 'Player X claimed that…' "
                 "or 'I heard that…'.\n")
        for entry in self.social_log[-max_entries:]:
            text += f"  T{entry['tick']} [{entry['actor']}]: {entry['claim']}\n"
        return text + "\n"

    def commitment_prompt(self) -> str:
        """Return a prompt snippet about task commitment."""
        if self.task_commitment >= 0.8:
            return (
                "\n⚠️ TASK COMMITMENT: 100% — You are currently working on "
                "a task. You MUST complete it before moving unless there is "
                "an active kill threat in the room.\n"
            )
        elif self.task_commitment >= 0.5:
            return (
                "\n⚠️ TASK COMMITMENT: MODERATE — You recently started a task "
                "nearby. Consider returning to finish it before wandering.\n"
            )
        return ""

    def consistency_prompt(self) -> str:
        """Return a prompt snippet warning against self-contradiction.

        Shows the agent its own previous claims so it doesn't flip
        its story mid-meeting (a common LLM failure mode).
        """
        if not self.own_claims:
            return ""
        text = "\n## CONSISTENCY CHECK (your own previous statements) ##\n"
        text += ("WARNING: You already said the following publicly. "
                 "Do NOT contradict yourself or you will look suspicious.\n")
        for entry in self.own_claims[-4:]:
            text += f"  T{entry['tick']}: \"{entry['claim']}\"\n"
        if self.last_statement:
            text += (f"Your MOST RECENT claim: \"{self.last_statement}\"\n"
                     "If you want to add new information, BUILD on this — "
                     "do not retract it unless you have new hard evidence.\n")
        return text + "\n"

    def crisis_prompt(self) -> str:
        """Return a prompt snippet about crisis dispatch role."""
        if self.crisis_role == "CRISIS_RESPONDER":
            return (
                "\n🚨 CRISIS DISPATCH: You are one of the 2 NEAREST players "
                "to the sabotage. YOU must respond immediately. "
                "Drop your task and run to fix it.\n"
            )
        elif self.crisis_role == "IGNORE_ALARM":
            return (
                "\n✅ CRISIS DISPATCH: Other players are closer to the "
                "sabotage. IGNORE the alarm and continue your current task. "
                "Do NOT abandon your work.\n"
            )
        return ""


class Player:
    def __init__(self, name, identity, color, personality, location=None):
        # Basic player information
        self.name = f"{name}: {color}"
        self.color = color
        self.identity = identity  # e.g., "Crewmate" or "Imposter"
        self.location = location  # Initially, the player might not have a location
        self.personality = personality

        # Player history
        self.observation_history = []
        self.action_history = []
        self.location_info = None
        
        # Memory Stream (internally: verified_presence_log): code-generated record of
        # who was actually in the same room as this player at each timestep.
        # Presented to agents as personal memory (not an omniscient log) to enforce
        # epistemic boundaries. Recorded by the game engine after each timestep.
        self.verified_presence_log = []

        # ═══ PERSISTENT STRUCTURED MEMORY ═══
        # Separates "what I SAW" from "what others SAID" so the LLM
        # never confuses hearsay with first-hand evidence.
        self.memory = MemoryState(player_name=f"{name}: {color}",
                                  role=identity)

        # Player options
        self.COMMON_ACTIONS = COMMON_ACTIONS
        self.SPECIAL_ACTIONS = []
        self.available_actions = []

        # Player status
        self.is_alive = True
        self.tasks = []
        self.reported_death = False
        
        # Death metadata: tracked by the engine when a player dies
        self.death_timestep = None   # Timestep when this player died
        self.death_cause = None      # "KILLED" or "EJECTED"

    def __repr__(self) -> str:
        return f"{self.name} ({self.identity})"

    def __str__(self) -> str:
        return self.__repr__()

    def assign_tasks(self, tasks):
        self.tasks = tasks
        for task in tasks:
            task.assign_to(self)

    def get_all_actions(self):
        """Return the actions that the player can take."""
        return self.COMMON_ACTIONS + self.SPECIAL_ACTIONS

    def set_available_actions(self, actions):
        self.available_actions = actions

    def get_available_actions(self):
        if self.is_alive:
            return self.available_actions
        else:
            # Ghosts can only do task-related actions (MoveTo, CompleteTask)
            ghost_actions = [a for a in self.available_actions 
                             if a.name in ("MOVE", "COMPLETE TASK", "COMPLETE FAKE TASK")]
            return ghost_actions

    def make_action(self, env, action, choose_location="Cafeteria"):
        if action.name == "ViewMonitor":
            action.execute(env, self, choose_location)
        else:
            action.execute(env, self)
        if env.current_phase == "task":
            record = {
                "timestep": env.timestep,
                "phase": env.current_phase,
                "action": action,
            }
        elif env.current_phase == "meeting":
            round = env.game_config["discussion_rounds"] - env.discussion_rounds_left
            record = {
                "timestep": env.timestep,
                "phase": env.current_phase,
                "round": round,
                "action": action,
            }
        self.action_history.append(record)

    def receive(self, message, info_type):
        if info_type == "location":
            self.location_info = message
        elif info_type == "action":
            self.observation_history.append(message)

    def location_info_prompt(self):
        import re as _re
        text = self.location_info
        # If LIGHTS sabotage is active, redact visible player names for Crewmates.
        # Impostors are unaffected (they sabotaged the lights and have night vision).
        active_sabs = getattr(self, 'active_sabotages', set())
        if "LIGHTS" in active_sabs and self.identity != "Impostor" and text:
            text = _re.sub(
                r'(Living Players in .+?:) .+',
                r'\1 [⚡ LIGHTS OUT — VISION REDUCED, CANNOT IDENTIFY PLAYERS]',
                text
            )
        return text

    def available_actions_prompt(self):
        text = f"Available actions (you are at {self.location} — you can ONLY choose from this list):\n"
        for i, action in enumerate(self.get_available_actions()):
            text += f"{i+1}. {action}\n"
        return text

    def action_history_prompt(self, recent_num=4):
        text = "Action history:\n"
        if len(self.action_history) == 0:
            text += "No actions have been taken yet.\n"
        else:
            for i, record in enumerate(self.action_history[-recent_num:]):
                timestep = record["timestep"]
                current_phase = record["phase"]
                action = record["action"]
                if current_phase == "task":
                    if type(action) == CompleteTask:
                        action_text = str(action)
                    else:
                        action_text = action.action_text()
                    text += (
                        f"Timestep {timestep}: [{current_phase} phase] {action_text}\n"
                    )
                elif current_phase == "meeting":
                    round = record["round"]
                    text += f"Timestep {timestep}: [{current_phase} phase - round {round}] {action.action_text()}\n"
        text += "\n"
        return text

    def observation_history_prompt(self, recent_num=4):
        text = "Observation history:\n"
        if len(self.observation_history) == 0:
            text += "No observations have been made yet.\n"
        else:
            # PINNED: Always show eyewitness evidence regardless of recency.
            # This prevents critical evidence from being lost when discussion messages
            # push older observations out of the recent window.
            critical_obs = []
            regular_obs = []
            for obs in self.observation_history:
                obs_upper = obs.upper()
                if "[CONFIRMED EYEWITNESS]" in obs_upper or ("KILL" in obs_upper and "SAW" in obs_upper):
                    critical_obs.append(obs)
                else:
                    regular_obs.append(obs)
            
            if critical_obs:
                text += "⚠️ PINNED — YOUR EYEWITNESS EVIDENCE (100% reliable, never doubt this):\n"
                for obs in critical_obs:
                    text += f"  🔴 {obs}\n"
                text += "\n"
            
            # Show recent regular observations
            for i, message in enumerate(regular_obs[-recent_num:]):
                text += f"{i+1}. {message}\n"
        text += "\n"
        return text

    def tasks_prompt(self):
        # If COMMS sabotage is active, Crewmates cannot see their task list
        active_sabs = getattr(self, 'active_sabotages', set())
        if "COMMS" in active_sabs and self.identity != "Impostor":
            return "Your Assigned Tasks:\n📡 [COMMUNICATIONS JAMMED — Task list temporarily unavailable. You cannot see your tasks or the task bar.]\n\n"
        
        if self.identity == "Impostor":
            text = "Your FAKE Tasks (use these as cover — stand in these rooms to look busy, and reference them by name when asked):\n"
        else:
            text = "Your Assigned Tasks:\n"
        if len(self.tasks) == 0:
            text += "No tasks have been assigned yet.\n"
        else:
            all_done = all(task.check_completion() for task in self.tasks)
            for i, task in enumerate(self.tasks):
                if task.check_completion():
                    # Completed tasks: show as done, NO path (prevents revisit loops)
                    text += f"{i+1}. {task} ✅ DONE — do NOT revisit this task\n"
                elif task.duration < task.max_duration:
                    # In-progress multi-turn task: show remaining turns
                    text += f"{i+1}. {task} ⏳ IN PROGRESS ({task.duration} turn{'s' if task.duration > 1 else ''} remaining — STAY in this room!)\n"
                else:
                    # Incomplete tasks: show path to guide the agent
                    turns_str = f" (requires {task.max_duration} turn{'s' if task.max_duration > 1 else ''})" if task.max_duration > 1 else ""
                    text += f"{i+1}. {task} [INCOMPLETE]{turns_str}\n"
                    path = task.find_path(self.location, identity=self.identity)
                    if len(path) > 1:
                        path = "->".join(path)
                    else:
                        path = "You are already at the task location."
                    text += f"   Path: {path}\n"
            
            # ALL TASKS COMPLETE: Give Crewmates explicit roaming instructions
            # This prevents the "ghost hallucination" where done-crewmates think they're dead
            if all_done and self.identity != "Impostor" and self.is_alive:
                text += "\n🎯 ALL YOUR TASKS ARE COMPLETE! You are still ALIVE and in the game.\n"
                text += "Your new mission: BE A WATCHDOG.\n"
                text += "- Follow other players and observe their behavior.\n"
                text += "- If you see anything suspicious (kill, vent), call an EMERGENCY MEETING immediately.\n"
                text += "- Stick with groups for safety. Do NOT wander alone.\n"
                text += "- You are NOT a ghost. You are a living Crewmate helping your team.\n"
        text += "\n"
        return text

    def verified_presence_prompt(self, max_entries=8):
        """Personal memory stream of what this player actually saw at each timestep.
        Structured as recall of rooms visited and who was present. This only
        contains information from rooms the player was physically in — it says
        nothing about rooms the player did NOT visit.
        """
        if not self.verified_presence_log:
            return ""
        
        text = "## YOUR MEMORY STREAM (what you personally remember seeing):\n"
        text += "⚠️ This ONLY covers rooms YOU were in. You have NO information about rooms you did not visit.\n"
        # Show the most recent entries (oldest to newest)
        entries = self.verified_presence_log[-max_entries:]
        for entry in entries:
            ts = entry["timestep"]
            room = entry["room"]
            others = entry["players_seen"]
            if others:
                text += f"  T{ts}: I was at {room} and saw {', '.join(others)}\n"
            else:
                text += f"  T{ts}: I was at {room} — no one else was there\n"
        text += "Use this to recall where YOU were and who YOU saw. You can only confirm or deny claims about rooms you personally visited.\n\n"
        return text

    def all_info_prompt(self):
        # ═══════════════════════════════════════════════════════════════
        # EXPLICIT STATUS HEADER: Rigid block the LLM cannot ignore.
        # Prevents hallucinations like "I'm a ghost" while alive.
        # ═══════════════════════════════════════════════════════════════
        status_str = "ALIVE" if self.is_alive else "DEAD (GHOST)"
        role_str = self.identity.upper()
        
        text = "╔══════════════════════════════════════════════╗\n"
        text += f"║ CURRENT STATUS : {status_str:<28}║\n"
        text += f"║ ROLE           : {role_str:<28}║\n"
        text += f"║ CURRENT ROOM   : {self.location:<28}║\n"
        text += "╚══════════════════════════════════════════════╝\n"
        
        if self.is_alive:
            text += "⚠️ You are a LIVING player. You are NOT a ghost. You are NOT dead. Do NOT reference being dead, being a ghost, or 'watching from the afterlife.'\n\n"
        else:
            text += "☠️ You are DEAD. You are a GHOST. You cannot speak in meetings or vote. Complete your tasks to help your team win.\n\n"
        
        # ═══════════════════════════════════════════════════════════════
        # OBJECTIVE HISTORY LOG: Rolling 3-turn summary of actual moves.
        # Prevents the LLM from inventing alibis or misremembering rooms.
        # ═══════════════════════════════════════════════════════════════
        if self.verified_presence_log:
            recent = self.verified_presence_log[-3:]
            path_parts = [f"T{e['timestep']}: {e['room']}" for e in recent]
            text += f"SYSTEM NOTE — YOUR RECENT MOVEMENTS (factual, cannot be disputed):\n"
            text += f"  {' → '.join(path_parts)}\n\n"
        
        # ═══════════════════════════════════════════════════════════════
        # CONTEXT SCRATCHPAD: Structured summary of critical facts.
        # LLMs forget what happened 2 turns ago. This compiles key info
        # (deaths, sightings, accusations) into one block so the agent
        # doesn't say "I have no information" when they DO have information.
        # ═══════════════════════════════════════════════════════════════
        scratchpad_lines = []
        
        # Deaths: who is confirmed dead?
        for obs in self.observation_history:
            obs_upper = obs.upper()
            if "CONFIRMED DEAD" in obs_upper or "CASUALTY" in obs_upper:
                # Extract the key death info
                scratchpad_lines.append(obs.strip())
                break  # Only need the most recent casualty report
        
        # Key sightings from Memory Stream: who you saw where
        if self.verified_presence_log:
            for entry in self.verified_presence_log:
                ts = entry["timestep"]
                room = entry["room"]
                others = entry["players_seen"]
                if others:
                    scratchpad_lines.append(f"T{ts}: You saw {', '.join(others)} in {room}")
        
        # Eyewitness evidence
        for obs in self.observation_history:
            obs_upper = obs.upper()
            if "[CONFIRMED EYEWITNESS]" in obs_upper or ("SAW" in obs_upper and ("KILL" in obs_upper or "VENT" in obs_upper)):
                scratchpad_lines.append(f"⚠️ {obs.strip()}")
        
        if scratchpad_lines:
            text += "## 📋 MEMORY SCRATCHPAD (key facts you MUST remember):\n"
            for line in scratchpad_lines[:12]:  # Cap at 12 entries to avoid prompt bloat
                text += f"  • {line}\n"
            text += "\n"
        
        text += self.location_info_prompt()
        text += self.verified_presence_prompt()
        text += self.observation_history_prompt()
        text += self.action_history_prompt()
        text += self.tasks_prompt()
        text += self.available_actions_prompt()
        return text

    # ═══════════════════════════════════════════════════════════════
    # STRUCTURED MEMORY STATE (JSON): The core of the State-Based Architecture.
    # This replaces text-heavy "stream of consciousness" prompts with a machine-
    # readable JSON block that the LLM must treat as ground truth.
    # The engine writes this; the LLM only reads it.
    # ═══════════════════════════════════════════════════════════════

    def get_memory_state_json(self, env=None):
        """Build the structured Player Memory State JSON.
        
        This is the ONLY information the agent receives about the world.
        It is a filtered, subjective view derived from the Global State (env).
        The LLM cannot modify or dispute this block.
        """
        import json as _json

        # --- Identity ---
        identity_block = {
            "color": self.color,
            "name": self.name,
            "role": self.identity.lower(),
            "status": "alive" if self.is_alive else "dead",
        }
        if not self.is_alive:
            identity_block["death_cause"] = self.death_cause or "unknown"
            identity_block["death_turn"] = self.death_timestep

        # --- Current Perception (what you see RIGHT NOW) ---
        # Derived from location_info which the engine already computed
        visible_players = []
        dead_bodies = []
        if self.location_info:
            import re as _re
            # Extract living players from location info
            match = _re.search(r'Living Players in .+?: (.+)', self.location_info)
            if match:
                raw = match.group(1).strip()
                if "LIGHTS OUT" not in raw and raw.lower() != "none":
                    visible_players = [p.strip() for p in raw.split(",") if p.strip()]
            # Extract dead bodies
            body_match = _re.search(r'Dead Bodies in .+?: (.+)', self.location_info)
            if body_match:
                dead_bodies = [b.strip() for b in body_match.group(1).split(",") if b.strip()]

        # Remove self from visible players
        visible_players = [p for p in visible_players if self.name not in p]

        active_sabs = list(getattr(self, 'active_sabotages', set()))

        perception = {
            "location": self.location,
            "visible_players": visible_players,
            "dead_bodies": dead_bodies,
            "sabotage_active": active_sabs if active_sabs else None,
        }

        # --- Short-Term Memory (last 5 turns of movement + sightings) ---
        short_term = []
        for entry in self.verified_presence_log[-5:]:
            short_term.append({
                "turn": entry["timestep"],
                "location": entry["room"],
                "saw": entry["players_seen"],
            })

        # --- Long-Term Memory (suspects, cleared, witnessed crimes) ---
        suspects = []
        cleared = []
        witnessed_crimes = []
        for obs in self.observation_history:
            obs_upper = obs.upper()
            if "[CONFIRMED EYEWITNESS]" in obs_upper:
                witnessed_crimes.append(obs.strip())
            elif "KILL" in obs_upper and "SAW" in obs_upper:
                witnessed_crimes.append(obs.strip())
            elif "VENT" in obs_upper and "SAW" in obs_upper:
                witnessed_crimes.append(obs.strip())

        long_term = {
            "witnessed_crimes": witnessed_crimes,
            "suspects": suspects,  # Populated by the LLM's World State Ledger
            "cleared": cleared,
        }

        # --- Tasks ---
        tasks_block = []
        for t in self.tasks:
            task_info = {
                "name": t.name,
                "location": t.location,
                "status": "done" if t.check_completion() else (
                    "in_progress" if t.duration < t.max_duration else "pending"
                ),
            }
            if not t.check_completion() and t.duration < t.max_duration:
                task_info["turns_remaining"] = t.duration
            elif not t.check_completion():
                task_info["turns_required"] = t.max_duration
            tasks_block.append(task_info)

        # --- Meeting Notes (what was said in meetings) ---
        meeting_notes = []
        for obs in self.observation_history:
            if "said:" in obs or "Discussion Summary" in obs:
                meeting_notes.append(obs.strip())

        state = {
            "my_identity": identity_block,
            "current_perception": perception,
            "short_term_memory": short_term,
            "long_term_memory": long_term,
            "tasks": tasks_block,
            "meeting_notes": meeting_notes[-8:] if meeting_notes else [],
        }

        return state

    def get_danger_score(self, env=None):
        """Calculate a danger score (0-100) for Crewmate self-preservation.
        
        High danger = stop tasks, seek safety.
        Components:
          - Alone in room: +30
          - Suspect nearby (from observation history): +25
          - Sabotage active: +20
          - Near a dead body: +25
          - Low player count (<=3 alive): +15
        """
        if self.identity == "Impostor" or not self.is_alive:
            return 0

        score = 0

        # Check if alone (only self in room)
        visible = []
        if self.location_info:
            import re as _re
            match = _re.search(r'Living Players in .+?: (.+)', self.location_info)
            if match:
                raw = match.group(1).strip()
                if "LIGHTS OUT" not in raw and raw.lower() != "none":
                    visible = [p.strip() for p in raw.split(",") if p.strip() and self.name not in p]
        
        if not visible:
            score += 30  # Alone in room

        # Sabotage active
        active_sabs = getattr(self, 'active_sabotages', set())
        if active_sabs:
            score += 20
            if "OXYGEN" in active_sabs or "REACTOR" in active_sabs:
                score += 15  # Critical sabotage

        # Dead body in room
        if self.location_info and "dead body" in self.location_info.lower():
            score += 25

        # Low player count
        if env:
            alive_count = sum(1 for p in env.players if p.is_alive)
            if alive_count <= 3:
                score += 15

        return min(score, 100)

    def get_alibi_ledger(self):
        """Build a structured alibi from short_term_memory for meeting use.
        
        Returns a list of location claims the player can make, derived
        strictly from their verified_presence_log (ground truth).
        This prevents the LLM from inventing alibis.
        """
        alibi = []
        for entry in self.verified_presence_log[-5:]:
            claim = {
                "turn": entry["timestep"],
                "location": entry["room"],
                "witnesses": entry["players_seen"],
            }
            alibi.append(claim)
        return alibi

    def check_contradictions(self, claims):
        """Cross-reference another player's claims against own memory.
        
        Args:
            claims: list of dicts [{"player": "X", "claimed_location": "Y", "turn": T}]
        
        Returns:
            list of contradiction dicts with evidence
        """
        contradictions = []
        for claim in claims:
            turn = claim.get("turn")
            claimed_loc = claim.get("claimed_location")
            claimer = claim.get("player")

            # Find my memory for that turn
            for entry in self.verified_presence_log:
                if entry["timestep"] == turn:
                    my_loc = entry["room"]
                    saw = entry["players_seen"]

                    # HARD CONTRADICTION: I was in that room and did NOT see them
                    if my_loc == claimed_loc and claimer not in " ".join(saw):
                        contradictions.append({
                            "type": "HARD_LIE",
                            "player": claimer,
                            "claim": f"was in {claimed_loc} at T{turn}",
                            "evidence": f"I was in {my_loc} at T{turn} and did NOT see {claimer}",
                        })
                    # ALIBI CONFIRM: I was there and DID see them
                    elif my_loc == claimed_loc and any(claimer in s for s in saw):
                        contradictions.append({
                            "type": "CONFIRMED",
                            "player": claimer,
                            "claim": f"was in {claimed_loc} at T{turn}",
                            "evidence": f"I was in {my_loc} at T{turn} and DID see {claimer}",
                        })
                    break
        return contradictions

    def has_witnessed_crime(self):
        """Check if player has witnessed a kill or vent action.
        Ghosts cannot act on witnessed crimes (no meetings/reports), so always return False for dead players.
        """
        if not self.is_alive:
            return False
        for obs in self.observation_history:
            obs_upper = obs.upper()
            if "KILL" in obs_upper or "VENT" in obs_upper:
                return True
        return False

    def has_suspicious_observation(self):
        """Check if player has any suspicious observation (not definite crime)."""
        for obs in self.observation_history:
            obs_lower = obs.lower()
            if "suspicious" in obs_lower or "impostor" in obs_lower:
                return True
        return False


class Crewmate(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, identity="Crewmate")
        self.SPECIAL_ACTIONS = CREWMATE_ACTIONS


class Impostor(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, identity="Impostor")
        self.SPECIAL_ACTIONS = IMPOSTER_ACTIONS
        self.kill_cooldown = 0
        
        # ═══ PHANTOM ALIBI: Fake Memory Track ═══
        # The Impostor maintains a parallel "fake" history of where they
        # CLAIM to have been. Updated each turn by the engine.
        # During meetings, the agent queries this instead of real memory.
        self.fake_memory = []  # [{turn, location, fake_action}]

    def update_fake_memory(self, timestep, real_location, fake_location=None, fake_action=None):
        """Add an entry to the Impostor's fake memory track.
        
        Called by the engine after each action. If the Impostor killed,
        fake_location should be the alibi room, not the real kill room.
        """
        entry = {
            "turn": timestep,
            "location": fake_location or real_location,
            "action": fake_action or "doing tasks",
        }
        self.fake_memory.append(entry)

    def get_fake_alibi_for_meeting(self):
        """Return the Impostor's prepared fake alibi for meetings.
        
        This gives the LLM a consistent story to tell, preventing
        the 'I was in Storage... wait, no, Admin' stumbling.
        """
        if not self.fake_memory:
            return "I was doing my tasks around the ship."
        
        recent = self.fake_memory[-5:]
        alibi_parts = []
        for entry in recent:
            alibi_parts.append(f"T{entry['turn']}: {entry['location']} ({entry['action']})")
        return alibi_parts

