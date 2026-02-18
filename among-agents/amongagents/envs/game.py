import random
import asyncio
import time

import numpy as np
import json
import os

from amongagents.agent.agent import HumanAgent, LLMAgent, LLMHumanAgent, RandomAgent
from amongagents.agent.neutral_prompts import (
    TASK_PHASE_INSTRUCTION,
    CrewmatePersonalities,
    ImpostorPersonalities,
)
from amongagents.envs.configs.agent_config import (
    ALL_LLM,
    ALL_RANDOM,
    CREWMATE_LLM,
    IMPOSTOR_LLM,
)
from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME
from amongagents.envs.map import Map, Spaceship
from amongagents.envs.player import PLAYER_COLORS, Crewmate, Impostor
from amongagents.envs.task import TaskAssignment
from amongagents.envs.tools import GetBestPath

# Set Flask environment variable to True by default
if "FLASK" not in os.environ:
    os.environ["FLASK"] = "True"

class AmongUs:
    def __init__(
        self,
        game_config=SEVEN_MEMBER_GAME,
        include_human=False,
        test=False,
        personality=False,
        agent_config=IMPOSTOR_LLM,
        interviewer=None,
        UI=None,
        game_index=0,
    ):
        """
        include_human: bool
            Whether to include a human player in the game.
        test: bool
            Whether to run the game in test mode. (All controlled by human inputs)
        agent_config: dict
            Agent initialization plan.
        interviewer: Interviewer
            Interviewer object to be used for the game to ask questions.
        UI: MapUI
            UI object to be used for the game to display the map.
        game_index: int
            Index of the game for logging purposes.
        """
        self.game_config = game_config
        self.include_human = include_human
        self.is_human_turn = False
        self.human_index = None
        self.test = test
        self.personality = personality
        self.identities = None
        self.agent_config = agent_config
        self.interviewer = interviewer
        self.UI = UI
        self.game_index = game_index
        self.map = Map()
        self.players = []
        self.agents = {}
        self.task_assignment = TaskAssignment(self.map.ship_map, self.game_config)
        self.current_phase = "TASK"
        self.timestep = 0
        self.activity_log = []
        self.important_activity_log = []
        self.camera_record = {}
        self.votes = {}
        self.vote_info_one_round = {}
        self.discussion_rounds_left = 0
        self.message_system = MessageSystem(game_config)
        self.game_over = False
        self.winner = None
        self.last_update = time.time()
        self.all_phases = ["meeting", "task"]
        self.summary_json = {f"Game {game_index}": {"config": game_config}}
        self.list_of_impostors = []

    def initialize_game(self):
        # reset game state
        if self.UI:
            self.UI.reset()
        self.players = []
        self.timestep = 0
        self.activity_log = []
        self.important_activity_log = []
        self.camera_record = {}
        self.button_num = 0
        self.task_assignment = TaskAssignment(self.map.ship_map, self.game_config)
        # meeting
        self.discussion_rounds_left = self.game_config["discussion_rounds"]
        self.votes = {}
        self.vote_info_one_round = {}
        self.meeting_caller = None  # Track who called the meeting
        self.dead_bodies = []  # Track locations of bodies separately from ghosts
        
        # Sabotage state: tracks active sabotages with remaining duration (turns)
        # e.g., {"LIGHTS": 2, "COMMS": 3} — decremented each timestep, removed at 0
        self.active_sabotages = {}

        # game state
        self.current_phase = "task"
        self.initialize_players()
        self.initialize_agents()
        self.agent_log = []

    # Starting locations pool: mix of safe (central) and isolated (peripheral) rooms.
    # 2 players always start in Cafeteria (safe hub), rest are scattered.
    # This creates immediate isolation — the Impostor may find a victim in Turn 0.
    SPAWN_LOCATIONS = [
        "Cafeteria", "Cafeteria",       # 2 guaranteed safe spawns
        "Admin", "Weapons", "Medbay",   # Medium-risk (adjacent to Cafeteria)
        "Electrical", "Navigation",     # High-risk (isolated, far from Cafeteria)
        "Reactor", "Security",          # High-risk
        "Upper Engine", "Shields",      # Varied
    ]
    
    def initialize_players(self):
        self.players = []
        num_players = self.game_config["num_players"]
        num_impostors = self.game_config["num_impostors"]
        num_crewmates = num_players - num_impostors
        identities = ["Crewmate"] * num_crewmates + ["Impostor"] * num_impostors
        colors = np.random.choice(PLAYER_COLORS, num_players, replace=False)
        np.random.shuffle(identities)
        self.identities = identities
        
        # Scatter starting positions: pick from the spawn pool (no duplicates beyond the 2 Cafeteria slots)
        spawn_pool = list(self.SPAWN_LOCATIONS[:num_players])
        random.shuffle(spawn_pool)
        
        for i in range(num_players):
            start_location = spawn_pool[i] if i < len(spawn_pool) else "Cafeteria"
            
            if identities[i] == "Crewmate":
                if self.personality:
                    crewmate_personality = random.choice(
                        list(CrewmatePersonalities.keys())
                    )
                else:
                    crewmate_personality = None
                player = Crewmate(
                    name=f"Player {i+1}",
                    color=colors[i],
                    location=start_location,
                    personality=crewmate_personality,
                )
            else:
                if self.personality:
                    imposter_personality = random.choice(
                        list(ImpostorPersonalities.keys())
                    )
                else:
                    imposter_personality = None
                player = Impostor(
                    name=f"Player {i+1}",
                    color=colors[i],
                    location=start_location,
                    personality=imposter_personality,
                )
            self.players.append(player)
            self.camera_record[player.name] = "stand quietly and do nothing"
        self.task_assignment.assign_tasks_to_players(self.players)
        self.update_map()

    def initialize_agents(self):
        random_idx = np.random.choice(len(self.players))
        if self.test:
            self.agents = [LLMHumanAgent(player) for player in self.players]
        else:
            tools = [GetBestPath(network=self.map.ship_map)]

            agent_dict = {
                "LLM": lambda player: LLMAgent(player, tools, self.game_index, self.agent_config, self.list_of_impostors),
                "Random": RandomAgent,
            }
            self.agents = []
            for i, player in enumerate(self.players):
                if self.include_human and i == random_idx:
                    # Create HumanAgent with game_id set to game_index
                    human_agent = HumanAgent(player, game_index=self.game_index)
                    human_agent.game_id = self.game_index
                    self.agents.append(human_agent)
                    self.human_index = i
                    print(f"{i} Initializing player {player.name} with identity {player.identity} and LLM choice {self.agents[-1].model}")
                    # Update max_steps for human agent
                    if hasattr(self.agents[-1], 'update_max_steps'):
                        self.agents[-1].update_max_steps(self.game_config.get("max_timesteps", 50))
                else:
                    self.agents.append(agent_dict[self.agent_config[player.identity]](player))
                    print(f"{i} Initializing player {player.name} with identity {player.identity} and LLM choice {self.agents[-1].model}")
                if player.identity == "Impostor":
                    self.list_of_impostors.append(player.name)
                    
                # add to summary json
                self.summary_json[f"Game {self.game_index}"]["Player " + str(i+1)] = {
                    "name": player.name,
                    "color": player.color,
                    "identity": player.identity,
                    "model": self.agents[-1].model,
                    "personality": player.personality,
                    "tasks": [task.name for task in player.tasks],
                }

    def report_winner(self, winner):
        winner_reason_map = {
            1: "Impostors win! (Crewmates being outnumbered or tied to impostors))",
            2: "Crewmates win! (Impostors eliminated)",
            3: "Crewmates win! (All task completed)",
            4: "Impostors win! (Time limit reached)",
        }
        text = winner_reason_map[winner]
        if self.UI:
            self.UI.report(text)
            self.UI.quit_UI()
        print(text)
        # add to summary json
        self.summary_json[f"Game {self.game_index}"]["winner"] = winner
        self.summary_json[f"Game {self.game_index}"]["winner_reason"] = winner_reason_map[winner]
        # finally, append the summary json to the experiment path as a single line json
        summary_path = os.path.join(os.environ["EXPERIMENT_PATH"], "summary.json")
        with open(summary_path, "a") as f:
            json.dump(self.summary_json, f, separators=(",", ": "))
            f.write("\n")

        return winner

    def check_game_over(self):
        num_impostors = sum(
            [
                1
                for player in self.players
                if player.identity == "Impostor" and player.is_alive
            ]
        )
        num_crewmates = sum(
            [
                1
                for player in self.players
                if player.identity == "Crewmate" and player.is_alive
            ]
        )
        if num_impostors >= num_crewmates:
            return 1  # Impostors win
        elif num_impostors == 0:
            return 2  # Crewmates win
        elif self.task_assignment.check_task_completion() == 1.0:
            return 3  # Crewmates win (task completed)
        elif self.timestep >= self.game_config["max_timesteps"]:
            return 4  # Impostors win (time limit)
        return 0  # Game continues

    def check_actions(self):
        # Check if a CRITICAL sabotage (O2 or Reactor) is active
        critical_sabotage_active = False
        critical_sab_type = None
        if hasattr(self, 'active_sabotages'):
            for sab in ("OXYGEN", "REACTOR"):
                if sab in self.active_sabotages:
                    critical_sabotage_active = True
                    critical_sab_type = sab
                    break
        
        for player in self.players:
            all_actions = player.get_all_actions()
            available_actions = []
            for action in all_actions:
                action_executables = action.can_execute_actions(self, player)
                available_actions.extend(action_executables)
            
            # GLOBAL EMERGENCY: When O2/Reactor is sabotaged, Crewmates CANNOT do tasks.
            # They must run to the sabotage room and FIX it, or everyone dies.
            # Impostors are unaffected (they want the sabotage to succeed).
            if critical_sabotage_active and player.identity != "Impostor" and self.current_phase == "task":
                available_actions = [a for a in available_actions if a.name != "COMPLETE TASK"]
            
            # COMMITMENT FLAG: If a player has a multi-turn task in progress
            # AND that task is in their CURRENT room, remove MOVE actions to lock
            # them in. This prevents the LLM from walking away mid-task (the #1
            # cause of task progress stalling).
            #
            # ENHANCED: Also locks the player if task_commitment >= 0.8 and
            # ANY incomplete task exists in this room (even if not yet started).
            # This prevents "infinite commuters" who arrive at their task room
            # then immediately leave without starting it.
            # Exceptions: dead body in room (must report) or critical sabotage (must fix).
            if self.current_phase == "task" and player.is_alive:
                has_local_in_progress_task = any(
                    not t.check_completion() and t.duration < t.max_duration
                    and t.location == player.location
                    for t in player.tasks
                )
                has_local_pending_task = any(
                    not t.check_completion() and t.location == player.location
                    for t in player.tasks
                )
                should_lock = (
                    has_local_in_progress_task
                    or (has_local_pending_task
                        and player.memory.task_commitment >= 0.8)
                )
                if should_lock:
                    body_in_room = any(
                        b["location"] == player.location and not b["reported"]
                        for b in self.dead_bodies
                    )
                    if not body_in_room and not critical_sabotage_active:
                        available_actions = [a for a in available_actions if a.name != "MOVE"]
            
            player.set_available_actions(available_actions)

    def update_map(self):
        self.map.reset()
        for player in self.players:
            self.map.add_player(player)
        self.message_system.route_location_info_message(self)
        if self.UI:
            self.UI.draw_map(self)

    async def agent_step(self, agent):
        self.check_actions()
        # Ghosts can still act (complete tasks) during task phase, but skip meeting phase
        if not agent.player.is_alive:
            if self.current_phase == "meeting":
                return
            # In task phase, ghosts might have limited actions, but we still allow them to move/task

        # kill cooldown
        if agent.player.identity == "Impostor" and agent.player.kill_cooldown > 0:
            agent.player.kill_cooldown -= 1

        # Set current player for UI updates
        self.current_player = agent.player.name

        # interview
        if self.interviewer is not None:
            await self.interviewer.auto_question(self, agent)

        # choose action
        action = await agent.choose_action(self.timestep)
        
        # Enforce meeting phase restrictions
        if self.current_phase == "meeting":
            if not agent.player.is_alive:
                # Ghost trying to speak or vote — should be caught earlier, but just in case:
                return None
            
            if self.discussion_rounds_left == 0:
                # VOTING PHASE: action MUST be VOTE
                if action is None or action.name != "VOTE":
                    skip_reason = "SKIP" if action is None else f"invalid ({action.name})"
                    print(f"[VOTE SKIP] {agent.player.name}: {skip_reason}. Recording as SKIP.")
                    self.vote_info_one_round[agent.player.name] = "SKIP"
                    # Broadcast the skip so other agents see it
                    skip_msg = f"[VOTE] {agent.player.name} chose to SKIP their vote."
                    for player in self.players:
                        if player.is_alive and player != agent.player:
                            player.observation_history.append(skip_msg)
                    return
            else:
                # DISCUSSION PHASE: action MUST be SPEAK
                if action is None or action.name not in ["SPEAK", "VOTE"]:
                    from amongagents.envs.action import Speak
                    print(f"[WARNING] {agent.player.name} tried to {action} during meeting. Forcing SPEAK.")
                    action = Speak(agent.player.location)
                    action.message = "..."


        # ─── DEAD PLAYER ACTION VALIDATOR ("State Enforcer") ───
        # Hard engine-level guard: dead players can ONLY move or complete tasks.
        # Any other action (SPEAK, VOTE, KILL, CALL MEETING, REPORT, SABOTAGE)
        # is replaced with the first valid ghost action.
        if not agent.player.is_alive and action is not None:
            GHOST_ALLOWED_ACTIONS = {"MOVE", "COMPLETE TASK", "COMPLETE FAKE TASK"}
            if action.name not in GHOST_ALLOWED_ACTIONS:
                available = agent.player.get_available_actions()
                ghost_actions = [a for a in available if a.name in GHOST_ALLOWED_ACTIONS]
                if ghost_actions:
                    print(f"[GHOST GUARD] {agent.player.name} is DEAD but tried to {action.name}. Redirecting to {ghost_actions[0].name}.")
                    action = ghost_actions[0]
                else:
                    print(f"[GHOST GUARD] {agent.player.name} is DEAD with no valid ghost actions. Skipping turn.")
                    return

        # ─── NO-SKIP ENFORCER ───
        # None/skip is ONLY allowed during voting (VOTE SKIP).
        # In task phase and discussion, we MUST execute a real action.
        if action is None:
            if self.current_phase == "meeting" and self.discussion_rounds_left == 0:
                # Voting phase: None means SKIP — already handled above, just return
                return
            # Task phase or discussion: force the first available action
            available = agent.player.get_available_actions()
            if available:
                action = available[0]
                print(f"[NO-SKIP] {agent.player.name}: None action in {self.current_phase} phase. Forcing {action.name}.")
            else:
                print(f"[NO-SKIP] {agent.player.name}: No available actions at all. Skipping turn.")
                return
            
        observation_location = ""
        if action.name == "ViewMonitor":
            observation_location = agent.choose_observation_location(
                self.map.ship_map.nodes
            )
        self.camera_record[agent.player.name] = action
        if str(action).startswith("KILL"):
            location = agent.player.location
            players = self.map.get_players_in_room(location)
            witness = [player.name for player in players]
            additional_info = f"Location: {location}, Witness: {witness}"
            self.record_activity(agent.player, action, additional_info)
        else:
            self.record_activity(agent.player, action)
        
        # ─── PHASE GUARD: Engine-level action validation ───
        # The LLM must not decide when a meeting starts. Only the engine can
        # authorize phase transitions. Validate the resolved action against
        # the player's actual available_actions computed by check_actions().
        # If the action isn't in the list (hallucinated), reject it.
        available_actions = agent.player.get_available_actions()
        action_is_valid = False
        for valid_action in available_actions:
            if action.name == valid_action.name:
                # For targeted actions (MOVE, KILL, VOTE), verify the target matches
                if hasattr(action, 'new_location') and hasattr(valid_action, 'new_location'):
                    if action.new_location == valid_action.new_location:
                        action_is_valid = True
                        break
                elif hasattr(action, 'other_player') and hasattr(valid_action, 'other_player'):
                    if action.other_player == valid_action.other_player:
                        action_is_valid = True
                        break
                else:
                    action_is_valid = True
                    break
        
        if not action_is_valid:
            # Critical: meeting-triggering actions (CALL MEETING / REPORT DEAD BODY)
            # that aren't in the available list are HARD REJECTED to prevent
            # invalid phase transitions.
            if action.name in ("CALL MEETING", "REPORT DEAD BODY"):
                print(f"[PHASE GUARD] {agent.player.name} tried to {action.name} but it's NOT in available actions. REJECTED.")
                # Fall back to first available action or skip
                if available_actions:
                    action = available_actions[0]
                    print(f"[PHASE GUARD] Falling back to: {action.name}")
                else:
                    print(f"[PHASE GUARD] No available actions — skipping turn.")
                    return
            else:
                # Non-critical action mismatch: log warning but allow (may be close match)
                print(f"[PHASE GUARD WARNING] {agent.player.name}: action '{action.name}' not exactly in available list — allowing.")

        # Track meeting caller for discussion ordering
        if action.name == "CALL MEETING":
            self.meeting_caller = agent
            
        agent.player.make_action(self, action, observation_location)
        
        # ═══ PHANTOM ALIBI ENGINE: Update Impostor's fake memory track ═══
        # After each action, record what the Impostor "claims" to have done.
        # On a KILL, the fake memory uses the public alibi room instead of the real location.
        if agent.player.identity == "Impostor" and hasattr(agent.player, 'fake_memory'):
            fake_loc = agent.player.location
            fake_act = "doing tasks"
            if action.name == "KILL":
                # Use the alibi room computed by the agent's deception ledger
                fake_loc = getattr(agent, 'public_alibi', agent.player.location)
                # Pick a fake task name if available
                fake_tasks = [t for t in agent.player.tasks if not t.check_completion()]
                fake_act = f"doing {fake_tasks[0].name}" if fake_tasks else "doing tasks"
            elif action.name == "COMPLETE FAKE TASK":
                fake_act = f"completing {action.task.name}" if hasattr(action, 'task') else "doing tasks"
            elif action.name == "SABOTAGE":
                fake_act = "doing tasks"  # Never reveal sabotage in fake memory
            elif action.name == "VENT":
                fake_loc = getattr(agent, 'public_alibi', agent.player.location)
                fake_act = "walking around"
            elif action.name == "MOVE":
                fake_act = f"heading to {agent.player.location}"
            agent.player.update_fake_memory(self.timestep, agent.player.location, fake_loc, fake_act)
        
        self.update_map()

    async def game_step(self):
        if self.current_phase == "task":
            await self.task_phase_step()
        elif self.current_phase == "meeting":
            await self.meeting_phase()
        
        # --- Verified Presence Log: record room-occupancy snapshot ---
        # This is a CODE-GENERATED record (not LLM-generated) of who was in the
        # same room as each player at the end of this timestep. It prevents "memory
        # hallucination" where agents gaslight themselves about who they actually saw.
        # Recorded AFTER all actions resolve so it reflects the final state.
        if self.current_phase == "task":
            for player in self.players:
                if not player.is_alive or not player.location:
                    continue  # Ghosts / unplaced players don't need presence tracking
                others_in_room = self.map.get_players_in_room(player.location)
                others_names = [p.name for p in others_in_room if p != player and p.is_alive]
                player.verified_presence_log.append({
                    "timestep": self.timestep,
                    "room": player.location,
                    "players_seen": others_names,
                })

                # ═══ STRUCTURED MEMORY UPDATE ═══
                # Feed the LOS-filtered observation into MemoryState
                obs = self.get_player_observation(player)
                # Derive action_taken from the player's most recent action_history entry
                last_action_str = ""
                if player.action_history:
                    last_rec = player.action_history[-1]
                    if last_rec.get("timestep") == self.timestep:
                        last_action_str = str(last_rec.get("action", ""))
                player.memory.update_location(player.location, self.timestep,
                                              last_action_str or "—")
                # Record who was visible as a verified observation
                if obs["visible_players"]:
                    vis_msg = f"Saw {', '.join(obs['visible_players'])} in {player.location}"
                    player.memory.add_verified(self.timestep, vis_msg,
                                               player.location)
                if obs["dead_bodies"]:
                    body_msg = (f"Dead body found: "
                                f"{', '.join(obs['dead_bodies'])} in {player.location}")
                    player.memory.add_verified(self.timestep, body_msg,
                                               player.location, "VISUAL_CRIME")

                # Update task commitment
                player.memory.update_task_commitment(player.location,
                                                     player.tasks)
                # Sync dead status
                if not player.is_alive:
                    player.memory.set_dead()
        
        self.timestep += 1
        # Decrement sabotage cooldown
        if hasattr(self, 'sabotage_cooldown') and self.sabotage_cooldown > 0:
            self.sabotage_cooldown -= 1
        
        # Decrement active sabotage timers and remove expired ones
        if hasattr(self, 'active_sabotages'):
            expired = [sab for sab, timer in self.active_sabotages.items() if timer <= 1]
            for sab in expired:
                del self.active_sabotages[sab]
                # Notify all players the sabotage was auto-fixed
                fix_msg = f"[SYSTEM] {sab} sabotage has been automatically repaired."
                for p in self.players:
                    if p.is_alive:
                        p.observation_history.append(fix_msg)
            for sab in list(self.active_sabotages.keys()):
                self.active_sabotages[sab] -= 1
        
        # Propagate active sabotage state to all players (used by prompt generation)
        active_sab_set = set(self.active_sabotages.keys()) if hasattr(self, 'active_sabotages') else set()
        for player in self.players:
            player.active_sabotages = active_sab_set

        # ═══ CRISIS DISPATCH: tag nearest 2 Crewmates as responders ═══
        # Re-evaluate every tick while a critical sabotage is active so that
        # if the nearest player dies or fixes it, another can be dispatched.
        critical_active = False
        for sab in ("OXYGEN", "REACTOR"):
            if sab in active_sab_set:
                self.crisis_dispatch(sab)
                critical_active = True
                break
        if not critical_active:
            # Clear crisis roles when no critical sabotage is active
            for player in self.players:
                player.memory.crisis_role = None

        print(f"|", end="", flush=True)
        # import pdb; pdb.set_trace() # waiting after each timestep

    async def task_phase_step(self):
        """Phased execution: DECISIONS → MOVES → VISUALS → ACTIONS.

        All agents decide their action based on the SAME world-state
        (start of this timestep).  Then:
          Phase 1 – Execute all MOVEs/VENTs (players arrive in rooms).
          Phase 2 – Visual snapshot (update who sees whom AFTER moves).
          Phase 3 – Execute non-movement actions (KILL, TASK, SABOTAGE…).

        This ensures that a player who moves INTO a room on the same
        tick as a kill will WITNESS the kill (they arrived before it
        happened).
        """

        # ── PRE-CHECK: Forced body reports ──────────────────────────
        # If ANY living player is standing on an unreported body at the
        # START of this timestep, they report immediately and no other
        # actions happen.
        for agent in self.agents:
            if agent.player.is_alive:
                body_in_room = any(
                    b["location"] == agent.player.location and not b["reported"]
                    for b in self.dead_bodies
                )
                if body_in_room:
                    from amongagents.envs.action import CallMeeting
                    forced_report = CallMeeting(
                        current_location=agent.player.location)
                    print(f"[FORCED REPORT] {agent.player.name} found a "
                          f"dead body in {agent.player.location} — auto-reporting!")
                    agent.player.observation_history.append(
                        f"[SYSTEM] You discovered a dead body in "
                        f"{agent.player.location}! Reporting immediately."
                    )
                    self.meeting_caller = agent
                    agent.player.make_action(self, forced_report)
                    self.update_map()
                    return  # Meeting starts — no further actions

        # ═══════════════════════════════════════════════════════════
        # PHASE 0 — DECISIONS: Every agent picks an action.
        # All decisions are based on the world-state at the START
        # of this timestep (nobody has moved or acted yet).
        # ═══════════════════════════════════════════════════════════
        self.check_actions()
        # (agent, validated_action, observation_location)
        decisions: list = []

        for agent in self.agents:
            # Kill cooldown tick
            if (agent.player.identity == "Impostor"
                    and agent.player.kill_cooldown > 0):
                agent.player.kill_cooldown -= 1

            self.current_player = agent.player.name

            # Human player flag
            agent_model = getattr(agent, 'model', '')
            if 'homosapiens' in agent_model:
                self.is_human_turn = True
            else:
                self.is_human_turn = False

            # Interview hook (optional)
            if self.interviewer is not None:
                await self.interviewer.auto_question(self, agent)

            # LLM / Random decision
            action = await agent.choose_action(self.timestep)

            # ─── DEAD PLAYER VALIDATOR ───
            if not agent.player.is_alive and action is not None:
                GHOST_OK = {"MOVE", "COMPLETE TASK", "COMPLETE FAKE TASK"}
                if action.name not in GHOST_OK:
                    available = agent.player.get_available_actions()
                    ghost_acts = [a for a in available
                                  if a.name in GHOST_OK]
                    if ghost_acts:
                        print(f"[GHOST GUARD] {agent.player.name}: "
                              f"{action.name} → {ghost_acts[0].name}")
                        action = ghost_acts[0]
                    else:
                        continue  # No valid ghost action — skip

            # ─── NO-SKIP ENFORCER ───
            if action is None:
                available = agent.player.get_available_actions()
                if available:
                    action = available[0]
                    print(f"[NO-SKIP] {agent.player.name}: forcing "
                          f"{action.name}")
                else:
                    continue

            # ─── PHASE GUARD: reject hallucinated meetings ───
            if action.name in ("CALL MEETING", "REPORT DEAD BODY"):
                available = agent.player.get_available_actions()
                if not any(a.name == action.name for a in available):
                    print(f"[PHASE GUARD] {agent.player.name}: "
                          f"{action.name} REJECTED (not available)")
                    action = available[0] if available else None
                    if action is None:
                        continue

            # ViewMonitor observation location
            obs_loc = ""
            if action.name == "ViewMonitor":
                obs_loc = agent.choose_observation_location(
                    self.map.ship_map.nodes)

            decisions.append((agent, action, obs_loc))

        # ═══════════════════════════════════════════════════════════
        # PHASE 1 — RESOLVE MOVEMENTS (MOVE + VENT)
        # Players arrive in their target rooms BEFORE any kills,
        # tasks, or sabotages are processed.
        # ═══════════════════════════════════════════════════════════
        for agent, action, _ in decisions:
            if action.name not in ("MOVE", "VENT"):
                continue
            self.camera_record[agent.player.name] = action
            self.record_activity(agent.player, action)
            agent.player.make_action(self, action)

            # Phantom alibi for Impostor movements
            if (agent.player.identity == "Impostor"
                    and hasattr(agent.player, 'fake_memory')):
                if action.name == "VENT":
                    fake_loc = getattr(agent, 'public_alibi',
                                       agent.player.location)
                    agent.player.update_fake_memory(
                        self.timestep, agent.player.location,
                        fake_loc, "walking around")
                else:
                    agent.player.update_fake_memory(
                        self.timestep, agent.player.location,
                        agent.player.location,
                        f"heading to {agent.player.location}")

        # ═══════════════════════════════════════════════════════════
        # PHASE 2 — VISUAL SNAPSHOT
        # Recalculate room occupancy so everyone sees who is present
        # AFTER all moves have landed.
        # ═══════════════════════════════════════════════════════════
        self.update_map()

        # ═══════════════════════════════════════════════════════════
        # PHASE 3 — RESOLVE NON-MOVEMENT ACTIONS
        # KILL, COMPLETE TASK, SABOTAGE, FIX SABOTAGE, VIEW MONITOR,
        # CALL MEETING / REPORT DEAD BODY.
        # ═══════════════════════════════════════════════════════════
        for agent, action, obs_loc in decisions:
            if action.name in ("MOVE", "VENT"):
                continue  # Already resolved in Phase 1

            # Re-validate KILL targets (target may have moved in Phase 1)
            if action.name == "KILL":
                if not action.other_player.is_alive:
                    print(f"[PHASE GUARD] {agent.player.name}: kill target "
                          f"already dead — skipping.")
                    continue
                if agent.player.location != action.other_player.location:
                    print(f"[PHASE GUARD] {agent.player.name}: kill target "
                          f"{action.other_player.name} moved away — "
                          f"skipping kill.")
                    continue

            # Record activity (with witness list for kills)
            self.camera_record[agent.player.name] = action
            if action.name == "KILL":
                players = self.map.get_players_in_room(agent.player.location)
                witness = [p.name for p in players]
                self.record_activity(agent.player, action,
                                     f"Location: {agent.player.location}, "
                                     f"Witness: {witness}")
            else:
                self.record_activity(agent.player, action)

            # Track meeting caller
            if action.name == "CALL MEETING":
                self.meeting_caller = agent

            # Execute the action
            if action.name == "ViewMonitor":
                agent.player.make_action(self, action, obs_loc)
            else:
                agent.player.make_action(self, action)

            # Phantom alibi for non-movement Impostor actions
            if (agent.player.identity == "Impostor"
                    and hasattr(agent.player, 'fake_memory')
                    and action.name not in ("MOVE", "VENT")):
                fake_loc = agent.player.location
                fake_act = "doing tasks"
                if action.name == "KILL":
                    fake_loc = getattr(agent, 'public_alibi',
                                       agent.player.location)
                    fake_tasks = [t for t in agent.player.tasks
                                  if not t.check_completion()]
                    fake_act = (f"doing {fake_tasks[0].name}"
                                if fake_tasks else "doing tasks")
                elif action.name == "COMPLETE FAKE TASK":
                    fake_act = (f"completing {action.task.name}"
                                if hasattr(action, 'task')
                                else "doing tasks")
                elif action.name == "SABOTAGE":
                    fake_act = "doing tasks"
                agent.player.update_fake_memory(
                    self.timestep, agent.player.location,
                    fake_loc, fake_act)

            self.update_map()

            # If a meeting was triggered, stop processing actions
            if self.current_phase == "meeting":
                break

        # ── POST-CHECK: Body report after kills ─────────────────
        # If a kill happened in Phase 3 and a living player is now
        # standing on the body, they will discover it at the START
        # of the next timestep (via the pre-check above).

    async def meeting_phase(self):
        # Ensure phase is set correctly
        self.current_phase = "meeting"
        
        # Players stay in their pre-meeting positions. No teleport to Cafeteria.
        # This preserves spatial state: after the meeting, players resume from
        # where they were, keeping in-progress task locks and positions intact.

        self.update_map()
        
        # CRITICAL: Update available actions for the new phase and location
        self.check_actions()
        
        # ── CASUALTY REPORT: inject dead-player awareness into all living agents ──
        # Without this, agents ignore that someone is dead and fail to investigate
        # the kill, leading to aimless "what task were you doing?" loops.
        dead_players = [p for p in self.players if not p.is_alive]
        caller_name = self.meeting_caller.player.name if self.meeting_caller else "unknown"
        
        if dead_players:
            casualty_lines = ["╔══════════════════════════════════════╗"]
            casualty_lines.append("║     ☠️  CASUALTY REPORT  ☠️          ║")
            casualty_lines.append("╠══════════════════════════════════════╣")
            casualty_lines.append(f"║ Meeting called by: {caller_name}")
            
            for dp in dead_players:
                # Find where the body was (from dead_bodies list or last known location)
                body_location = "unknown"
                for body in self.dead_bodies:
                    if body["player_name"] == dp.name:
                        body_location = body["location"]
                        break
                casualty_lines.append(f"║ CONFIRMED DEAD: {dp.name} — body found in {body_location}")
            
            casualty_lines.append("╠══════════════════════════════════════╣")
            casualty_lines.append("║ KEY QUESTION: Who was near the body?")
            casualty_lines.append("║ Investigate: Who was last seen with")
            casualty_lines.append(f"║ the victim? Who has NO alibi?")
            casualty_lines.append("╚══════════════════════════════════════╝")
            casualty_msg = "\n".join(casualty_lines)
            
            # Inject into all living players' observation history
            for player in self.players:
                if player.is_alive:
                    player.observation_history.append(casualty_msg)
        else:
            # Emergency button pressed with no deaths — note this
            button_msg = f"[EMERGENCY MEETING] Called by {caller_name}. No confirmed deaths."
            for player in self.players:
                if player.is_alive:
                    player.observation_history.append(button_msg)

        # Helper function to calculate dynamic speaker priority
        def get_speaker_priority(agent):
            """Score based on urgency - higher = speaks sooner.
            
            Priority factors:
            - Defense: Player was accused (+15)
            - Witnessed crime: Saw kill/vent (+10)  
            - Suspicious activity: Has something suspicious to share (+5)
            """
            score = 0
            player_name = agent.player.name
            
            # Check if this player was accused in recent observations
            for other_agent in self.agents:
                if other_agent == agent:
                    continue
                for obs in other_agent.player.observation_history[-5:]:
                    obs_lower = obs.lower()
                    if player_name.lower() in obs_lower and ("impostor" in obs_lower or "kill" in obs_lower or "suspicious" in obs_lower or "vote" in obs_lower):
                        score += 15  # Defense priority - was accused
                        break
            
            # Witnessed crime - definite evidence
            if agent.player.has_witnessed_crime():
                score += 10
            # Suspicious activity - not definite but worth sharing
            elif agent.player.has_suspicious_observation():
                score += 5
                    
            return score

        # Build ordered speaker list for discussion
        living_agents = [a for a in self.agents if a.player.is_alive]
        
        # Staged Discussion (Testimony → Accusation/Defense → Final Arguments)
        for round_num in range(self.game_config["discussion_rounds"]):
            stage_names = {0: "TESTIMONY", 1: "ACCUSATION/DEFENSE", 2: "FINAL ARGUMENTS"}
            stage_name = stage_names.get(round_num, f"ROUND {round_num}")
            print(f"Discussion round {round_num} — {stage_name}")
            
            # Set the current meeting stage on all players so agents can adapt their prompts
            for player in self.players:
                player.current_meeting_stage = round_num
            
            # First round: meeting caller speaks first
            if round_num == 0 and self.meeting_caller and self.meeting_caller.player.is_alive:
                ordered_agents = [self.meeting_caller]
                other_agents = [a for a in living_agents if a != self.meeting_caller]
                # Sort others by dynamic priority (highest first)
                other_agents.sort(key=get_speaker_priority, reverse=True)
                ordered_agents.extend(other_agents)
            else:
                # Subsequent rounds: order by dynamic priority
                ordered_agents = sorted(living_agents, key=get_speaker_priority, reverse=True)
            
            for agent in ordered_agents:
                if not agent.player.is_alive:
                    continue  # Ghosts cannot participate in meetings
                if 'homosapiens' in agent.model:
                    self.is_human_turn = True
                else:
                    self.is_human_turn = False
                await self.agent_step(agent)
            self.discussion_rounds_left -= 1
            
            # Summarize the completed round to prevent context overflow.
            # Replaces individual speech observations with a condensed summary
            # so that earlier eyewitness evidence isn't pushed out of context.
            self._summarize_discussion_round(round_num)
            
            # Update game state after each round
            self.check_actions()
            self.update_map()

        # Voting phase (only living players can vote — exactly once each)
        print("Voting phase")
        self.vote_info_one_round = {}
        voted_players = set()  # Guard against duplicate voting
        for agent in self.agents:
            if not agent.player.is_alive:
                continue  # Ghosts cannot vote
            if agent.player.name in voted_players:
                print(f"[WARNING] {agent.player.name} already voted — skipping duplicate.")
                continue
            if 'homosapiens' in agent.model:
                self.is_human_turn = True
            else:
                self.is_human_turn = False
            await self.agent_step(agent)
            voted_players.add(agent.player.name)
            # Update game state after each vote
            self.check_actions()
            self.update_map()

        # Vote out
        self.voteout()
        self.update_map()

    def voteout(self):
        round = self.game_config["discussion_rounds"] - self.discussion_rounds_left
        
        # Guard against empty votes dict (e.g., if all players failed to vote)
        if not self.votes:
            print("== No votes were cast — no one was voted out ==")
            no_vote_msg = "VOTE RESULT: No votes were cast. No one was ejected."
            # Broadcast to all living players so they have a record
            for player in self.players:
                if player.is_alive:
                    player.observation_history.append(no_vote_msg)
            import_event = {
                "timestep": self.timestep,
                "phase": self.current_phase,
                "round": round,
                "action": no_vote_msg,
                "player": "all players",
            }
            self.important_activity_log.append(import_event)
            self.current_phase = "task"
            self.discussion_rounds_left = self.game_config["discussion_rounds"]
            self.votes = {}
            return
        
        # === BUILD FORMATTED VOTE TALLY ===
        # Sort by vote count descending for readability
        sorted_targets = sorted(self.votes.items(), key=lambda x: x[1], reverse=True)
        max_votes = sorted_targets[0][1]
        total_votes = sum(self.votes.values())
        
        # Count SKIPs from vote_info_one_round
        skip_count = sum(1 for v in self.vote_info_one_round.values() if v == "SKIP")
        
        tally_lines = ["╔══════════════════════════════════╗"]
        tally_lines.append("║        VOTE TALLY RESULTS        ║")
        tally_lines.append("╠══════════════════════════════════╣")
        
        # Individual votes: who voted for whom
        for voter, vote_target in self.vote_info_one_round.items():
            tally_lines.append(f"║ {voter} → {vote_target}")
        tally_lines.append("╠══════════════════════════════════╣")
        
        # Aggregated counts
        for player_obj, count in sorted_targets:
            bar = "█" * count
            tally_lines.append(f"║ {player_obj.name}: {count} vote{'s' if count != 1 else ''} {bar}")
        if skip_count > 0:
            tally_lines.append(f"║ SKIP: {skip_count} vote{'s' if skip_count != 1 else ''}")
        
        # Determine outcome
        # ═══ CRITICAL FIX: SKIP votes must be compared against player votes ═══
        # Previously, the code only checked if one player had the most votes
        # among non-SKIP votes. This meant a single vote for Player X would
        # eject them even if 3 other players voted SKIP.
        # Fix: A player is only ejected if they have STRICTLY MORE votes than
        # the SKIP count AND they are the sole leader.
        players_with_max_votes = [
            player for player, votes in self.votes.items() if votes == max_votes
        ]
        
        tally_lines.append("╠══════════════════════════════════╣")
        
        if (len(players_with_max_votes) == 1
                and max_votes > skip_count):
            # One player has strictly more votes than SKIP — eject them
            ejected_player = players_with_max_votes[0]
            ejected_player.is_alive = False
            ejected_player.death_timestep = self.timestep
            ejected_player.death_cause = "EJECTED"
            tally_lines.append(f"║ RESULT: {ejected_player.name} was EJECTED.")
            
            # Count remaining impostors
            remaining_impostors = sum(1 for p in self.players if p.is_alive and p.identity == "Impostor")
            if ejected_player.identity == "Impostor":
                tally_lines.append(f"║ {ejected_player.name} was an Impostor. ({remaining_impostors} Impostor{'s' if remaining_impostors != 1 else ''} remain)")
            else:
                tally_lines.append(f"║ {ejected_player.name} was NOT an Impostor. ({remaining_impostors} Impostor{'s' if remaining_impostors != 1 else ''} remain)")
            
            result_text = f"{ejected_player.name} was voted out ({max_votes} votes)."
            print(f"== {ejected_player.name} was voted out ({max_votes}/{total_votes} votes) ==")
        elif len(players_with_max_votes) > 1:
            # TIE between players: no one is ejected
            tied_names = [p.name for p in players_with_max_votes]
            tally_lines.append(f"║ RESULT: TIE between {', '.join(tied_names)} ({max_votes} votes each).")
            tally_lines.append(f"║ No one was ejected.")
            result_text = f"Tie vote ({', '.join(tied_names)} each got {max_votes} votes). No one was ejected."
            print(f"== Vote tied between {', '.join(tied_names)} — no one was voted out ==")
        else:
            # SKIP won (skip_count >= max_votes) or no one got enough votes
            tally_lines.append(f"║ RESULT: No one was ejected (SKIP won with {skip_count} votes).")
            result_text = f"No one was ejected. SKIP won with {skip_count} votes."
            print(f"== No one was voted out (SKIP won: {skip_count} votes vs max player votes: {max_votes}) ==")
        
        tally_lines.append("╚══════════════════════════════════╝")
        tally_message = "\n".join(tally_lines)
        
        # Print full tally to console
        print(tally_message)
        
        # Broadcast tally to ALL living players so they remember it for future meetings
        for player in self.players:
            if player.is_alive:
                player.observation_history.append(f"[VOTE RESULT] {result_text}")
        
        # Build vote_info for the activity log
        vote_info = [f"{voter} voted for {target}" for voter, target in self.vote_info_one_round.items()]
        
        import_event = {
            "timestep": self.timestep,
            "phase": self.current_phase,
            "round": round,
            "action": f"{result_text} Detailed: {vote_info}",
            "player": "all players",
        }
        self.important_activity_log.append(import_event)
        self.current_phase = "task"
        self.discussion_rounds_left = self.game_config["discussion_rounds"]
        self.votes = {}

    def _summarize_discussion_round(self, round_num):
        """After a discussion round completes, replace individual speech observations
        with a single condensed summary. This prevents context overflow where discussion
        messages push critical eyewitness evidence out of the observation window.
        
        The speech tag format from Speak.execute is:
        [Discussion Round {round_num}] Player X: color said: "message"
        """
        # Calculate the tag used by Speak.execute for this round
        # Speak.execute uses: env.game_config["discussion_rounds"] - env.discussion_rounds_left
        # After self.discussion_rounds_left is decremented, the round number in the tag is:
        speak_round_num = self.game_config["discussion_rounds"] - self.discussion_rounds_left
        tag = f"[Discussion Round {speak_round_num}]"
        
        # Collect all speeches from this round from any living player's observation history
        speeches = []
        sample_player = None
        for agent in self.agents:
            if agent.player.is_alive:
                sample_player = agent.player
                break
        
        if not sample_player:
            return
        
        for obs in sample_player.observation_history:
            if tag in obs:
                speeches.append(obs)
        
        if not speeches:
            return
        
        # Build condensed summary
        summary_lines = [f"=== Round {round_num + 1} Discussion Summary ==="]
        for speech in speeches:
            # Strip the round tag prefix for cleaner display
            clean_speech = speech.replace(tag + " ", "")
            summary_lines.append(f"  - {clean_speech}")
        summary = "\n".join(summary_lines)
        
        # Replace individual speech observations with summary for all living players
        for agent in self.agents:
            if agent.player.is_alive:
                agent.player.observation_history = [
                    obs for obs in agent.player.observation_history
                    if tag not in obs
                ]
                agent.player.observation_history.append(summary)

    # ═══════════════════════════════════════════════════════════════
    # OBSERVATION FILTER: Line-of-Sight (LOS) sanitized view
    # Instead of passing the full GameState, this function calculates
    # what a specific player can actually see from their current room.
    # ═══════════════════════════════════════════════════════════════

    def get_player_observation(self, player):
        """Generate a LOS-filtered observation for a single player.

        Returns a dict describing ONLY what this player can perceive
        from their current room.  Everything else is invisible.
        """
        room = player.location
        if not room:
            return {"location": None, "visible_players": [],
                    "dead_bodies": [], "events": []}

        # Players in the same room (LOS = True)
        visible = self.map.get_players_in_room(room)
        visible_names = [p.name for p in visible
                         if p != player and p.is_alive]

        # Dead bodies in this room only
        bodies_here = [
            b["player_name"] for b in self.dead_bodies
            if b["location"] == room and not b["reported"]
        ]

        # Active sabotage events (global — everyone hears the alarm)
        active_sabs = list(self.active_sabotages.keys()) if hasattr(
            self, 'active_sabotages') else []

        return {
            "location": room,
            "visible_players": visible_names,
            "dead_bodies": bodies_here,
            "sabotage_active": active_sabs,
        }

    # ═══════════════════════════════════════════════════════════════
    # CRISIS DISPATCH: Only the 2 nearest players respond to sabotage.
    # Everyone else ignores the alarm and continues their tasks.
    # ═══════════════════════════════════════════════════════════════

    def crisis_dispatch(self, sabotage_type: str):
        """Calculate the 2 nearest living Crewmates to the sabotage
        fix location and tag them as CRISIS_RESPONDER.  All others
        get IGNORE_ALARM so they keep doing tasks.

        Uses networkx shortest_path_length on the map graph.
        """
        import networkx as nx
        from amongagents.envs.action import FixSabotage

        fix_room = FixSabotage.FIX_LOCATIONS.get(sabotage_type)
        if not fix_room:
            return

        # Calculate graph distances for all living crewmates
        distances = []
        for p in self.players:
            if not p.is_alive or p.identity == "Impostor":
                continue
            try:
                dist = nx.shortest_path_length(
                    self.map.ship_map, p.location, fix_room, weight=None
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = 999
            distances.append((p, dist))

        # Sort by distance (nearest first)
        distances.sort(key=lambda x: x[1])

        # Tag the 2 nearest as responders, everyone else as ignore
        responders = {d[0].name for d in distances[:2]}
        for p in self.players:
            if not p.is_alive or p.identity == "Impostor":
                p.memory.crisis_role = None
                continue
            if p.name in responders:
                p.memory.crisis_role = "CRISIS_RESPONDER"
                p.memory.current_intent = "CRISIS_RESPONSE"
            else:
                p.memory.crisis_role = "IGNORE_ALARM"
                # Don't change intent — they stay on TASK_EXECUTION

    def check_monitor(self, room):
        players = self.map.get_players_in_room(room)
        return players

    async def run_game(self):
        self.initialize_game()
        game_over = self.check_game_over()
        while not game_over:
            await self.game_step()
            game_over = self.check_game_over()

        # interview
        if self.interviewer is not None:
            for agent in self.agents:
                await self.interviewer.auto_question(self, agent)
        return self.report_winner(game_over)

    def record_activity(self, player, action, additional_info=None):
        # ── Snapshot game state at record time (for post-game RL analysis) ──
        living_crew = sum(1 for p in self.players
                         if p.identity == "Crewmate" and p.is_alive)
        living_imps = sum(1 for p in self.players
                         if p.identity == "Impostor" and p.is_alive)
        task_pct = self.task_assignment.check_task_completion() * 100.0
        sabotage = bool(getattr(self, 'active_sabotages', {}))
        snapshot = {
            "living_crew": living_crew,
            "living_imps": living_imps,
            "task_pct": task_pct,
            "sabotage_active": sabotage,
            "player_alive": player.is_alive,
            "player_location": getattr(player, "location", "Unknown"),
        }

        if self.current_phase == "task":
            record = {
                "timestep": self.timestep,
                "phase": self.current_phase,
                "action": action,
                "player": player,
                "state": snapshot,
            }
        elif self.current_phase == "meeting":
            round = self.game_config["discussion_rounds"] - self.discussion_rounds_left
            record = {
                "timestep": self.timestep,
                "phase": self.current_phase,
                "round": round,
                "action": action,
                "player": player,
                "state": snapshot,
            }
        self.activity_log.append(record)
        # print(record)
        # print('.', end='', flush=True)
        self.message_system.route_real_time_message(self, record)
        if str(record["action"]).startswith("COMPLETE TASK"):
            imprtant_event = {
                "timestep": self.timestep,
                "phase": self.current_phase,
                "action": str(action),
                "player": player.name,
            }
            self.important_activity_log.append(record)
        if str(record["action"]).startswith("KILL"):
            imprtant_event = {
                "timestep": self.timestep,
                "phase": self.current_phase,
                "action": str(action) + "|||" + additional_info,
                "player": player.name,
            }
            self.important_activity_log.append(imprtant_event)


class MessageSystem:
    def __init__(self, game_config):
        self.game_config = game_config

    def send_message(self, player, message, info_type):
        player.receive(message, info_type)

    def create_action_message(self, record):
        timestep = record["timestep"]
        current_phase = record["phase"]
        player = record["player"]
        action = record["action"]
        if current_phase == "task":
            message = f"Timestep {timestep}: [{current_phase}] {player.name} {action.action_text()}"
        elif current_phase == "meeting":
            round = record["round"]
            message = f"Timestep {timestep}: [{current_phase} phase - round {round}] {player.name} {action.action_text()}"
        return message

    def create_location_message(self, record, env):
        if env.current_phase == "task":
            phase_info = "Task phase"
            instruction = TASK_PHASE_INSTRUCTION
        elif env.current_phase == "meeting":
            max_rounds = env.game_config["discussion_rounds"]
            round = max_rounds - env.discussion_rounds_left
            stage_names = {0: "Testimony", 1: "Accusation/Defense", 2: "Final Arguments"}
            stage_name = stage_names.get(round, f"Round {round}")
            phase_info = f"Meeting phase - {stage_name} ({round + 1}/{max_rounds})"
            instruction = "Players are discussing who to vote out. Listen carefully to what others say."
        message = f"Game Time: {env.timestep}/{env.game_config['max_timesteps']}\n"
        message += f"Current phase: {phase_info}\n"
        message += f"{instruction}\n"
        
        # Room context
        players_text = ", ".join(record["players"])
        bodies_text = ", ".join(record["bodies"])
        
        # ═══ STATE CHECK: Full player status block ═══
        # Forces the LLM to acknowledge every player's state before generating.
        # Includes death cause and timestep so agents can reason about timelines.
        all_living = [p.name for p in env.players if p.is_alive]
        dead_players = [p.name for p in env.players if not p.is_alive]
        total_players = len(env.players)
        
        message += f"[STATE CHECK] — {total_players} players in this game:\n"
        for p in env.players:
            if p.is_alive:
                # Show role ONLY to the player themselves (or to fellow Impostors)
                role_tag = ""
                if hasattr(record, '__contains__') and False:
                    pass  # Placeholder — role visibility handled below
                message += f"  {p.name}: ALIVE\n"
            else:
                cause = getattr(p, 'death_cause', 'KILLED')
                ts = getattr(p, 'death_timestep', '?')
                message += f"  {p.name}: DEAD ({cause} T{ts}) — CANNOT VOTE, CANNOT SPEAK, GHOST TASKS ONLY\n"
        message += f"⚠️ ONLY these {total_players} players exist. Do NOT reference any player not in this list.\n\n"
        
        message += "[[ROOM_CONTEXT_START]]\n"
        message += f"Current Location: {record['location']}\n"
        message += f"Living Players in {record['location']}: {players_text if players_text else 'None'}\n"
        if record["bodies"]:
             message += f"Dead Bodies in {record['location']}: {bodies_text}\n"
        
        # Dead players list (object permanence)
        if dead_players:
            message += f"Confirmed Dead Players (cannot be in any room): {', '.join(dead_players)}\n"
        
        # Task bar (grounding)
        task_pct = int(env.task_assignment.check_task_completion() * 100)
        message += f"Global Task Bar: {task_pct}% complete\n"
        
        message += "[[ROOM_CONTEXT_END]]\n\n"
        return message

    def route_location_info_message(self, env):
        for location in env.map.ship_map:
            # Only include living players in the location info
            players = env.map.get_players_in_room(location, include_new_deaths=False)
            player_names = [player.name for player in players]
            
            # Include bodies from the new dead_bodies system
            bodies_names = []
            for body in env.dead_bodies:
                if body["location"] == location and not body["reported"]:
                    bodies_names.append(f"{body['player_name']} (dead body)")
            
            record = {"location": location, "players": player_names, "bodies": bodies_names}
            
            # Send message to everyone in the room (living or ghost)
            # Ghosts should still know where they are!
            all_players_in_room = env.map.ship_map.nodes[location]["players"]
            for player in all_players_in_room:
                self.send_message(
                    player,
                    self.create_location_message(record, env),
                    info_type="location",
                )

    def route_real_time_message(self, env, record):
        player = record["player"]
        action = record["action"]
        location = action.current_location
        new_location = (
            action.new_location if hasattr(action, "new_location") else location
        )  # could be different from action.current_location if player moved or vented
        
        # Witness Privacy: If action is KILL, don't tell the victim.
        is_kill = action.name == "KILL"
        victim = getattr(action, 'other_player', None) if is_kill else None

        is_vent = action.name == "VENT"

        for other_player in env.players:
            if other_player == player:
                continue
            
            # Victim shouldn't "hear" the system announce their killer
            if is_kill and other_player == victim:
                continue

            if other_player.location == location or other_player.location == new_location:
                msg = self.create_action_message(record)
                # Tag KILL/VENT witnesses with certainty marker
                if (is_kill or is_vent) and other_player.location == location:
                    msg = f"[CONFIRMED EYEWITNESS] {msg} -- You SAW this happen. This is 100% proof, NOT a theory."
                # Tag VISUAL TASK completion — proves the player is a Crewmate
                if action.name == "COMPLETE TASK" and getattr(action, 'task', None) and getattr(action.task, 'is_visual', False):
                    if action.task.check_completion():
                        msg = f"[VISUAL TASK CONFIRMED] {msg} -- You SAW them complete a visual task. This PROVES they are a Crewmate."
                self.send_message(
                    other_player, msg, info_type="action"
                )
                # ═══ STRUCTURED MEMORY: Route into MemoryState ═══
                other_player.memory.update_memory(
                    msg, env.timestep, other_player.location)
