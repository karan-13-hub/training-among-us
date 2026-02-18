import re


class Action:
    def __init__(self, name, current_location=None):
        self.name = name
        self.current_location = current_location

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def execute(self, env, player):
        return

    def action_text(self):
        return str(self)

    @staticmethod
    def can_execute_actions(env, player):
        return []


class MoveTo(Action):
    def __init__(self, current_location, new_location):
        super().__init__("MOVE", current_location=current_location)
        self.new_location = new_location

    def __repr__(self):
        return f"{self.name} from {self.current_location} to {self.new_location}"

    def execute(self, env, player):
        super().execute(env, player)
        # Task progress is PERSISTENT: if a player leaves mid-task (e.g., forced
        # by sabotage or body report), they keep their progress and can finish
        # it later by returning to the room. The Commitment Flag in check_actions()
        # normally prevents voluntary movement during in-progress tasks.
        player.location = self.new_location

    @staticmethod
    def can_execute_actions(env, player):
        if env.current_phase == "task":
            if not player.is_alive:
                # Ghosts have "no-clip" movement -- can move to ANY room in one turn
                all_rooms = list(env.map.ship_map.nodes)
                return [MoveTo(player.location, room) for room in all_rooms if room != player.location]
            new_locations = env.map.get_adjacent_rooms(player.location)
            return [MoveTo(player.location, location) for location in new_locations]
        else:
            return []


class Vent(MoveTo):
    def __init__(self, current_location, new_location):
        super().__init__(current_location, new_location)
        self.name = "VENT"

    @staticmethod
    def can_execute_actions(env, player):
        if env.current_phase == "task":
            new_locations = env.map.get_adjacent_rooms_vent(player.location)
            return [Vent(player.location, location) for location in new_locations]
        else:
            return []


class CallMeeting(Action):
    def __init__(self, current_location):
        super().__init__("CALL MEETING", current_location=current_location)

    def __repr__(self):
        if self.current_location == "Cafeteria":
            return f"{self.name} using the emergency button at {self.current_location}"
        else:
            return f"REPORT DEAD BODY at {self.current_location}"

    def execute(self, env, player):
        super().execute(env, player)
        env.current_phase = "meeting"
        env.button_num += 1
        # Mark all bodies as reported
        for body in env.dead_bodies:
            body["reported"] = True
        for p in env.players:
            if not p.is_alive and not p.reported_death:
                p.reported_death = True

    @staticmethod
    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        if env.current_phase == "task":
            current_location = player.location
            
            # Emergency button at Cafeteria: requires at least 1 completed task
            # to prevent "Timestep 0 Meeting" momentum killers.
            # Body reports are ALWAYS allowed regardless of task count.
            if (
                current_location == "Cafeteria"
                and env.button_num < env.game_config["max_num_buttons"]
            ):
                # Check if player has completed at least 1 task (meeting cooldown)
                has_completed_task = any(t.check_completion() for t in player.tasks)
                has_witnessed_crime = player.has_witnessed_crime()
                if has_completed_task or has_witnessed_crime:
                    return [CallMeeting(current_location=current_location)]
                # Otherwise, emergency button is locked -- must complete a task first
            
            # Body reports: ALWAYS allowed when an unreported body is in the room
            for body in env.dead_bodies:
                if body["location"] == current_location and not body["reported"]:
                    return [CallMeeting(current_location=current_location)]
        return []


class Vote(Action):
    def __init__(self, current_location, other_player):
        super().__init__("VOTE", current_location=current_location)
        self.other_player = other_player

    def __repr__(self):
        return f"{self.name} {self.other_player.name}"

    def execute(self, env, player):
        super().execute(env, player)
        env.vote_info_one_round[player.name] = self.other_player.name
        env.votes[self.other_player] = env.votes.get(self.other_player, 0) + 1

    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        if env.current_phase == "meeting" and env.discussion_rounds_left == 0:
            alive_players_excluding_self = [
                p for p in env.players if p.is_alive and p != player
            ]
            return [
                Vote(player.location, other_player)
                for other_player in alive_players_excluding_self
            ]
        else:
            return []


class Speak(Action):
    def __init__(self, current_location):
        super().__init__("SPEAK", current_location=current_location)
        self.message = "..."

    def provide_message(self, message):
        self.message = message

    def __repr__(self):
        msg = getattr(self, 'message', '...')
        return f"{self.name}: {msg}"

    def execute(self, env, player):
        super().execute(env, player)
        # Broadcast speech to ALL players so they can respond coherently
        round_num = env.game_config["discussion_rounds"] - env.discussion_rounds_left
        message = f"[Discussion Round {round_num}] {player.name} said: \"{self.message}\""
        for other_player in env.players:
            if other_player != player and other_player.is_alive:
                other_player.observation_history.append(message)
                # ═══ Route into MemoryState as HEARSAY ═══
                other_player.memory.add_hearsay(
                    env.timestep, player.name, message)

        # ═══ CONSISTENCY CHECK: Record the speaker's own claim ═══
        # So the agent can see its own previous statements in later rounds
        # and avoid self-contradiction.
        player.memory.record_own_statement(env.timestep, self.message)

    def can_execute_actions(env, player):
        if not player.is_alive:
            return []  # Ghosts cannot speak in meetings
        if env.current_phase == "meeting" and env.discussion_rounds_left == 0:
            return []
        # ADDITION BY ME
        if env.current_phase == "task":
            return []
        
        return [Speak(current_location=player.location)]


class ViewMonitor(Action):
    def __init__(self, current_location):
        super().__init__("ViewMonitor", current_location=current_location)

    def __repr__(self):
        return f"VIEW MONITOR"

    def execute(self, env, player, choose_location):
        super().execute(env, player)
        message = (
            "Monitor Record: {" + "Location: " + choose_location + ", Observation: {"
        )
        if len(env.check_monitor(choose_location)) == 0:
            message += "No one here"
        else:
            for agent in env.players:
                if agent in env.check_monitor(choose_location):
                    message += "(" + agent.name + "): "

                    pattern = r"MOVE from ([\w\s]+) to ([\w\s]+)"
                    action = str(env.camera_record[agent.name])
                    match = re.match(pattern, action)
                    if match:
                        start_location = match.group(1)
                        end_location = match.group(2)
                        action = "enter " + end_location
                    message += action + ", "

                else:
                    pattern = r"MOVE from ([\w\s]+) to ([\w\s]+)"
                    action = str(env.camera_record[agent.name])
                    print("action", action)
                    match = re.match(pattern, action)
                    if match:
                        start_location = match.group(1)
                        end_location = match.group(2)
                        print("start_location", start_location)
                        print("choose_location", choose_location)
                        if start_location == choose_location:
                            message += "(" + agent.name + "): "
                            action = "leave " + start_location
                            message += action + ", "

        message += "}}"
        print(message)
        player.observation_history.append(message)
        # TODO: Implement this

    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        available_tasks = []
        if player.location == "Security":
            return [ViewMonitor("Security")]
        else:
            return []


class CompleteTask(Action):
    def __init__(self, current_location, task):
        super().__init__("COMPLETE TASK", current_location=current_location)
        self.task = task

    def __repr__(self):
        # Multi-turn task progress display
        if self.task.max_duration > 1:
            if self.task.duration == self.task.max_duration:
                # Fresh task — not started yet
                return f"{self.name} - {self.task.name} (requires {self.task.duration} turns — you MUST stay in this room!)"
            elif self.task.duration == 1:
                # Last turn — will complete
                return f"{self.name} - {self.task.name} (FINAL TURN — completes this turn!)"
            else:
                # In progress
                return f"{self.name} - {self.task.name} ({self.task.duration} turns remaining — LOCKED, progress saved)"
        return f"{self.name} - {self.task.name}"

    def action_text(self):
        if getattr(self.task, 'is_visual', False) and self.task.check_completion():
            return f"Visibly completing task {self.task.name} (VISUAL CONFIRMATION)"
        if self.task.max_duration > 1 and self.task.duration > 0:
            return f"Working on task {self.task.name} ({self.task.duration} turn{'s' if self.task.duration > 1 else ''} left)"
        return f"Seemingly doing task"

    def execute(self, env, player):
        super().execute(env, player)
        # Guard: don't complete an already-finished task (prevents task duplication bug)
        if self.task.check_completion():
            return
        self.task.do_task()

    def can_execute_actions(env, player):
        # Ghosts CAN complete tasks (contributes to Crewmate task bar win condition)
        available_tasks = []
        if env.current_phase == "task":
            current_location = player.location

            for task in player.tasks:
                if task.location == current_location and not task.check_completion():
                    available_tasks.append(CompleteTask(current_location, task))

        return available_tasks


class Sabotage(Action):
    SABOTAGE_TYPES = {
        "Electrical": "LIGHTS",       # Reduces crewmate observation range
        "O2": "OXYGEN",               # Forces crewmates to O2 to fix
        "Reactor": "REACTOR",         # Forces crewmates to Reactor to fix
        "Communications": "COMMS",    # Disables crewmate task list visibility
    }
    
    def __init__(self, current_location, sabotage_type=None):
        super().__init__("SABOTAGE", current_location=current_location)
        self.sabotage_type = sabotage_type or "LIGHTS"
    
    def __repr__(self):
        return f"{self.name} {self.sabotage_type} (from {self.current_location})"
    
    def action_text(self):
        return f"Triggered {self.sabotage_type} sabotage"

    # Duration (in turns) that each sabotage type stays active
    SABOTAGE_DURATIONS = {
        "LIGHTS": 3,
        "COMMS": 3,
        "OXYGEN": 4,   # Critical — longer duration to force response
        "REACTOR": 4,
    }
    
    def execute(self, env, player):
        super().execute(env, player)
        # Record the sabotage event so all players are notified
        sabotage_msg = f"[SABOTAGE] {self.sabotage_type} has been sabotaged! "
        if self.sabotage_type == "LIGHTS":
            sabotage_msg += "⚡ Vision is reduced — you CANNOT identify other players until lights are restored!"
        elif self.sabotage_type == "OXYGEN":
            sabotage_msg += "🔴 Oxygen is depleting! Crewmates must go to O2 to fix it or everyone dies!"
        elif self.sabotage_type == "REACTOR":
            sabotage_msg += "🔴 Reactor is melting down! Crewmates must go to Reactor to fix it or everyone dies!"
        elif self.sabotage_type == "COMMS":
            sabotage_msg += "📡 Communications are down — task list and task bar are HIDDEN until fixed!"
        
        # Broadcast to all living players (including the impostor for consistency)
        for p in env.players:
            if p.is_alive:
                p.observation_history.append(sabotage_msg)
                # ═══ Route sabotage alert into MemoryState ═══
                p.memory.update_memory(sabotage_msg, env.timestep,
                                       p.location)
        
        # ═══ CRISIS DISPATCH: tag nearest 2 Crewmates ═══
        env.crisis_dispatch(self.sabotage_type)

        # Activate the sabotage with a timer (mechanical effect)
        if not hasattr(env, 'active_sabotages'):
            env.active_sabotages = {}
        duration = Sabotage.SABOTAGE_DURATIONS.get(self.sabotage_type, 3)
        env.active_sabotages[self.sabotage_type] = duration
        
        # Set a cooldown so sabotage can't be spammed
        if not hasattr(env, 'sabotage_cooldown'):
            env.sabotage_cooldown = 0
        env.sabotage_cooldown = 3  # Can't sabotage again for 3 timesteps

    @staticmethod
    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        if env.current_phase != "task":
            return []
        # Sabotage cooldown check
        if hasattr(env, 'sabotage_cooldown') and env.sabotage_cooldown > 0:
            return []
        # Only impostors can sabotage (defense-in-depth, already filtered by SPECIAL_ACTIONS)
        if player.identity != "Impostor":
            return []
        # Return available sabotage types based on map rooms
        sabotage_actions = []
        for room, sab_type in Sabotage.SABOTAGE_TYPES.items():
            sabotage_actions.append(Sabotage(current_location=player.location, sabotage_type=sab_type))
        return sabotage_actions


class FixSabotage(Action):
    """Action to repair a critical sabotage (O2 or Reactor).
    
    Only available to Crewmates when a critical sabotage is active AND the
    player is in the correct room (O2 for OXYGEN, Reactor for REACTOR).
    Clearing the sabotage removes it from the active_sabotages dict.
    """
    # Map sabotage type → required room to fix it
    FIX_LOCATIONS = {
        "OXYGEN": "O2",
        "REACTOR": "Reactor",
        "LIGHTS": "Electrical",
        "COMMS": "Communications",
    }
    
    def __init__(self, current_location, sabotage_type):
        super().__init__("FIX SABOTAGE", current_location=current_location)
        self.sabotage_type = sabotage_type
    
    def __repr__(self):
        return f"{self.name} - Repair {self.sabotage_type} (at {self.current_location})"
    
    def action_text(self):
        return f"Fixed {self.sabotage_type} sabotage"
    
    def execute(self, env, player):
        super().execute(env, player)
        # Remove the sabotage from active state
        if hasattr(env, 'active_sabotages') and self.sabotage_type in env.active_sabotages:
            del env.active_sabotages[self.sabotage_type]
            # Notify all players
            fix_msg = f"[SYSTEM] {player.name} has repaired the {self.sabotage_type} sabotage! Crisis averted."
            for p in env.players:
                if p.is_alive:
                    p.observation_history.append(fix_msg)
            # Update all players' sabotage state immediately
            active_sab_set = set(env.active_sabotages.keys())
            for p in env.players:
                p.active_sabotages = active_sab_set
    
    @staticmethod
    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        if env.current_phase != "task":
            return []
        if not hasattr(env, 'active_sabotages'):
            return []
        # Anyone (Crewmate or Impostor) can fix a sabotage if they're in the right room
        fix_actions = []
        for sab_type, timer in env.active_sabotages.items():
            required_room = FixSabotage.FIX_LOCATIONS.get(sab_type)
            if required_room and player.location == required_room:
                fix_actions.append(FixSabotage(current_location=player.location, sabotage_type=sab_type))
        return fix_actions


class Kill(Action):
    def __init__(self, current_location, other_player):
        super().__init__("KILL", current_location=current_location)
        self.other_player = other_player

    def __repr__(self):
        return f"{self.name} {self.other_player.name}"

    def execute(self, env, player):
        super().execute(env, player)
        if player.kill_cooldown > 0:
            msg = f"[System] Action FAILED: Kill is on cooldown for {player.kill_cooldown} more turns."
            player.receive(msg, "action")
            return

        # --- Physics Validation (Defense-in-Depth) ---
        # Even if can_execute_actions was correct, re-validate before committing
        # the irreversible state change. This catches race conditions and parser bugs.
        if not self.other_player.is_alive:
            msg = f"[System] Action FAILED: {self.other_player.name} is already dead."
            player.receive(msg, "action")
            return
        if player.location != self.other_player.location:
            msg = f"[System] Action FAILED: {self.other_player.name} is not in {player.location}."
            player.receive(msg, "action")
            return

        self.other_player.is_alive = False
        self.other_player.death_timestep = env.timestep
        self.other_player.death_cause = "KILLED"
        player.kill_cooldown = env.game_config["kill_cooldown"]
        # Leave a body at the kill site
        env.dead_bodies.append({
            "location": self.current_location,
            "player_name": self.other_player.name,
            "reported": False
        })

    @staticmethod
    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        if env.current_phase == "task":
            current_location = player.location
            other_players = env.map.get_players_in_room(current_location)
            # Filter: must be opposite identity AND alive (defense-in-depth)
            other_players = [
                p for p in other_players
                if p.identity != player.identity and p.is_alive
            ]
            return [
                Kill(current_location=current_location, other_player=p)
                for p in other_players
            ]
        else:
            return []


class CompleteFakeTask(CompleteTask):
    def __init__(self, current_location, task):
        super().__init__(current_location, task)
        self.name = "COMPLETE FAKE TASK"

    def __repr__(self):
        return f"{self.name} - {self.task.name}"

    def action_text(self):
        return f"Seemingly doing task"

    def execute(self, env, player):
        # CompleteTask.execute already calls do_task() — don't call it again
        super().execute(env, player)

    def can_execute_actions(env, player):
        if not player.is_alive:
            return []
        available_tasks = []
        if env.current_phase == "task":
            current_location = player.location
            for task in player.tasks:
                # Impostors CANNOT fake visual tasks (they can't produce
                # the visible animation, so observers would catch them)
                if task.location == current_location and not task.check_completion():
                    if not getattr(task, 'is_visual', False):
                        available_tasks.append(CompleteFakeTask(current_location, task))

        return available_tasks


COMMON_ACTIONS = [MoveTo, CallMeeting, Vote, Speak, ViewMonitor, FixSabotage]
CREWMATE_ACTIONS = [CompleteTask]
IMPOSTER_ACTIONS = [Sabotage, Vent, Kill, CompleteFakeTask]
