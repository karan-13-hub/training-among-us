"""
UPDATED notebook cells for test_pipeline.ipynb

These two blocks fix the RL analysis cell and the rollout display cell.
The core fix: use state snapshots embedded in activity_log records
instead of reading post-game object state.

═══════════════════════════════════════════════════════════════════
CELL 1: RL Analysis (replaces the analysis cell that builds `df`)
═══════════════════════════════════════════════════════════════════
"""

# ── RL Analysis (FIXED: uses embedded state snapshots) ────────
from amongagents.agent.actor import ActorModule
from amongagents.agent.rewards import RewardEngine, CriticModule
from collections import defaultdict
import pandas as pd

WINNER_MAP = {1: "Impostor (kills)", 2: "Crewmate (voting)",
              3: "Crewmate (tasks)", 4: "Impostor (sabotage)"}

def normalize_action(raw):
    """Normalize action name for reward engine."""
    raw = str(raw).upper().strip()
    for prefix in ["COMPLETE FAKE TASK", "COMPLETE TASK", "MOVE",
                    "CALL MEETING", "KILL", "VENT", "SABOTAGE",
                    "REPORT DEAD BODY", "VOTE", "SPEAK", "FIX SABOTAGE",
                    "VIEW MONITOR", "SKIP"]:
        if raw.startswith(prefix):
            return prefix.replace(" ", "_")
    return raw.split()[0].replace(" ", "_") if raw else "UNKNOWN"


def build_actor_observations(activity_log):
    """Build observation events from activity log for ActorModule."""
    observations_by_timestep = defaultdict(list)
    for record in activity_log:
        t = record["timestep"]
        player = record["player"]
        action = record["action"]
        raw_name = getattr(action, "name", str(action))

        # Extract witnesses from state snapshot
        obs = {
            "subject": player.name,
            "action": raw_name,
            "location": record.get("state", {}).get("player_location",
                        getattr(player, "location", "Unknown")),
        }
        observations_by_timestep[t].append(obs)
    return observations_by_timestep


# ── Analyze all games ─────────────────────────────────────────
critic = CriticModule()
reward_engine = RewardEngine()

all_rows = []
all_belief_snapshots = []

for game_idx, (winner_code, game) in enumerate(game_results, 1):
    print(f"\n═══ Analyzing Game {game_idx} ═══")

    # Build actor modules
    actors = {}
    for p in game.players:
        actors[p.name] = ActorModule(player=p, all_players=game.players)

    obs_by_t = build_actor_observations(game.activity_log)

    ejected_roles = []
    prev_state = None

    timesteps = sorted(set(r["timestep"] for r in game.activity_log))

    for t in timesteps:
        actions_at_t = [r for r in game.activity_log if r["timestep"] == t]

        # ── Use EMBEDDED snapshot from the first record at this timestep ──
        snap = actions_at_t[0].get("state", {})
        living_crew = snap.get("living_crew",
                     sum(1 for p in game.players
                         if p.identity == "Crewmate" and p.is_alive))
        living_imps = snap.get("living_imps",
                     sum(1 for p in game.players
                         if p.identity == "Impostor" and p.is_alive))
        task_pct = snap.get("task_pct",
                   game.task_assignment.check_task_completion() * 100.0)
        sabotage = snap.get("sabotage_active",
                   bool(getattr(game, 'active_sabotages', {})))

        curr_state = {
            "living_crewmates": living_crew,
            "living_impostors": living_imps,
            "task_completion_pct": task_pct,
            "sabotage_active": sabotage,
            "ejected_roles": list(ejected_roles),
            "winner": None,
        }

        # On final timestep, set winner
        if t == timesteps[-1]:
            winner_team = 'Crewmate' if winner_code in (2, 3) else 'Impostor'
            curr_state["winner"] = winner_team

        if prev_state is None:
            prev_state = curr_state.copy()

        # Critic evaluation
        v_crew = critic.evaluate_state_value(curr_state, "Crewmate")
        v_imp = critic.evaluate_state_value(curr_state, "Impostor")

        # Update actor beliefs
        observations = obs_by_t.get(t, [])
        for actor in actors.values():
            actor.update_beliefs(observations)

        # Process each action
        for record in actions_at_t:
            player = record["player"]
            raw_action = getattr(record["action"], "name", str(record["action"]))
            norm_action = normalize_action(raw_action)

            # ── Use snapshot for agent alive & location ──
            rec_snap = record.get("state", {})

            class _Agent:
                def __init__(self, role, alive, team):
                    self.role = role
                    self.team = team
                    self.alive = alive

            agent_proxy = _Agent(
                role=player.identity,
                alive=rec_snap.get("player_alive", player.is_alive),
                team=player.identity,
            )

            # Determine witnesses
            witnesses = []
            if hasattr(game, "map") and "player_location" in rec_snap:
                try:
                    room_players = game.map.get_players_in_room(
                        rec_snap["player_location"])
                    witnesses = [p.name for p in room_players
                                 if p != player and p.is_alive]
                except Exception:
                    pass

            reward = reward_engine.calculate_step_reward(
                agent_proxy,
                prev_state,
                curr_state,
                action_log={"action": norm_action, "witnesses": witnesses},
            )

            # Get beliefs
            actor = actors.get(player.name)
            if actor:
                if player.identity == 'Crewmate':
                    beliefs = dict(actor.suspicion_matrix)
                else:
                    beliefs = dict(actor.second_order_beliefs)
            else:
                beliefs = {}

            all_rows.append({
                "game": game_idx,
                "timestep": t,
                "player": player.name,
                "role": player.identity,
                "action": raw_action,
                "norm_action": norm_action,
                "reward": reward,
                "v_crew": v_crew,
                "v_imp": v_imp,
                "location": rec_snap.get("player_location",
                            getattr(player, "location", "?")),
                "alive": rec_snap.get("player_alive", player.is_alive),
                "living_crew": living_crew,
                "living_imps": living_imps,
                "task_pct": task_pct,
            })

            # Belief snapshots for heatmap
            for target, suspicion in beliefs.items():
                all_belief_snapshots.append({
                    "game": game_idx,
                    "timestep": t,
                    "observer": player.name,
                    "observer_role": player.identity,
                    "target": target,
                    "suspicion": suspicion,
                })

        prev_state = curr_state.copy()

    print(f"  {len(actions_at_t)} actions at final timestep, "
          f"{len([r for r in all_rows if r['game'] == game_idx])} total rows")

df = pd.DataFrame(all_rows)
beliefs_df = pd.DataFrame(all_belief_snapshots)

print(f"\n═══ Analysis Complete ═══")
print(f"Total rows: {len(df)}  Belief snapshots: {len(beliefs_df)}")
print(f"\nReward summary by role:")
for role in ['Crewmate', 'Impostor']:
    rdf = df[df['role'] == role]
    if len(rdf) > 0:
        print(f"  {role}: Σr={rdf['reward'].sum():+.2f}  mean={rdf['reward'].mean():+.3f}  actions={len(rdf)}")


"""
═══════════════════════════════════════════════════════════════════
CELL 2: Rollout Display (replaces the rollout display cell)
═══════════════════════════════════════════════════════════════════
"""

# ── Complete Game Rollout with RL Metrics ──────────────────────
# Uses embedded state snapshots for accurate per-timestep display.

INSPECT_GAME = 1  # Change this to inspect a different game (1-5)

winner_code, game = game_results[INSPECT_GAME - 1]

# Build identity map
identity_map = {p.name: p.identity for p in game.players}

# Build task objects (for accurate completion tracking)
task_objects = {}
for p in game.players:
    if hasattr(p, 'tasks') and len(p.tasks) > 0:
        task_objects[p.name] = list(p.tasks)
    else:
        task_objects[p.name] = []

task_assignments = {
    name: [str(t) for t in tasks]
    for name, tasks in task_objects.items()
}

# Get RL data for this game
game_df = df[df['game'] == INSPECT_GAME].copy()
game_beliefs = beliefs_df[beliefs_df['game'] == INSPECT_GAME].copy() if len(beliefs_df) > 0 else pd.DataFrame()

# Find impostor(s)
impostor_names = {p.name for p in game.players if p.identity == 'Impostor'}

# Header
print(f'Game: {INSPECT_GAME}')
print(f'Players:')
for name, identity in identity_map.items():
    tasks = task_assignments.get(name, [])
    tag = ' [IMP]' if identity == 'Impostor' else ''
    print(f'  {name:25s} {identity}{tag}   Tasks: {tasks}')
print(f'Total logged actions: {len(game.activity_log)}')
print(f'RL analysis rows: {len(game_df)}')
print('=' * 100)

# Group activity_log by timestep
log_by_t = defaultdict(list)
for record in game.activity_log:
    log_by_t[record['timestep']].append(record)

timesteps = sorted(log_by_t.keys())

for t in timesteps:
    records = log_by_t[t]
    phase = records[0].get('phase', 'task')

    # Detect meeting sub-phase
    actions_in_step = [str(r['action']) for r in records]
    has_speak = any('SPEAK' in a for a in actions_in_step)
    has_vote = any('VOTE' in a for a in actions_in_step)

    if phase == 'meeting' or has_speak or has_vote:
        phase_label = 'MEETING'
    else:
        phase_label = 'TASK'

    # Get RL metrics from DataFrame
    t_df = game_df[game_df['timestep'] == t]
    if len(t_df) > 0:
        v_crew = t_df.iloc[0]['v_crew']
        v_imp = t_df.iloc[0]['v_imp']
        task_pct = t_df.iloc[0]['task_pct']
        living_crew = int(t_df.iloc[0]['living_crew'])
        living_imps = int(t_df.iloc[0]['living_imps'])
    else:
        v_crew = v_imp = task_pct = 0.0
        living_crew = living_imps = 0

    print(f"\n{'━' * 100}")
    print(f'  TIMESTEP {t}  [{phase_label} PHASE]  │  '
          f'V(Crew)={v_crew:.3f}  V(Imp)={v_imp:.3f}  │  '
          f'Alive: {living_crew}C/{living_imps}I  │  '
          f'Tasks: {task_pct:.0f}%')
    print(f"{'━' * 100}")

    if phase_label == 'MEETING':
        # Separate speak, vote, and other actions
        speak_records = [r for r in records if 'SPEAK' in str(r['action'])]
        vote_records = [r for r in records if 'VOTE' in str(r['action'])]
        other_records = [r for r in records
                         if r not in speak_records and r not in vote_records]

        for record in other_records:
            player = record['player']
            name = player.name
            identity = identity_map.get(name, '?')
            action = str(record['action'])
            tag = ' [IMP]' if identity == 'Impostor' else ''

            rl_row = t_df[t_df['player'] == name]
            reward_str = ''
            if len(rl_row) > 0:
                reward_str = f'  r={rl_row.iloc[0]["reward"]:+.2f}'
            print(f'    {name}{tag}:  {action}{reward_str}')

        if speak_records:
            players_in_game = len(identity_map)
            num_rounds = max(1, len(speak_records) // players_in_game) if players_in_game > 0 else 1

            for rnd in range(num_rounds):
                print(f"\n  >>> Discussion Round {rnd + 1} {'─' * 50}")
                start = rnd * players_in_game
                end = start + players_in_game
                for record in speak_records[start:end]:
                    player = record['player']
                    name = player.name
                    identity = identity_map.get(name, '?')
                    action = str(record['action'])
                    tag = ' [IMP]' if identity == 'Impostor' else ''

                    rl_row = t_df[t_df['player'] == name]
                    reward_str = ''
                    sus_str = ''
                    if len(rl_row) > 0:
                        reward_str = f'r={rl_row.iloc[0]["reward"]:+.2f}'

                    if identity == 'Crewmate' and len(game_beliefs) > 0:
                        sus_rows = game_beliefs[
                            (game_beliefs['timestep'] == t) &
                            (game_beliefs['observer'] == name) &
                            (game_beliefs['target'].isin(impostor_names))
                        ]
                        if len(sus_rows) > 0:
                            sus_val = sus_rows.iloc[0]['suspicion']
                            sus_str = f'sus(imp)={sus_val:.2f}'

                    metrics = '  '.join(filter(None, [reward_str, sus_str]))
                    metrics_display = f'  [{metrics}]' if metrics else ''

                    print(f'    {name}{tag}:{metrics_display}')
                    speech = action
                    if speech.startswith('SPEAK:'):
                        speech = speech[6:].strip()
                    elif speech.startswith('SPEAK '):
                        speech = speech[6:].strip()
                    print(f'      "{speech}"')

        if vote_records:
            print(f"\n  >>> Voting {'─' * 55}")
            vote_counts = {}
            skip_count = 0
            seen_voters = set()
            for record in vote_records:
                player = record['player']
                name = player.name
                if name in seen_voters:
                    continue
                seen_voters.add(name)
                identity = identity_map.get(name, '?')
                action = str(record['action'])
                tag = ' [IMP]' if identity == 'Impostor' else ''
                print(f'    {name}{tag}:  {action}')

                if 'SKIP' in action.upper():
                    skip_count += 1
                elif 'VOTE' in action.upper():
                    target = action.replace('VOTE', '').strip()
                    if target:
                        vote_counts[target] = vote_counts.get(target, 0) + 1

            print(f"\n  >>> Vote Result {'─' * 49}")
            if vote_counts:
                max_v = max(vote_counts.values())
                top_targets = [t for t, v in vote_counts.items() if v == max_v]
                for target, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
                    s = 's' if count != 1 else ''
                    print(f'    {target}: {count} vote{s}')
                if skip_count > 0:
                    s = 's' if skip_count != 1 else ''
                    print(f'    SKIP: {skip_count} vote{s}')
                print(f'    ' + '-' * 34)
                if len(top_targets) == 1 and max_v > skip_count:
                    ejected = top_targets[0]
                    ident = identity_map.get(ejected, '?')
                    reveal = 'an Impostor' if ident == 'Impostor' else 'NOT an Impostor'
                    print(f'    Result: {ejected} was EJECTED ({max_v} votes). They were {reveal}.')
                elif skip_count >= max_v:
                    print(f'    Result: No one was ejected (SKIP won).')
                else:
                    print(f'    Result: TIE — no one was ejected.')
            elif skip_count > 0:
                print(f'    Result: No one was ejected (all {skip_count} votes were SKIP).')

    else:
        # TASK PHASE
        for record in records:
            player = record['player']
            name = player.name
            identity = identity_map.get(name, '?')
            action = str(record['action'])
            tag = ' [IMP]' if identity == 'Impostor' else ''

            # Use snapshot location instead of post-game location
            rec_snap = record.get('state', {})
            location = rec_snap.get('player_location', getattr(player, 'location', '?'))

            rl_row = t_df[t_df['player'] == name]
            reward_str = ''
            sus_str = ''
            if len(rl_row) > 0:
                r = rl_row.iloc[0]
                reward_str = f'r={r["reward"]:+.2f}'

            if identity == 'Crewmate' and len(game_beliefs) > 0:
                sus_rows = game_beliefs[
                    (game_beliefs['timestep'] == t) &
                    (game_beliefs['observer'] == name) &
                    (game_beliefs['target'].isin(impostor_names))
                ]
                if len(sus_rows) > 0:
                    sus_val = sus_rows.iloc[0]['suspicion']
                    sus_str = f'sus(imp)={sus_val:.2f}'
            elif identity == 'Impostor' and len(game_beliefs) > 0:
                threat_rows = game_beliefs[
                    (game_beliefs['timestep'] == t) &
                    (game_beliefs['observer'] == name)
                ]
                if len(threat_rows) > 0:
                    max_threat = threat_rows['suspicion'].max()
                    sus_str = f'max_threat={max_threat:.2f}'

            metrics = '  '.join(filter(None, [reward_str, sus_str]))
            metrics_display = f'  [{metrics}]' if metrics else ''

            print(f"\n  {name}{tag}  @{location}{metrics_display}")
            print(f'    Action:  {action}')

        # Task status summary (using actual task objects)
        print(f"\n  {'·' * 100}")
        print(f'  📋 Task Status after Timestep {t}:')
        for pname in identity_map:
            tasks = task_objects.get(pname, [])
            identity = identity_map[pname]
            tag = ' [IMP]' if identity == 'Impostor' else ''
            done_list = [str(tk) for tk in tasks if tk.check_completion()]
            remain_list = [str(tk) for tk in tasks if not tk.check_completion()]
            done_str = ', '.join(done_list) if done_list else '—'
            remain_str = ', '.join(remain_list) if remain_list else '— (all done!)'
            print(f'    {pname}{tag}:  ✅ {done_str}  |  ⏳ {remain_str}')
        print(f"  {'·' * 100}")

# ── Final game result ──
print(f"\n{'═' * 100}")
winner_team = 'Crewmate' if winner_code in (2, 3) else 'Impostor'
print(f'  GAME RESULT: {WINNER_MAP.get(winner_code, winner_code)}')
print(f'  Winner: {winner_team}  │  Timesteps: {game.timestep}')

# ── RL Summary for this game ──
g_rl = game_df
if len(g_rl) > 0:
    print(f"\n  RL Metrics Summary:")
    for role in ['Crewmate', 'Impostor']:
        rdf = g_rl[g_rl['role'] == role]
        if len(rdf) > 0:
            print(f'    {role}: Σr={rdf["reward"].sum():+.2f}  '
                  f'mean_r={rdf["reward"].mean():+.3f}  '
                  f'actions={len(rdf)}')
print(f"{'═' * 100}")
