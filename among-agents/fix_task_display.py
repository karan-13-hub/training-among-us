#!/usr/bin/env python3
"""Patch test_pipeline.ipynb to fix the suspicion chart:
1. Fix action name mismatch in build_actor_observations (spaces → underscores)
2. Add markers to the suspicion chart so short lines are visible
"""
import json

NB_PATH = "test_pipeline.ipynb"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

patches_applied = 0

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])

    # ── Patch 1: Fix action name mapping in build_actor_observations ──
    if "def build_actor_observations(activity_log):" in src:
        old_obs_build = (
            '        obs = {\n'
            '            "subject": player.name,\n'
            '            "action": raw_name,\n'
            '            "location": record.get("state", {}).get("player_location",\n'
            '                        getattr(player, "location", "Unknown")),\n'
            '        }'
        )
        new_obs_build = (
            '        # Normalize action names to match ActorModule.update_beliefs expectations\n'
            '        # e.g. "COMPLETE TASK" → "COMPLETE_TASK", "COMPLETE FAKE TASK" → "FAKE_TASK"\n'
            '        belief_action = raw_name.replace(" ", "_")\n'
            '        if belief_action == "COMPLETE_FAKE_TASK":\n'
            '            belief_action = "FAKE_TASK"\n'
            '        obs = {\n'
            '            "subject": player.name,\n'
            '            "action": belief_action,\n'
            '            "location": record.get("state", {}).get("player_location",\n'
            '                        getattr(player, "location", "Unknown")),\n'
            '        }'
        )
        if old_obs_build in src:
            src = src.replace(old_obs_build, new_obs_build)
            lines = src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]]
            if lines[-1]:
                cell["source"].append(lines[-1])
            patches_applied += 1
            print("✅ Patch 1: Fixed action name normalization in build_actor_observations")
        else:
            print("⚠️  Patch 1: Could not find exact target in build_actor_observations")

    # ── Patch 2: Add markers to suspicion chart ──
    if "Crewmate Suspicion of Impostor Over Time" in src:
        old_chart = (
            "    fig_b = px.line(\n"
            "        crew_beliefs, x='timestep', y='suspicion',\n"
            "        color='observer',\n"
            "        title=f'Game {game_idx}: Crewmate Suspicion of Impostor Over Time',\n"
            "        template='plotly_dark',\n"
            "        labels={'suspicion': 'Suspicion (0=Safe, 1=Accused)', 'timestep': 'Timestep'},\n"
            "    )\n"
            "    fig_b.update_yaxes(range=[0, 1])\n"
            "    fig_b.show()"
        )
        new_chart = (
            "    fig_b = px.line(\n"
            "        crew_beliefs, x='timestep', y='suspicion',\n"
            "        color='observer',\n"
            "        markers=True,\n"
            "        title=f'Game {game_idx}: Crewmate Suspicion of Impostor Over Time',\n"
            "        template='plotly_dark',\n"
            "        labels={'suspicion': 'Suspicion (0=Safe, 1=Accused)', 'timestep': 'Timestep'},\n"
            "    )\n"
            "    fig_b.update_traces(line=dict(width=2), marker=dict(size=6))\n"
            "    fig_b.update_yaxes(range=[0, 1])\n"
            "    fig_b.show()"
        )
        if old_chart in src:
            src = src.replace(old_chart, new_chart)
            lines = src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]]
            if lines[-1]:
                cell["source"].append(lines[-1])
            cell["outputs"] = []  # Clear cached output
            patches_applied += 1
            print("✅ Patch 2: Added markers and thicker lines to suspicion chart")
        else:
            print("⚠️  Patch 2: Could not find exact chart target")

if patches_applied > 0:
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"\n✅ Applied {patches_applied} patches to {NB_PATH}.")
    print("   Re-run the RL Analysis cell and the Suspicion chart cell to see the fix.")
else:
    print("\n❌ No patches applied — targets not found.")
