task_config = {
    # Short tasks: 2 turns (up from 1) — creates vulnerability windows for Impostor
    "Download Data": {"duration": 2, "task_type": "short"},
    "Accept Diverted Power": {"duration": 2, "task_type": "short"},
    "Chart Course": {"duration": 2, "task_type": "short"},
    "Stabilize Steering": {"duration": 2, "task_type": "short"},
    "Clean O2 Filter": {"duration": 2, "task_type": "short"},
    "Prime Shields": {"duration": 2, "task_type": "short"},
    "Upload Data": {"duration": 2, "task_type": "short"},
    "Calibrate Distributor": {"duration": 2, "task_type": "short"},
    "Divert Power": {"duration": 2, "task_type": "short"},
    "Unlock Manifolds": {"duration": 2, "task_type": "short"},
    # Visual tasks: completing these produces visible proof → clears the player
    "Submit Scan": {"duration": 2, "task_type": "short", "visual": True},
    # Long tasks: 3 turns (up from 2) — high-risk commitment for Crewmates
    "Empty Garbage": {"duration": 3, "task_type": "long", "visual": True},
    "Clear Asteroids": {"duration": 3, "task_type": "long", "visual": True},
    "Empty Chute": {"duration": 3, "task_type": "long"},
    "Align Engine Output": {"duration": 3, "task_type": "long"},
    "Fuel Engines": {"duration": 3, "task_type": "long"},
    "Start Reactor": {"duration": 3, "task_type": "long"},
    "Inspect Sample": {"duration": 3, "task_type": "long"},
    # Common tasks: 2 turns (up from 1)
    "Fix Wiring": {"duration": 2, "task_type": "common"},
    "Swipe Card": {"duration": 2, "task_type": "common"},
}
