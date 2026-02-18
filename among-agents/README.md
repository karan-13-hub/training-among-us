# AmongAgents: Reinforcement Learning for Strategic Social Deduction

An RL-augmented multi-agent simulation of Among Us, built on top of the [AmongAgents](https://github.com/SHD-Scientific/among-agents) framework. Agents are powered by LLMs (via vLLM / OpenRouter) and enhanced with a modular RL pipeline consisting of an **Actor**, **Critic**, **Reward Engine**, **Judge**, and a **Speaking Score** heuristic.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Actor — Theory of Mind Module](#actor--theory-of-mind-module)
- [Critic — State-Value Estimator](#critic--state-value-estimator)
- [Reward Engine](#reward-engine)
- [Judge — Hallucination Detection](#judge--hallucination-detection)
- [Speaking Score Heuristic](#speaking-score-heuristic)
- [Kill Risk Matrix](#kill-risk-matrix)
- [Meeting Role System](#meeting-role-system)
- [Prompt Architecture](#prompt-architecture)
- [Setup](#setup)

---

## Architecture Overview

The RL pipeline follows an **Actor–Critic** structure wired into a multi-agent game loop. At each timestep, every living agent performs the following cycle:

```
┌────────────────────────────────────────────────────────────────┐
│  Game State  ──►  Actor (ToM + LLM)  ──►  Action              │
│       │                                      │                 │
│       ▼                                      ▼                 │
│    Critic ──► V(s)                    Reward Engine ──► r(t)   │
│                                              │                 │
│                              Judge ◄─────────┘                 │
│                         (if SPEAK action)                      │
│                              │                                 │
│                     Speaking Score ──► Accept / Reject speech   │
└────────────────────────────────────────────────────────────────┘
```

| Module | File | Purpose |
|---|---|---|
| Actor | `agent/actor.py` | Theory of Mind belief updates + LLM action selection |
| Critic | `agent/critic.py` | Win probability V(s) ∈ [0, 1] for each team |
| Reward Engine | `agent/rewards.py` | Scalar reward r(t) per action with asymmetric tables |
| Judge | `agent/judge.py` | Hallucination detection via ground-truth comparison |
| LLMAgent | `agent/agent.py` | Full agent orchestration (2600+ lines) — meeting roles, speaking score, kill risk matrix, prompt composition |
| Prompts | `agent/neutral_prompts.py` | System prompts, meeting stage instructions, discussion roles |

---

## Actor — Theory of Mind Module

**File:** `agent/actor.py`

The Actor implements a **Theory of Mind (ToM)** layer that maintains mental-state matrices for each agent:

### Belief Matrices

| Role | Matrix | Interpretation |
|---|---|---|
| **Crewmate** | `suspicion_matrix` (First-Order) | "Who do I think is the Impostor?" — maps each player → suspicion ∈ [0.0, 1.0] |
| **Impostor** | `second_order_beliefs` (Second-Order) | "Who suspects ME?" — maps each player → threat ∈ [0.0, 1.0] |

Both matrices initialize at **0.5** (uniform/neutral).

### Deterministic Belief Update Rules

Beliefs are updated using **hard-coded rules** — no LLM is needed for this step. When an event is observed:

| Event Observed | Update | Rationale |
|---|---|---|
| KILL or VENT | Set to **1.0** (hard evidence) | Eyewitness certainty |
| SABOTAGE | Multiply current × **1.25** | Strong but indirect suspicion |
| FAKE_TASK | Multiply current × **1.10** | Weak suspicion signal |
| VISUAL_TASK | Multiply current × **0.90** | Soft clear — trust building |
| COMPLETE_TASK | Multiply current × **0.90** | Legitimate contribution |

All values are clamped to [0.0, 1.0].

**Crewmates** update their `suspicion_matrix` when they see _anyone_ perform these actions. **Impostors** update their `second_order_beliefs` only when they detect that _another player_ witnessed _them_ performing an incriminating action.

### ToM Overlay Prompts

The Actor constructs a composite prompt by appending a **ToM overlay** to the base system prompt from `neutral_prompts.py`:

- **Crewmate overlay**: Instructs the LLM to estimate suspicion scores for all players and use them to guide action selection (e.g., "High suspicion → observe from distance; Low suspicion → focus on tasks").
- **Impostor overlay**: Instructs the LLM to estimate threat scores and decide between elimination vs. framing strategies.

The LLM outputs a structured JSON including the belief matrix, a thought trace, and the chosen action.

---

## Critic — State-Value Estimator

**File:** `agent/critic.py`

The Critic produces **V(s, team) ∈ [0.0, 1.0]** — the estimated win probability for a given team from the current game state.

### Design Principles

1. **Central Critic**: Has access to complete global state during training (living counts, task %, ejections, sabotage status).
2. **Asymmetric Evaluation**: Separate heuristic subroutines for Crewmate vs. Impostor perspectives.
3. **Zero-Sum**: `V(Crewmate) + V(Impostor) ≈ 1.0` for any game state.
4. **Fake-Task Aware**: Only real completed tasks count toward task completion percentage.

### Heuristic Value Function (Crewmate)

The Crewmate value function combines three components:

```
V(crew) = 0.1 + task_factor + numbers_factor − sabotage_penalty
```

| Component | Formula | Range |
|---|---|---|
| `task_factor` | `(task_pct / 100) × 0.5` | 0.0 – 0.5 |
| `numbers_factor` | `((crew − imps) / (crew + imps)) × 0.4` | 0.0 – 0.4 |
| `sabotage_penalty` | `0.1 if sabotage active else 0.0` | 0.0 – 0.1 |

Terminal overrides: `V = 1.0` if all impostors ejected or tasks reach 100%; `V = 0.0` if crew ≤ impostors (impostor parity).

The Impostor value is simply `V(imp) = 1.0 − V(crew)`.

### LLM Critic (Future)

The current implementation uses the heuristic above as a placeholder. The architecture is designed for a drop-in replacement with an LLM-based critic or a learned neural critic.

---

## Reward Engine

**File:** `agent/rewards.py`

Calculates a **scalar reward r(t)** for every agent action. Rewards follow a strict priority hierarchy:

### Reward Priority Hierarchy

```
1. Terminal Rewards    → Game over this turn (overrides everything)
2. Social/Cognitive    → Hallucination, lie success/failure, vote strategy
3. Action Rewards      → Role-specific per-action values
```

### 1. Terminal Rewards

| Condition | Reward |
|---|---|
| Win + Alive | **+50.0** |
| Win + Dead (Martyr) | **+30.0** |
| Loss | **−20.0** |

### 2. Social & Cognitive Rewards

| Event | Role | Reward |
|---|---|---|
| Hallucination detected | Any | **−100.0** |
| Successful lie | Impostor | **+2.0** |
| Lie refuted | Impostor | **−5.0** |
| Voted for Impostor | Crewmate | **+5.0** |
| Voted for Crewmate | Crewmate | **−2.0** |
| Framed Crewmate (voted out) | Impostor | **+3.0** |
| Survived vote (not ejected) | Impostor | **+10.0** |

### 3. Action Rewards — Impostor

| Action | Witnesses | Reward | Rationale |
|---|---|---|---|
| KILL | 0 | **+15.0** (10 + 5 unseen bonus) | Clean kill |
| KILL | 1 | **+2.0** (10 − 8) | Risky but net positive |
| KILL | 2+ | **−6.0+** (10 − 8 × n) | Increasingly punished |
| VENT | 0 | **+1.0** | Safe mobility |
| VENT | 1+ | **−10.0** | Critical exposure |
| REPORT_BODY | — | **+3.0** | Self-report deception |
| FAKE_TASK | — | **+2.0** | Blending in |
| SABOTAGE | — | **+1.0** | Disruption |
| FIX_SABOTAGE | — | **+1.0** | Blending in |

### 3. Action Rewards — Crewmate

| Action | Condition | Reward |
|---|---|---|
| COMPLETE_TASK | Normal | **+2.0** |
| COMPLETE_TASK | Critical State | **+5.0** (endgame scaling) |
| FIX_SABOTAGE | — | **+3.0** |
| REPORT_BODY | — | **+2.0** |
| DIE | Normal | **−15.0** |
| DIE | Critical State | **−50.0** |

### Critical State Detection (Endgame Scaling)

A game enters **Critical State** when _either_:
- Living Crewmates ≤ 3
- Living Crewmates ≤ Living Impostors + 2

In this state, task completion and death rewards are amplified to reflect the heightened stakes.

---

## Judge — Hallucination Detection

**File:** `agent/judge.py`

The Judge is a **referee module** that detects when an agent "breaks reality" — making claims that contradict ground truth or their own prior statements.

### Asymmetric Validation Rules

| Role | Checked Against | Standard |
|---|---|---|
| **Crewmate** | Game Logs (ground truth) | **Strict Truth** — any factual mismatch is a hallucination |
| **Impostor** | Chat History (own prior statements) | **Self-Consistency** — contradicting yourself is a hallucination; lying about events is _allowed_ |

This asymmetry is fundamental: **Impostors are expected to lie** (part of the game), but they must maintain a **coherent narrative**. A Crewmate who claims to have been in Reactor when logs show they were in Admin is hallucinating. An Impostor who says "I was in Medbay" (when they were actually killing in Electrical) is playing the game correctly — but if they later say "I was in Admin the whole time," that's a self-contradiction.

### 3-Stage Pipeline

```
1. Fact Extraction   → Parse the agent's speech for checkable claims
2. Reality Check     → Compare claims against game logs (Crewmate) or chat history (Impostor)
3. LLM Judge Call    → A frozen LLM classifies the statement as hallucination or not
```

### Penalty

Hallucination carries a **−100.0 reward penalty** — the harshest single penalty in the system. This trains agents to stay grounded in their actual observations.

---

## Speaking Score Heuristic

**File:** `agent/agent.py` (methods: `_compute_valid_truths`, `_score_speech`)

A **post-generation validation system** that scores every SPEAK action on a 0–20+ scale before allowing it into the game. Negative total scores cause the speech to be **rejected and regenerated**.

### Stage 1: Reality Check (`_compute_valid_truths`)

Before scoring, the system pre-computes a **Line-of-Sight (LOS) truth table** from ground-truth data:

- **Rooms visited**: The set of rooms the agent physically occupied
- **Players seen per room**: Who was present when the agent was there
- **Witnessed crimes**: Kill or vent events with `[CONFIRMED EYEWITNESS]` tags
- **Impostor deception ledger**: Kill location, public alibi, kill victim (for Impostors only)

The agent can **ONLY** make claims about rooms in their `rooms_visited` set. Everything else is outside their epistemic boundary.

### Stage 2: Scoring Table (`_score_speech`)

| Category | Condition | Score | Description |
|---|---|---|---|
| **D. Hallucination** | X-Ray Vision | **−100** | Claiming observations in rooms never visited |
| | Meta-Gaming | **−50** | Referencing game internals ("verified presence log", "timestep 3") |
| | Self-Incrimination | **−50** | Impostor confessing to a kill or revealing kill location |
| | Spatial Non-Sequitur | **−20** | "I was in Room A, so you weren't in Room B" (invalid logic) |
| **A. Hard Evidence** | Kill Witness | **+20** | Referencing a witnessed kill (verified by LOS) |
| | Vent Witness | **+18** | Referencing a witnessed vent |
| | Hard Alibi | **+12** | "I was with [player] in [room]" — verified by presence log |
| | Path Contradiction | **+10** | Questioning impossible room-to-room travel |
| | Direct Defense | **+10** | Offering visual task proof |
| **B. Soft Evidence** | Task Logic | **+8** | Referencing task bar evidence |
| | Spatial Logic | **+8** | Distance/time impossibility argument |
| | Sighting | **+5** | Reporting seeing a player (verified) |
| **C. Noise/Fluff** | Uncertainty | **+2** | "I don't know" / "nothing suspicious" |
| | Skip Vote | **+1** | Suggesting to skip |
| | Agreement | **+1** | Echoing another player's point |

If a speech scores **negative**, it is rejected and the LLM is prompted to regenerate.

---

## Kill Risk Matrix

**File:** `agent/agent.py` (method: `_compute_kill_risk_matrix`)

For Impostor agents, a **per-target risk assessment** is computed before each action prompt. This gives the LLM tactical awareness about which kills are safe.

### Risk Score Formula

```
risk = min(1.0, witness_risk + 0.4 × exposure + escape_penalty)
```

| Factor | Formula | Interpretation |
|---|---|---|
| `witness_risk` | `min(1.0, (num_killable − 1) × 0.35)` | How many other players could witness the kill |
| `exposure` | `co_location_count / total_timesteps` | How often the target has already seen us (higher = riskier, target might report) |
| `escape_penalty` | `0.25 if no vent available, else 0.0` | No vent escape adds flat risk |

Targets are sorted by risk score (lowest = safest kills first) and injected into the Impostor's action prompt as structured intelligence.

---

## Meeting Role System

**File:** `agent/agent.py` (method: `_assign_meeting_role`), `agent/neutral_prompts.py`

During meetings, each agent is **dynamically assigned a discussion role** based on their current observation history. Roles are reassigned **every discussion round** — adapting as new information emerges mid-meeting.

### Crewmate Role Priority Stack

| Priority | Role | Trigger | Behavior |
|---|---|---|---|
| 1 | **Counter-Attacker** | Accused + Has eyewitness evidence | "I SAW [Player] kill — they're accusing me to save themselves!" |
| 2 | **Defender** | Being accused | Defend with specifics: rooms, timesteps, tasks, alibi witnesses |
| 3 | **Prosecutor** | Witnessed Kill/Vent | Present hard evidence clearly and forcefully |
| 4 | **Detective** | Has location data | Ask questions, find inconsistencies in testimonies |
| 5 | **Bystander** | No strong evidence | Listen, evaluate, vouch for confirmed locations |

### Impostor Role Strategy

Impostors **never receive the Prosecutor role** (too aggressive / draws suspicion). Instead:
- If accused → **Defender** (deflect and redirect)
- If has "witnessed" something → **Detective** (frame without over-committing)
- Otherwise → Random choice between **Detective** and **Bystander** (blend in)

### Meeting Stages

Meetings follow a **3-stage structured debate**:

| Stage | Name | Purpose |
|---|---|---|
| 0 | **Testimony** | Share facts only — where you were, who you saw, what you witnessed. No accusations. |
| 1 | **Accusation & Defense** | Compare testimonies, call out contradictions, defend yourself if accused. Answer questions from Stage 0. |
| 2 | **Final Arguments** | Summarize evidence, state voting intent. No new accusations. |

### Anti-Parrot Mechanism

When multiple players share the same role, each receives a different **Speaking Style** to prevent verbatim echoing:
- Direct & Brief
- Detailed & Methodical
- Emotional & Urgent
- Analytical & Logical
- Conversational & Natural

---

## Prompt Architecture

**File:** `agent/neutral_prompts.py`

### Base System Prompts

Each agent receives a **role-specific system prompt** (Crewmate or Impostor) containing:
- Identity and objectives
- Map configuration with room adjacencies and vent connections
- Kill strategy (Impostor) or task-first rules (Crewmate)
- Public alibi system and fake task system (Impostor)
- Sabotage strategy guide (Impostor)
- Witness reporting and deduction protocol (Crewmate)
- Global constraints (line-of-sight, state persistence, meeting triggers)
- Anti-hallucination rules (Hard Memory vs. Social Memory distinction)

### Structured Output Format

All agents output in a standardized format:

```
[World State Ledger]     ← Room occupancy, movement log, vouch/sus, deception plan
[Thinking Process]       ← Mandatory visual scan, safety check, goal alignment
[Action]                 ← Single action from the available actions list
```

Brevity constraints enforce conciseness (80-word ledger, 100-word thinking) to prevent truncation before the critical `[Action]` line.

### Ghost Prompt

Dead players receive a simplified `GHOST_SYSTEM_PROMPT` — no social deduction, no meetings. Their only actions are MOVE (to any room, no-clip) and COMPLETE TASK.

---

## Setup

```bash
conda create -n among-agents python=3.10
pip install -r requirements.txt
pip install -e .
```

### Quick Start

```bash
python main.py
```

Or use the notebook: `notebooks/run_game.ipynb`

### Configs

| Parameter | Values | Description |
|---|---|---|
| `include_human` | True / False | Add a human player to the game |
| `personality` | True / False | Assign personality archetypes to agents |
| `agent_config` | ALL_LLM / ALL_RANDOM / CREWMATE_LLM / IMPOSTOR_LLM | LLM agent type assignment |

### RL Pipeline Notebook

The `test_pipeline.ipynb` notebook provides a full end-to-end RL training loop with:
- Game rollout visualization with per-timestep metrics
- Reward breakdown analysis
- Critic value trajectories
- Suspicion matrix evolution
- Speaking score distributions
