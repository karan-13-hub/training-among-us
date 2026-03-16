"""
rollout_collector.py — Trajectory Collection for PPO

Runs Among Us games with the LoRA policy replacing LLMAgent.send_request,
recording all data needed for PPO:
  - log_probs_old   : π_old(a | s) for each step
  - hidden_pools    : backbone hidden state (pooled) for belief + critic
  - values          : V(s_t) from CriticModel
  - beliefs         : belief vector at each step
  - belief_targets  : rule-derived supervision targets
  - rewards         : r_t from RewardEngine

After all games in a collection round finish, GAE advantages are computed.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from training.belief_model import (
    BeliefModel,
    build_belief_targets,
    beliefs_to_tensor,
    BELIEF_RULES,
)
from training.critic_model import (
    CriticModel,
    extract_game_features,
)

logger = logging.getLogger(__name__)


# ── Trajectory Dataclass ──────────────────────────────────────────────────

@dataclass
class StepData:
    """Data for a single agent step."""
    timestep: int
    player_name: str
    role: str                       # "Crewmate" | "Impostor"
    is_impostor: bool

    # Policy data
    prompt: str                     # full text prompt sent to LLM
    action_text: str                # raw action string generated
    input_ids: torch.Tensor         # [1, L_prompt]
    action_ids: torch.Tensor        # [1, L_action]
    log_prob_old: float             # scalar log π_old(a|s)
    hidden_pool: torch.Tensor       # [hidden_size] — pooled backbone hidden state

    # Auxiliary head data
    belief_vec: torch.Tensor        # [belief_dim] — beliefs BEFORE this step
    belief_target: torch.Tensor     # [belief_dim] — rule-derived target update
    value: float                    # V(s_t) from CriticModel
    game_feat: torch.Tensor         # [5] — game features at this step

    # Reward
    reward: float                   # r_t from RewardEngine

    # Game state snapshot
    game_state: Dict[str, Any]


@dataclass
class Trajectory:
    """Completed trajectory for one player across one game."""
    player_name: str
    role: str
    steps: List[StepData] = field(default_factory=list)

    # Filled in after game ends via compute_gae()
    advantages: Optional[torch.Tensor] = None   # [T]
    returns: Optional[torch.Tensor] = None      # [T]

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        final_value: float = 0.0,
    ):
        """
        Compute Generalised Advantage Estimates in-place.

        A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        where δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        Parameters
        ----------
        final_value : float
            V(s_{T+1}) — bootstrap value at episode end (0 if terminal).
        """
        T = len(self.steps)
        if T == 0:
            self.advantages = torch.zeros(0)
            self.returns    = torch.zeros(0)
            return

        rewards = torch.tensor([s.reward for s in self.steps], dtype=torch.float32)
        values  = torch.tensor([s.value  for s in self.steps], dtype=torch.float32)

        advantages = torch.zeros(T)
        last_gae   = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1].item() if t + 1 < T else final_value
            delta = rewards[t] + gamma * next_val - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        self.returns    = advantages + values
        self.advantages = advantages

    @property
    def log_probs_old(self) -> torch.Tensor:
        return torch.tensor([s.log_prob_old for s in self.steps], dtype=torch.float32)

    @property
    def hidden_pools(self) -> torch.Tensor:
        return torch.stack([s.hidden_pool for s in self.steps])  # [T, H]

    @property
    def belief_vecs(self) -> torch.Tensor:
        return torch.stack([s.belief_vec for s in self.steps])   # [T, belief_dim]

    @property
    def belief_targets(self) -> torch.Tensor:
        return torch.stack([s.belief_target for s in self.steps]) # [T, belief_dim]

    @property
    def game_feats(self) -> torch.Tensor:
        return torch.stack([s.game_feat for s in self.steps])    # [T, 5]

    @property
    def values_tensor(self) -> torch.Tensor:
        return torch.tensor([s.value for s in self.steps], dtype=torch.float32)  # [T]

    @property
    def input_ids_list(self) -> List[torch.Tensor]:
        return [s.input_ids for s in self.steps]

    @property
    def action_ids_list(self) -> List[torch.Tensor]:
        return [s.action_ids for s in self.steps]


# ── Rollout Collector ─────────────────────────────────────────────────────

@dataclass
class RolloutConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    belief_dim: int = 9
    device: str = "cuda"
    num_games: int = 5


class RolloutCollector:
    """
    Collects game trajectories by patching LLMAgent.send_request with
    the LoRA policy, capturing log-probs, hidden states, beliefs, values,
    and rewards at every step.

    Usage
    -----
    collector = RolloutCollector(policy, belief_model, critic_model, reward_engine, config)
    trajectories = await collector.collect(game_factory, num_games=5)
    """

    def __init__(
        self,
        policy,          # LoRAQwenPolicy
        belief_model: BeliefModel,
        critic_model: CriticModel,
        reward_engine,   # RewardEngine
        config: Optional[RolloutConfig] = None,
    ):
        self.policy       = policy
        self.belief_model = belief_model
        self.critic_model = critic_model
        self.reward_engine = reward_engine
        self.config = config or RolloutConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # Set models to eval during rollout
        self.belief_model.eval()
        self.critic_model.eval()
        self.policy.eval()

    # ── Collection ────────────────────────────────────────────────────

    async def collect(
        self,
        game_factory,    # callable() → AmongUs instance
        num_games: Optional[int] = None,
    ) -> List[Trajectory]:
        """
        Run *num_games* games and return a list of Trajectory objects
        (one per agent per game).
        """
        n = num_games or self.config.num_games
        all_trajectories: List[Trajectory] = []

        for game_idx in range(n):
            logger.info(f"=== Rollout game {game_idx + 1}/{n} ===")
            game = game_factory()
            game.initialize_game()

            # Per-game step buffer: player_name → Trajectory
            traj_map: Dict[str, Trajectory] = {
                agent.player.name: Trajectory(
                    player_name=agent.player.name,
                    role=agent.player.identity,
                )
                for agent in game.agents
            }

            # Current belief state per player
            belief_state: Dict[str, Dict[str, float]] = {}
            player_orders: Dict[str, List[str]] = {}
            for agent in game.agents:
                others = [a.player.name for a in game.agents
                          if a.player.name != agent.player.name]
                player_orders[agent.player.name] = others
                belief_state[agent.player.name] = {n: 0.5 for n in others}

            # Patch send_request to intercept prompts and actions
            self._install_hook(game, traj_map, belief_state, player_orders)

            # Run the game
            winner = await self._run_game_loop(game)

            # Uninstall hook
            self._uninstall_hook(game)

            # Finalize trajectories: assign final rewards + compute GAE
            for agent in game.agents:
                name = agent.player.name
                traj = traj_map[name]
                traj.compute_gae(
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    final_value=0.0,   # terminal state
                )
                all_trajectories.append(traj)
                logger.debug(
                    f"  Trajectory {name}: {len(traj.steps)} steps, "
                    f"cum_reward={sum(s.reward for s in traj.steps):.2f}"
                )

        return all_trajectories

    # ── Game Loop ─────────────────────────────────────────────────────

    async def _run_game_loop(self, game) -> Optional[int]:
        """Step through the game until game_over or max_timesteps."""
        winner = None
        while not game.game_over:
            await game.game_step()
            winner = game.check_game_over()
            if winner:
                game.game_over = True
                game.winner = winner
                game.report_winner(winner)
                break
        return winner

    # ── Hook Installation ─────────────────────────────────────────────

    def _install_hook(self, game, traj_map, belief_state, player_orders):
        """Monkey-patch LLMAgent.send_request on all agents in this game."""
        collector = self

        # We need to capture 'agent' in the closure for each agent separately
        for agent in game.agents:
            if not hasattr(agent, 'send_request'):
                continue   # non-LLM agents

            original_send = agent.__class__.send_request

            async def _hooked_send(self_agent, messages, _orig=original_send, _collector=collector,
                                   _traj_map=traj_map, _belief_state=belief_state,
                                   _player_orders=player_orders, _game=game):
                return await _collector._process_step(
                    self_agent, messages, _traj_map, _belief_state,
                    _player_orders, _game
                )

            # Bind to instance to avoid altering the class for all agents
            import types
            agent.send_request = types.MethodType(_hooked_send, agent)

    def _uninstall_hook(self, game):
        """Remove hooks by deleting instance-level overrides."""
        for agent in game.agents:
            if 'send_request' in agent.__dict__:
                del agent.__dict__['send_request']

    # ── Single Step Processing ────────────────────────────────────────

    async def _process_step(
        self,
        agent,
        messages: List[Dict],
        traj_map: Dict[str, Trajectory],
        belief_state: Dict[str, Dict[str, float]],
        player_orders: Dict[str, List[str]],
        game,
    ) -> str:
        """
        Called in place of LLMAgent.send_request for each agent step.
        Generates an action using the LoRA policy and records all data.
        """
        player_name = agent.player.name
        is_impostor = agent.player.identity == "Impostor"
        role        = "Impostor" if is_impostor else "Crewmate"
        role_id     = 1 if is_impostor else 0

        # Build prompt string from messages list
        prompt_str = self._messages_to_prompt(messages)

        # ─── Switch to role-specific LoRA adapter ─────────────────────────
        # If the policy has named adapters, activate the one matching this role.
        # Falls back transparently if only a single (default) adapter exists.
        _policy = self.policy.module if hasattr(self.policy, "module") else self.policy
        if hasattr(_policy, "has_role") and _policy.has_role(role):
            _policy.set_active_role(role)

        # ─── Generate with LoRA policy ─────────────────────────────────────
        action_text, input_ids, action_ids = _policy.generate(prompt_str)

        # ─── Compute log-prob + hidden pool ────────────────────────────────
        with torch.no_grad():
            log_prob, hidden_pool = _policy.get_log_probs_and_hidden(
                input_ids.to(self.device),
                action_ids.to(self.device),
            )
        log_prob_val = log_prob.item()
        hidden_pool  = hidden_pool.squeeze(0).cpu()   # [H]

        # ─── Current belief vector ─────────────────────────────────
        others    = player_orders.get(player_name, [])
        cur_beliefs = belief_state.get(player_name, {})
        belief_dim  = self.config.belief_dim
        belief_vec  = beliefs_to_tensor(cur_beliefs, others, belief_dim)

        # ─── Belief head: forward pass ──────────────────────────────
        role_tensor = torch.tensor([role_id], dtype=torch.long)
        with torch.no_grad():
            updated_belief = self.belief_model(
                hidden_pool.unsqueeze(0).to(self.device),
                belief_vec.unsqueeze(0).to(self.device),
                role_tensor.to(self.device),
            ).squeeze(0).cpu()    # [belief_dim]

        # Build rule-based belief target from recent observations
        obs_log = self._extract_obs_events(agent)
        rule_updated = build_belief_targets(obs_log, cur_beliefs, player_name)
        belief_target = beliefs_to_tensor(rule_updated, others, belief_dim)

        # ─── Sync model beliefs back to actor_module (prompt-facing) ───────
        # The actor_module's suspicion_matrix / second_order_beliefs feed
        # format_belief_for_prompt(), which is injected into every LLM prompt.
        # We push the hidden pool so the model output replaces rule values there.
        if hasattr(agent, 'actor_module') and agent.actor_module is not None:
            try:
                agent.actor_module.update_beliefs_from_model(hidden_pool)
            except Exception as _sync_err:
                logger.debug(f"Belief model sync failed for {player_name}: {_sync_err}")

        # Update belief_state with model-predicted values for the next step's prior.
        # We use model output (updated_belief) for the prior, but KEEP rule_updated
        # as the supervision target (belief_target) so the model learns correct labels.
        model_belief_dict = {
            name: float(updated_belief[i].clamp(0.0, 1.0).item())
            for i, name in enumerate(others)
            if i < len(updated_belief)
        }
        belief_state[player_name] = model_belief_dict

        # ─── Game features ─────────────────────────────────────────
        gs_dict = self._make_game_state_dict(game, agent)
        game_feat = extract_game_features(gs_dict, is_impostor)

        # ─── Critic: forward pass ──────────────────────────────────
        with torch.no_grad():
            value = self.critic_model(
                hidden_pool.unsqueeze(0).to(self.device),
                game_feat.unsqueeze(0).to(self.device),
            ).item()

        # ─── Reward ────────────────────────────────────────────────
        # We compute reward post-hoc with placeholder action_log
        # (a more complete reward requires game state AFTER the action,
        #  so full reward is computed in PPOTrainer after game_step)
        reward = 0.0  # will be filled in by PPOTrainer

        # ─── Record step ───────────────────────────────────────────
        step = StepData(
            timestep=game.timestep,
            player_name=player_name,
            role=agent.player.identity,
            is_impostor=is_impostor,
            prompt=prompt_str,
            action_text=action_text,
            input_ids=input_ids.cpu(),
            action_ids=action_ids.cpu(),
            log_prob_old=log_prob_val,
            hidden_pool=hidden_pool,
            belief_vec=belief_vec,
            belief_target=belief_target,
            value=value,
            game_feat=game_feat,
            reward=reward,
            game_state=gs_dict,
        )
        traj_map[player_name].steps.append(step)

        return action_text


    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _messages_to_prompt(messages: List[Dict]) -> str:
        """Flatten OpenAI-style messages to a single prompt string."""
        parts = []
        for m in messages:
            role        = m.get("role", "user")
            content     = m.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_obs_events(agent) -> List[Dict]:
        """
        Extract recent observation history into the event format expected
        by build_belief_targets().
        This is a best-effort heuristic parse of the text observations.
        """
        events = []
        obs    = getattr(agent.player, 'observation_history', [])
        # Only read the last few observations since the previous step
        for obs_str in obs[-3:]:
            obs_up = obs_str.upper()
            for action_key in BELIEF_RULES:
                if action_key in obs_up:
                    # Try to extract subject (who did it)
                    # Observation format: "Player X: <color> <action>"
                    import re
                    m = re.search(r'Player\s+\d+:\s+\w+', obs_str)
                    if m:
                        subject = m.group(0)
                        events.append({
                            "subject":   subject,
                            "action":    action_key,
                            "witnesses": [agent.player.name],
                        })
                    break
        return events

    @staticmethod
    def _make_game_state_dict(game, agent) -> Dict[str, Any]:
        """Build a game_state dict from the live game object."""
        living_crew = sum(
            1 for p in game.players
            if p.identity == "Crewmate" and p.is_alive
        )
        living_imps = sum(
            1 for p in game.players
            if p.identity == "Impostor" and p.is_alive
        )
        task_pct = game.task_assignment.check_task_completion() * 100.0
        sab_active = bool(getattr(game, 'active_sabotages', {}))
        return {
            "living_crewmates":    living_crew,
            "living_impostors":    living_imps,
            "task_completion_pct": task_pct,
            "sabotage_active":     sab_active,
            "winner":              game.winner,
        }
