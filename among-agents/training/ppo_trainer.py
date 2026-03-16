"""
ppo_trainer.py — Joint PPO Trainer

Trains the LoRA policy, BeliefModel, and CriticModel jointly using
Proximal Policy Optimization (PPO).

Loss:
    L = -L_CLIP  +  c₁·L_value  +  c₂·L_belief  -  c₃·H(π)

Where:
    L_CLIP   = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]   (clipped policy objective)
    L_value  = MSE(V(s_t), returns_t)                       (critic regression)
    L_belief = MSE(predicted_beliefs, belief_targets)        (belief head regression)
    H(π)     = -Σ π log π                                   (entropy bonus)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.rollout_collector import Trajectory

logger = logging.getLogger(__name__)


# ── PPO Config ────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Loss coefficients
    clip_eps: float        = 0.2
    value_loss_coef: float = 0.5
    belief_loss_coef: float = 0.3
    entropy_coef: float    = 0.01

    # Optimisation
    ppo_lr_policy: float  = 1e-5   # LoRA adapter learning rate
    ppo_lr_heads: float   = 3e-4   # belief + critic head learning rate
    weight_decay: float   = 1e-5
    max_grad_norm: float  = 0.5
    ppo_epochs: int       = 4      # grad steps per batch of trajectories
    mini_batch_size: int  = 16     # steps per mini-batch

    # KL early stop
    target_kl: float      = 0.02   # stop epoch if mean KL exceeds this

    # Normalise advantages
    normalise_advantages: bool = True

    device: str = "cuda"
    ddp: bool = False           # True when training with DistributedDataParallel


# ── PPO Trainer ───────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for the Among Us RL pipeline.

    Parameters
    ----------
    policy       : LoRAQwenPolicy — the shared backbone + LoRA adapters
    belief_model : BeliefModel    — suspicion / second-order belief head
    critic_model : CriticModel    — value function head
    config       : PPOConfig
    """

    def __init__(
        self,
        policy,          # LoRAQwenPolicy
        belief_model,    # BeliefModel
        critic_model,    # CriticModel
        config: Optional[PPOConfig] = None,
    ):
        self.policy       = policy
        self.belief_model = belief_model
        self.critic_model = critic_model
        self.config       = config or PPOConfig()
        self.device       = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        # Unwrap DDP shell for direct access to model methods
        self._raw_policy       = policy.module  if hasattr(policy,       "module") else policy
        self._raw_belief_model = belief_model.module if hasattr(belief_model, "module") else belief_model
        self._raw_critic_model = critic_model.module if hasattr(critic_model, "module") else critic_model

        self._raw_belief_model.to(self.device)
        self._raw_critic_model.to(self.device)

        # Separate optimisers: policy (LoRA only) vs aux heads
        self.policy_opt = torch.optim.AdamW(
            self._raw_policy.trainable_parameters(),
            lr=config.ppo_lr_policy,
            weight_decay=config.weight_decay,
        )
        aux_params = list(self._raw_belief_model.parameters()) + list(self._raw_critic_model.parameters())
        self.aux_opt = torch.optim.AdamW(
            aux_params,
            lr=config.ppo_lr_heads,
            weight_decay=config.weight_decay,
        )

        # Training history
        self.history: List[Dict] = []

    # ── Public API ────────────────────────────────────────────────────

    def update(
        self,
        trajectories: List[Trajectory],
        role_filter: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run PPO update on a batch of trajectories.

        Parameters
        ----------
        trajectories : List[Trajectory]
        role_filter  : "Crewmate" | "Impostor" | None
            When set, only trajectories for that role are used, and only that
            role’s LoRA adapter parameters are updated.  When None, all
            trajectories are used (original behaviour).

        Returns
        -------
        metrics : dict with keys
            policy_loss, value_loss, belief_loss, entropy, kl_mean, total_loss
        """
        cfg = self.config

        # ── Filter trajectories by role when requested ─────────────────────
        if role_filter is not None:
            trajectories = [t for t in trajectories if t.role == role_filter]
            if not trajectories:
                logger.warning(
                    f"[PPO] No trajectories for role '{role_filter}' in this batch — skipping update."
                )
                return {}

        # ── Activate the correct LoRA adapter params ──────────────────────
        # Switch the active adapter so only that role’s params get gradients.
        if role_filter is not None and hasattr(self._raw_policy, "has_role"):
            if self._raw_policy.has_role(role_filter):
                self._raw_policy.set_active_role(role_filter)
                # Re-build the policy optimizer over the new active params
                self.policy_opt.param_groups[0]["params"] = \
                    self._raw_policy.trainable_parameters(role_filter)
                logger.info(f"[PPO] Updating adapter for role: {role_filter}")

        batch = self._pack_trajectories(trajectories)
        if batch is None:
            logger.warning("Empty trajectory batch — skipping update.")
            return {}

        # Normalise advantages across the whole batch
        if cfg.normalise_advantages and batch["advantages"].numel() > 1:
            adv = batch["advantages"]
            batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Set to train mode
        self.policy.train()
        self.belief_model.train()
        self.critic_model.train()

        aggregated = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "belief_loss": 0.0, "entropy": 0.0,
            "kl_mean": 0.0,     "total_loss": 0.0,
        }
        n_updates = 0

        for epoch in range(cfg.ppo_epochs):
            # Shuffle and split into mini-batches
            perm = torch.randperm(batch["advantages"].shape[0])
            kl_epoch = 0.0
            n_mini  = 0

            for start in range(0, len(perm), cfg.mini_batch_size):
                idx = perm[start : start + cfg.mini_batch_size]
                if len(idx) == 0:
                    continue

                metrics = self._mini_batch_update(batch, idx)

                kl_epoch += metrics["kl_mean"]
                n_mini   += 1
                for k, v in metrics.items():
                    aggregated[k] += v
                n_updates += 1

            # KL early-stop check
            mean_kl = kl_epoch / max(n_mini, 1)
            if mean_kl > cfg.target_kl:
                logger.info(
                    f"PPO early stop at epoch {epoch+1}: "
                    f"KL={mean_kl:.5f} > target {cfg.target_kl}"
                )
                break

        if n_updates > 0:
            for k in aggregated:
                aggregated[k] /= n_updates

        self.history.append(aggregated)
        self._log_metrics(aggregated)
        return aggregated

    # ── Mini-Batch Update ─────────────────────────────────────────────

    def _mini_batch_update(
        self,
        batch: Dict[str, torch.Tensor],
        idx: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute and apply one PPO gradient step on mini-batch *idx*."""
        cfg = self.config

        # Gather mini-batch tensors
        advantages    = batch["advantages"][idx].to(self.device)
        returns       = batch["returns"][idx].to(self.device)
        log_probs_old = batch["log_probs_old"][idx].to(self.device)
        hidden_pools  = batch["hidden_pools"][idx].to(self.device)
        belief_vecs   = batch["belief_vecs"][idx].to(self.device)
        belief_tgts   = batch["belief_targets"][idx].to(self.device)
        game_feats    = batch["game_feats"][idx].to(self.device)
        role_ids      = batch["role_ids"][idx].to(self.device)
        input_ids_list  = [batch["input_ids_list"][i]  for i in idx.tolist()]
        action_ids_list = [batch["action_ids_list"][i] for i in idx.tolist()]

        # ── Policy: compute new log-probs ────────────────────────────
        # We recompute log π_new(a|s) and also extract updated hidden states
        log_probs_new_list = []
        hidden_pools_new_list = []
        for inp, act in zip(input_ids_list, action_ids_list):
            lp, hp = self.policy.get_log_probs_and_hidden(
                inp.to(self.device), act.to(self.device)
            )
            log_probs_new_list.append(lp)
            hidden_pools_new_list.append(hp.squeeze(0))

        log_probs_new = torch.stack(log_probs_new_list)  # [B]
        hidden_new    = torch.stack(hidden_pools_new_list) # [B, H]

        # ── PPO Policy Loss (CLIP) ───────────────────────────────────
        ratio = torch.exp(log_probs_new - log_probs_old)  # r_t = π_new / π_old
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Entropy Bonus ────────────────────────────────────────────
        # Approx entropy from log-probs: H ≈ -E[log π]
        entropy = -log_probs_new.mean()

        # ── KL Divergence (monitoring) ────────────────────────────────
        with torch.no_grad():
            kl = (log_probs_old - log_probs_new).mean()

        # ── Critic Value Loss ────────────────────────────────────────
        value_pred = self.critic_model(hidden_new, game_feats)  # [B]
        value_loss = F.mse_loss(value_pred, returns)

        # ── Belief Loss ──────────────────────────────────────────────
        belief_pred = self.belief_model(hidden_new, belief_vecs, role_ids)  # [B, D]
        belief_loss = self.belief_model.loss(belief_pred, belief_tgts)

        # ── Total Loss ───────────────────────────────────────────────
        total_loss = (
            policy_loss
            + cfg.value_loss_coef  * value_loss
            + cfg.belief_loss_coef * belief_loss
            - cfg.entropy_coef     * entropy
        )

        # ── Backprop ─────────────────────────────────────────────────
        self.policy_opt.zero_grad()
        self.aux_opt.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(self._raw_policy.trainable_parameters(), cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(
            list(self._raw_belief_model.parameters()) + list(self._raw_critic_model.parameters()),
            cfg.max_grad_norm,
        )

        self.policy_opt.step()
        self.aux_opt.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "belief_loss": belief_loss.item(),
            "entropy":     entropy.item(),
            "kl_mean":     kl.item(),
            "total_loss":  total_loss.item(),
        }

    # ── Trajectory Packing ────────────────────────────────────────────

    def _pack_trajectories(
        self, trajectories: List[Trajectory]
    ) -> Optional[Dict]:
        """Flatten all trajectories into a single batch dict."""
        log_probs_old = []
        hidden_pools  = []
        belief_vecs   = []
        belief_tgts   = []
        game_feats    = []
        role_ids      = []
        advantages    = []
        returns       = []
        input_ids_list  = []
        action_ids_list = []

        for traj in trajectories:
            if traj.advantages is None or len(traj.steps) == 0:
                continue
            T = len(traj.steps)
            is_imp = traj.role == "Impostor"

            log_probs_old.append(traj.log_probs_old)
            hidden_pools.append(traj.hidden_pools)
            belief_vecs.append(traj.belief_vecs)
            belief_tgts.append(traj.belief_targets)
            game_feats.append(traj.game_feats)
            role_ids.append(torch.full((T,), 1 if is_imp else 0, dtype=torch.long))
            advantages.append(traj.advantages)
            returns.append(traj.returns)
            input_ids_list.extend(traj.input_ids_list)
            action_ids_list.extend(traj.action_ids_list)

        if not advantages:
            return None

        return {
            "log_probs_old":    torch.cat(log_probs_old),
            "hidden_pools":     torch.cat(hidden_pools),
            "belief_vecs":      torch.cat(belief_vecs),
            "belief_targets":   torch.cat(belief_tgts),
            "game_feats":       torch.cat(game_feats),
            "role_ids":         torch.cat(role_ids),
            "advantages":       torch.cat(advantages),
            "returns":          torch.cat(returns),
            "input_ids_list":   input_ids_list,
            "action_ids_list":  action_ids_list,
        }

    # ── Checkpointing ─────────────────────────────────────────────────

    def save_checkpoint(self, directory: str, iteration: int):
        """Save all model weights at a given iteration."""
        ckpt_dir = os.path.join(directory, f"iter_{iteration:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        self._raw_policy.save_checkpoint(os.path.join(ckpt_dir, "policy"))
        torch.save(self._raw_belief_model.state_dict(),
                   os.path.join(ckpt_dir, "belief_model.pt"))
        torch.save(self._raw_critic_model.state_dict(),
                   os.path.join(ckpt_dir, "critic_model.pt"))
        torch.save(self.policy_opt.state_dict(),
                   os.path.join(ckpt_dir, "policy_opt.pt"))
        torch.save(self.aux_opt.state_dict(),
                   os.path.join(ckpt_dir, "aux_opt.pt"))
        logger.info(f"PPO checkpoint saved → {ckpt_dir}")

    def save_role_checkpoint(self, role: str, directory: str, iteration: int) -> None:
        """
        Save ONLY the LoRA adapter for *role* (+ belief/critic heads).
        Belief model and critic are shared across roles so are saved once.
        """
        ckpt_dir = os.path.join(directory, role.lower(), f"iter_{iteration:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if hasattr(self._raw_policy, "save_role_checkpoint"):
            self._raw_policy.save_role_checkpoint(role, os.path.join(ckpt_dir, "policy"))
        else:
            self._raw_policy.save_checkpoint(os.path.join(ckpt_dir, "policy"))
        torch.save(self._raw_belief_model.state_dict(),
                   os.path.join(ckpt_dir, "belief_model.pt"))
        torch.save(self._raw_critic_model.state_dict(),
                   os.path.join(ckpt_dir, "critic_model.pt"))
        torch.save(self.policy_opt.state_dict(),
                   os.path.join(ckpt_dir, "policy_opt.pt"))
        logger.info(f"PPO role-checkpoint '{role}' saved → {ckpt_dir}")

    def load_role_checkpoint(self, role: str, directory: str) -> None:
        """Load the role-specific LoRA adapter + shared heads from *directory*."""
        policy_dir = os.path.join(directory, "policy")
        if os.path.isdir(policy_dir) and hasattr(self._raw_policy, "load_role_checkpoint"):
            self._raw_policy.load_role_checkpoint(role, policy_dir)
        elif os.path.isdir(policy_dir):
            self._raw_policy.load_checkpoint(policy_dir)
        bpath = os.path.join(directory, "belief_model.pt")
        cpath = os.path.join(directory, "critic_model.pt")
        if os.path.exists(bpath):
            self._raw_belief_model.load_state_dict(
                torch.load(bpath, map_location=self.device))
        if os.path.exists(cpath):
            self._raw_critic_model.load_state_dict(
                torch.load(cpath, map_location=self.device))
        logger.info(f"PPO role-checkpoint '{role}' loaded from {directory}")

    def load_checkpoint(self, directory: str):
        """Load all model weights from a checkpoint directory."""
        self._raw_policy.load_checkpoint(os.path.join(directory, "policy"))
        bpath = os.path.join(directory, "belief_model.pt")
        cpath = os.path.join(directory, "critic_model.pt")
        if os.path.exists(bpath):
            self._raw_belief_model.load_state_dict(
                torch.load(bpath, map_location=self.device))
        if os.path.exists(cpath):
            self._raw_critic_model.load_state_dict(
                torch.load(cpath, map_location=self.device))
        popt = os.path.join(directory, "policy_opt.pt")
        aopt = os.path.join(directory, "aux_opt.pt")
        if os.path.exists(popt):
            self.policy_opt.load_state_dict(torch.load(popt))
        if os.path.exists(aopt):
            self.aux_opt.load_state_dict(torch.load(aopt))
        logger.info(f"PPO checkpoint loaded from {directory}")

    # ── Logging ───────────────────────────────────────────────────────

    @staticmethod
    def _log_metrics(metrics: Dict[str, float]):
        msg = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        logger.info(f"[PPO] {msg}")
