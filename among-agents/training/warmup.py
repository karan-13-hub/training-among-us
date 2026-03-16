"""
warmup.py — Supervised Warmup Trainer for Belief Model + Critic

Before PPO begins, both auxiliary heads are pre-trained with supervision:
  1. Belief Model: MSE on rule-derived belief targets (from KILL/VENT/TASK events)
  2. Critic Model: MSE on heuristic V(s) labels (from critic_model.py helpers)

The policy backbone is FROZEN during warmup; only the MLP heads are updated.

Usage
-----
    trainer = WarmupTrainer(policy, belief_model, critic_model, config)
    trainer.run()   # collect data → train belief → train critic
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.belief_model import (
    BeliefModel,
    BeliefModelConfig,
    build_belief_targets,
    beliefs_to_tensor,
)
from training.critic_model import (
    CriticModel,
    CriticModelConfig,
    extract_game_features,
    heuristic_value_from_state,
)

logger = logging.getLogger(__name__)


# ── Warmup Config ────────────────────────────────────────────────────────

@dataclass
class WarmupConfig:
    warmup_games: int = 20          # number of games for supervised data collection
    belief_epochs: int = 5
    critic_epochs: int = 5
    belief_batch_size: int = 64
    critic_batch_size: int = 64
    belief_lr: float = 1e-4
    critic_lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    max_players: int = 9            # max other players tracked (belief_dim)
    log_every: int = 50             # print loss every N steps


# ── Dataset Classes ──────────────────────────────────────────────────────

class BeliefDataset(Dataset):
    """
    Each sample is one (state, target) pair derived from game events.
    """

    def __init__(
        self,
        hidden_pools: List[torch.Tensor],     # each [hidden_size]
        belief_vecs:  List[torch.Tensor],     # each [belief_dim] — current beliefs
        role_ids:     List[int],              # 0=Crewmate, 1=Impostor
        targets:      List[torch.Tensor],     # each [belief_dim] — ground-truth targets
    ):
        assert len(hidden_pools) == len(belief_vecs) == len(role_ids) == len(targets)
        self.hidden_pools = hidden_pools
        self.belief_vecs  = belief_vecs
        self.role_ids     = role_ids
        self.targets      = targets

    def __len__(self):
        return len(self.hidden_pools)

    def __getitem__(self, idx):
        return (
            self.hidden_pools[idx],
            self.belief_vecs[idx],
            torch.tensor(self.role_ids[idx], dtype=torch.long),
            self.targets[idx],
        )


class CriticDataset(Dataset):
    """
    Each sample is (hidden_pool, game_features, heuristic_value).
    """

    def __init__(
        self,
        hidden_pools: List[torch.Tensor],   # each [hidden_size]
        game_feats:   List[torch.Tensor],   # each [5]
        values:       List[float],          # heuristic V(s) ∈ [0,1]
    ):
        assert len(hidden_pools) == len(game_feats) == len(values)
        self.hidden_pools = hidden_pools
        self.game_feats   = game_feats
        self.values       = values

    def __len__(self):
        return len(self.hidden_pools)

    def __getitem__(self, idx):
        return (
            self.hidden_pools[idx],
            self.game_feats[idx],
            torch.tensor(self.values[idx], dtype=torch.float32),
        )


# ── Synthetic Data Generator (no GPU / policy needed) ────────────────────

def _synthetic_hidden(hidden_size: int, device: str) -> torch.Tensor:
    """Gaussian noise as a mock hidden pool for offline warmup."""
    return torch.randn(hidden_size)


def _synthetic_game_state(num_players: int = 5, num_imps: int = 1) -> Dict[str, Any]:
    """Generate a random plausible game state dict."""
    crew = random.randint(num_imps, num_players - num_imps)
    task = random.uniform(0, 100)
    sab  = random.random() < 0.2
    return {
        "living_crewmates": crew,
        "living_impostors": num_imps,
        "task_completion_pct": task,
        "sabotage_active": sab,
        "winner": None,
    }


SYNTHETIC_ACTIONS = ["KILL", "VENT", "SABOTAGE", "FAKE_TASK", "VISUAL_TASK",
                     "COMPLETE_TASK", "MOVE", "STAY"]


def collect_synthetic_belief_data(
    n_samples: int = 5000,
    hidden_size: int = 2048,
    belief_dim: int = 9,
    device: str = "cpu",
) -> BeliefDataset:
    """
    Build a synthetic BeliefDataset using only rule-based belief targets.
    No real games or GPU required. Used for offline / CPU warmup.

    Each sample simulates:
      - current belief vector (random [0,1] per slot)
      - one observation event (random action by a random other player)
      - updated belief target derived by rules
    """
    hidden_pools, belief_vecs, role_ids, targets = [], [], [], []

    for _ in range(n_samples):
        is_impostor = random.random() < 0.2
        role = 1 if is_impostor else 0

        # Assign random player names for this sample
        player_names = [f"Player {i}" for i in range(1, belief_dim + 2)]
        self_name    = player_names[0]
        others       = player_names[1:]

        # Current belief vector (random)
        cur_beliefs  = {n: random.random() for n in others}
        cur_tensor   = beliefs_to_tensor(cur_beliefs, others, belief_dim)

        # One random observation event
        subject = random.choice(others)
        action  = random.choice(SYNTHETIC_ACTIONS)
        obs_log = [{"subject": subject, "action": action, "witnesses": [self_name]}]

        # Compute target using rules
        updated = build_belief_targets(obs_log, cur_beliefs, self_name)
        tgt_tensor = beliefs_to_tensor(updated, others, belief_dim)

        # Mock hidden state (Gaussian)
        h_pool = _synthetic_hidden(hidden_size, device)

        hidden_pools.append(h_pool)
        belief_vecs.append(cur_tensor)
        role_ids.append(role)
        targets.append(tgt_tensor)

    return BeliefDataset(hidden_pools, belief_vecs, role_ids, targets)


def collect_synthetic_critic_data(
    n_samples: int = 5000,
    hidden_size: int = 2048,
    device: str = "cpu",
) -> CriticDataset:
    """
    Build a synthetic CriticDataset using heuristic value labels.
    No real games or GPU required.
    """
    hidden_pools, game_feats, values = [], [], []

    for _ in range(n_samples):
        is_impostor = random.random() < 0.2
        gs = _synthetic_game_state()
        v  = heuristic_value_from_state(gs, is_impostor)
        gf = extract_game_features(gs, is_impostor)

        h_pool = _synthetic_hidden(hidden_size, device)
        hidden_pools.append(h_pool)
        game_feats.append(gf)
        values.append(v)

    return CriticDataset(hidden_pools, game_feats, values)


# ── WarmupTrainer ─────────────────────────────────────────────────────────

class WarmupTrainer:
    """
    Supervised pre-trainer for BeliefModel and CriticModel.

    Parameters
    ----------
    belief_model : BeliefModel
    critic_model : CriticModel
    config : WarmupConfig
    policy : Optional[LoRAQwenPolicy]
        If provided, hidden states come from the real policy backbone.
        If None, uses synthetic Gaussian hidden states (offline / unit-test mode).
    """

    def __init__(
        self,
        belief_model: BeliefModel,
        critic_model: CriticModel,
        config: Optional[WarmupConfig] = None,
        policy=None,    # LoRAQwenPolicy — optional
    ):
        self.belief_model = belief_model
        self.critic_model = critic_model
        self.config = config or WarmupConfig()
        self.policy = policy
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.belief_model.to(self.device)
        self.critic_model.to(self.device)

        self.belief_opt = torch.optim.AdamW(
            belief_model.parameters(),
            lr=config.belief_lr,
            weight_decay=config.weight_decay,
        )
        self.critic_opt = torch.optim.AdamW(
            critic_model.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )

        self.belief_history: List[float] = []
        self.critic_history: List[float] = []

    # ── Public API ────────────────────────────────────────────────────

    def run(self, belief_dataset=None, critic_dataset=None):
        """
        Full warmup sequence:
            1. Collect/generate supervised datasets (if not provided)
            2. Train belief model
            3. Train critic model
        """
        cfg = self.config

        if belief_dataset is None:
            hidden_size = self.belief_model.config.hidden_size
            logger.info("Generating synthetic belief warmup data …")
            belief_dataset = collect_synthetic_belief_data(
                n_samples=cfg.warmup_games * 200,
                hidden_size=hidden_size,
                belief_dim=self.belief_model.config.belief_dim,
                device="cpu",
            )

        if critic_dataset is None:
            hidden_size = self.critic_model.config.hidden_size
            logger.info("Generating synthetic critic warmup data …")
            critic_dataset = collect_synthetic_critic_data(
                n_samples=cfg.warmup_games * 200,
                hidden_size=hidden_size,
                device="cpu",
            )

        self.train_belief(belief_dataset)
        self.train_critic(critic_dataset)

    def train_belief(self, dataset: BeliefDataset) -> List[float]:
        """Train BeliefModel on supervised belief targets."""
        cfg = self.config
        loader = DataLoader(dataset, batch_size=cfg.belief_batch_size, shuffle=True)
        self.belief_model.train()
        losses = []

        logger.info(f"=== Warmup: Belief Model ({cfg.belief_epochs} epochs) ===")
        for epoch in range(cfg.belief_epochs):
            epoch_loss = 0.0
            for step, (hidden_pool, belief_vec, role_id, target) in enumerate(loader):
                hidden_pool = hidden_pool.to(self.device)
                belief_vec  = belief_vec.to(self.device)
                role_id     = role_id.to(self.device)
                target      = target.to(self.device)

                predicted = self.belief_model(hidden_pool, belief_vec, role_id)
                loss = self.belief_model.loss(predicted, target)

                self.belief_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.belief_model.parameters(), 1.0)
                self.belief_opt.step()

                epoch_loss += loss.item()
                if step % cfg.log_every == 0:
                    logger.debug(
                        f"  [Belief] Epoch {epoch+1}/{cfg.belief_epochs} "
                        f"Step {step} — loss={loss.item():.5f}"
                    )

            avg = epoch_loss / max(len(loader), 1)
            losses.append(avg)
            logger.info(f"  [Belief] Epoch {epoch+1}/{cfg.belief_epochs} — avg_loss={avg:.5f}")

        self.belief_history.extend(losses)
        return losses

    def train_critic(self, dataset: CriticDataset) -> List[float]:
        """Train CriticModel on heuristic V(s) targets."""
        cfg = self.config
        loader = DataLoader(dataset, batch_size=cfg.critic_batch_size, shuffle=True)
        self.critic_model.train()
        losses = []

        logger.info(f"=== Warmup: Critic Model ({cfg.critic_epochs} epochs) ===")
        for epoch in range(cfg.critic_epochs):
            epoch_loss = 0.0
            for step, (hidden_pool, game_feat, value_target) in enumerate(loader):
                hidden_pool  = hidden_pool.to(self.device)
                game_feat    = game_feat.to(self.device)
                value_target = value_target.to(self.device)

                predicted = self.critic_model(hidden_pool, game_feat)
                loss = self.critic_model.loss(predicted, value_target)

                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic_model.parameters(), 1.0)
                self.critic_opt.step()

                epoch_loss += loss.item()
                if step % cfg.log_every == 0:
                    logger.debug(
                        f"  [Critic] Epoch {epoch+1}/{cfg.critic_epochs} "
                        f"Step {step} — loss={loss.item():.5f}"
                    )

            avg = epoch_loss / max(len(loader), 1)
            losses.append(avg)
            logger.info(f"  [Critic] Epoch {epoch+1}/{cfg.critic_epochs} — avg_loss={avg:.5f}")

        self.critic_history.extend(losses)
        return losses

    def save_warmup_checkpoint(self, directory: str):
        """Save both head weights after warmup."""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.belief_model.state_dict(),
                   os.path.join(directory, "belief_model.pt"))
        torch.save(self.critic_model.state_dict(),
                   os.path.join(directory, "critic_model.pt"))
        logger.info(f"Warmup checkpoints saved to {directory}")

    def load_warmup_checkpoint(self, directory: str):
        """Load head weights from a warmup checkpoint."""
        bpath = os.path.join(directory, "belief_model.pt")
        cpath = os.path.join(directory, "critic_model.pt")
        if os.path.exists(bpath):
            self.belief_model.load_state_dict(
                torch.load(bpath, map_location=self.device))
            logger.info(f"Belief model loaded from {bpath}")
        if os.path.exists(cpath):
            self.critic_model.load_state_dict(
                torch.load(cpath, map_location=self.device))
            logger.info(f"Critic model loaded from {cpath}")
