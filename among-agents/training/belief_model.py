"""
belief_model.py — Trainable Belief Model

Implements a learnable belief update module that replaces the hard-coded
rule updates in ActorModule.update_beliefs().

Architecture:
    [hidden_pool, belief_vec, role_emb] → MLP → updated belief vector

Two separate output heads (weight-sharing in trunk):
  • Crewmate   → suspicion_matrix:        who is the Impostor?
  • Impostor   → second_order_beliefs:    who suspects ME?

Each output ∈ [0.0, 1.0] per other-player slot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class BeliefModelConfig:
    hidden_size: int = 2048         # backbone hidden_size from Qwen3-4B
    belief_dim: int = 9             # max number of other players tracked (≤ num_players-1)
    mlp_hidden: int = 256
    role_emb_dim: int = 8           # tiny learnable embedding for Crewmate vs Impostor
    dropout: float = 0.1


# ── Belief Model ──────────────────────────────────────────────────────────

class BeliefModel(nn.Module):
    """
    Trainable belief update module.

    Input vector per step:
        [hidden_pool (H), belief_vec (belief_dim), role_emb (role_emb_dim)]

    Output:
        updated_beliefs : [B, belief_dim]  — values ∈ [0, 1]

    The model is trained via supervised MSE loss against rule-derived
    ground-truth belief targets during warmup, then jointly with PPO loss.
    """

    NUM_ROLES = 2  # index 0 = Crewmate, 1 = Impostor

    def __init__(self, config: Optional[BeliefModelConfig] = None):
        super().__init__()
        self.config = config or BeliefModelConfig()
        cfg = self.config

        # Role embedding: learnable token per role (Crewmate / Impostor)
        self.role_embedding = nn.Embedding(self.NUM_ROLES, cfg.role_emb_dim)

        # Input size: hidden pool + current belief vector + role embedding
        input_size = cfg.hidden_size + cfg.belief_dim + cfg.role_emb_dim

        # Shared trunk (residual MLP)
        self.fc1 = nn.Linear(input_size, cfg.mlp_hidden)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layer_norm = nn.LayerNorm(cfg.mlp_hidden)

        # Residual projection from input to trunk dim
        self.input_proj = nn.Linear(input_size, cfg.mlp_hidden)

        # Separate output heads for each role
        # Crewmate head: suspicion_matrix
        self.crew_head = nn.Linear(cfg.mlp_hidden, cfg.belief_dim)
        # Impostor head: second_order_beliefs
        self.imp_head = nn.Linear(cfg.mlp_hidden, cfg.belief_dim)

    def forward(
        self,
        hidden_pool: torch.Tensor,          # [B, hidden_size]
        current_beliefs: torch.Tensor,      # [B, belief_dim]
        role_ids: torch.Tensor,             # [B]  — 0=Crewmate, 1=Impostor
    ) -> torch.Tensor:
        """
        Forward pass: predict updated belief vector.

        Returns
        -------
        updated_beliefs : [B, belief_dim]  — values in [0, 1] via sigmoid
        """
        B = hidden_pool.shape[0]

        # Role embedding
        r_emb = self.role_embedding(role_ids)              # [B, role_emb_dim]

        # Concatenate inputs
        x = torch.cat([hidden_pool, current_beliefs, r_emb], dim=-1)  # [B, input_size]

        # Residual MLP
        res = self.input_proj(x)                           # [B, mlp_hidden]
        h = F.relu(self.fc1(x))                            # [B, mlp_hidden]
        h = self.dropout(h)
        h = F.relu(self.fc2(h))                            # [B, mlp_hidden]
        h = self.layer_norm(h + res)                       # residual connection

        # Route to role-specific head
        # We compute both but select per sample using role_ids
        crew_out = self.crew_head(h)                       # [B, belief_dim]
        imp_out = self.imp_head(h)                         # [B, belief_dim]

        # Select output per sample based on role
        is_impostor = (role_ids == 1).float().unsqueeze(-1)  # [B, 1]
        raw = (1 - is_impostor) * crew_out + is_impostor * imp_out  # [B, belief_dim]
        return torch.sigmoid(raw)                          # ∈ [0, 1]

    def loss(
        self,
        predicted: torch.Tensor,   # [B, belief_dim]
        targets: torch.Tensor,     # [B, belief_dim]
        mask: Optional[torch.Tensor] = None,  # [B, belief_dim] — 0=padded slot
    ) -> torch.Tensor:
        """
        MSE loss between predicted and target belief vectors.
        Optionally masked to exclude padded player slots.
        """
        l = F.mse_loss(predicted, targets, reduction="none")   # [B, belief_dim]
        if mask is not None:
            l = l * mask
            return l.sum() / mask.sum().clamp(min=1)
        return l.mean()


# ── Belief Target Builder ─────────────────────────────────────────────────

BELIEF_RULES = {
    # action → (multiplier or absolute value, is_absolute)
    "KILL":          (1.00, True),     # hard evidence → set to 1.0
    "VENT":          (1.00, True),
    "SABOTAGE":      (1.25, False),    # ×1.25
    "FAKE_TASK":     (1.10, False),    # ×1.10
    "VISUAL_TASK":   (0.90, False),    # ×0.90 (exculpatory)
    "COMPLETE_TASK": (0.90, False),
}


def build_belief_targets(
    observation_log: List[Dict],
    current_beliefs: Dict[str, float],
    player_name: str,
) -> Dict[str, float]:
    """
    Build ground-truth belief targets from an observation log,
    mirroring the rule logic in ActorModule.update_beliefs().

    Parameters
    ----------
    observation_log : List[Dict]
        Each entry: {"subject": str, "action": str, "witnesses": List[str]}
    current_beliefs : Dict[str, float]
        Current belief vector {player_name: value}
    player_name : str
        The observing player (skip self-observations)

    Returns
    -------
    Dict[str, float]  — updated belief targets
    """
    updated = dict(current_beliefs)
    for event in observation_log:
        subject = event.get("subject", "")
        action = event.get("action", "")
        if subject == player_name:
            continue
        if subject not in updated:
            continue
        rule = BELIEF_RULES.get(action)
        if rule is None:
            continue
        value, is_absolute = rule
        cur = updated[subject]
        if is_absolute:
            updated[subject] = value
        else:
            updated[subject] = max(0.0, min(1.0, cur * value))
    return updated


def beliefs_to_tensor(
    beliefs: Dict[str, float],
    player_order: List[str],
    belief_dim: int,
) -> torch.Tensor:
    """
    Convert a {player_name: float} belief dict to a fixed-dim tensor.

    Players listed in *player_order* are placed at their index.
    Unused slots are padded with 0.5 (neutral).

    Returns
    -------
    tensor : [belief_dim]
    """
    vec = torch.full((belief_dim,), 0.5)
    for i, name in enumerate(player_order):
        if i >= belief_dim:
            break
        vec[i] = beliefs.get(name, 0.5)
    return vec
