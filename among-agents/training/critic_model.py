"""
critic_model.py — Trainable Critic (Value Function)

Implements V(s) ∈ [0, 1] — estimated win probability from a game state.

Architecture:
    [hidden_pool (H), game_features (5)] → MLP → scalar

Replaces the heuristic in CriticModule when injected via set_critic_model().
Trained during warmup via supervised MSE on heuristic V(s) labels, then
jointly fine-tuned with PPO value loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, Optional


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class CriticModelConfig:
    hidden_size: int = 2048     # backbone hidden_size from Qwen3-4B
    game_feature_dim: int = 5   # [crew, imps, task%, sabotage, is_impostor]
    mlp_hidden_1: int = 256
    mlp_hidden_2: int = 128
    dropout: float = 0.1


# ── Game Feature Extractor ────────────────────────────────────────────────

def extract_game_features(
    game_state: Dict[str, Any],
    is_impostor: bool,
    max_crew: int = 10,
    max_imps: int = 3,
) -> torch.Tensor:
    """
    Convert a game_state dict to a normalised 5-dim feature vector.

    game_state keys expected:
        living_crewmates, living_impostors, task_completion_pct,
        sabotage_active, winner (optional)

    Returns
    -------
    features : [5]  (float32 on CPU — move to device before use)
    """
    crew = game_state.get("living_crewmates", 0) / max_crew
    imps = game_state.get("living_impostors", 0) / max_imps
    task = game_state.get("task_completion_pct", 0.0) / 100.0
    sab  = float(bool(game_state.get("sabotage_active", False)))
    role = float(is_impostor)
    return torch.tensor([crew, imps, task, sab, role], dtype=torch.float32)


# ── Critic Model ──────────────────────────────────────────────────────────

class CriticModel(nn.Module):
    """
    Trainable value / critic head.

    Takes as input:
        - hidden_pool : [B, hidden_size]  — pooled backbone hidden states
        - game_feat   : [B, 5]            — normalised game feature vector

    Returns:
        value : [B]  ∈ [0, 1]  (win probability for the player's team)
    """

    def __init__(self, config: Optional[CriticModelConfig] = None):
        super().__init__()
        self.config = config or CriticModelConfig()
        cfg = self.config

        input_dim = cfg.hidden_size + cfg.game_feature_dim

        self.fc1 = nn.Linear(input_dim, cfg.mlp_hidden_1)
        self.fc2 = nn.Linear(cfg.mlp_hidden_1, cfg.mlp_hidden_2)
        self.out  = nn.Linear(cfg.mlp_hidden_2, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layer_norm = nn.LayerNorm(cfg.mlp_hidden_1)

        # Residual path
        self.residual_proj = nn.Linear(input_dim, cfg.mlp_hidden_1)

    def forward(
        self,
        hidden_pool: torch.Tensor,    # [B, hidden_size]
        game_feat:   torch.Tensor,    # [B, game_feature_dim]
    ) -> torch.Tensor:
        """
        Returns
        -------
        value : [B]  ∈ [0, 1]
        """
        x = torch.cat([hidden_pool, game_feat], dim=-1)   # [B, H+5]

        res = self.residual_proj(x)                        # [B, mlp_hidden_1]
        h = F.relu(self.fc1(x))                            # [B, mlp_hidden_1]
        h = self.dropout(h)
        h = self.layer_norm(h + res)                       # residual
        h = F.relu(self.fc2(h))                            # [B, mlp_hidden_2]
        h = self.dropout(h)
        value = torch.sigmoid(self.out(h)).squeeze(-1)     # [B]
        return value

    def loss(
        self,
        predicted: torch.Tensor,   # [B]
        targets:   torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """MSE loss clipped at [0,1] targets."""
        targets = targets.clamp(0.0, 1.0)
        return F.mse_loss(predicted, targets)


# ── Heuristic Critic (Fallback / Warmup Label Source) ────────────────────
# These are re-exported from the existing critic.py logic so warmup.py
# can call them without importing the full CriticModule.

def heuristic_crew_value(
    crew: int, imps: int, task_pct: float, sabotage: bool
) -> float:
    """Deterministic Crewmate win-probability heuristic (mirrors critic.py)."""
    if imps == 0:
        return 1.0
    if crew <= imps:
        return 0.0
    if task_pct >= 100.0:
        return 1.0
    task_factor = (task_pct / 100.0) * 0.5
    safe_margin = (crew - imps) / max(crew + imps, 1)
    numbers_factor = safe_margin * 0.4
    sab_penalty = 0.1 if sabotage else 0.0
    value = 0.1 + task_factor + numbers_factor - sab_penalty
    return float(max(0.0, min(1.0, value)))


def heuristic_imp_value(
    crew: int, imps: int, task_pct: float, sabotage: bool
) -> float:
    """Deterministic Impostor win-probability heuristic (mirrors critic.py)."""
    return round(1.0 - heuristic_crew_value(crew, imps, task_pct, sabotage), 4)


def heuristic_value_from_state(
    game_state: Dict[str, Any],
    is_impostor: bool,
) -> float:
    """
    Compute heuristic V(s) for warmup supervision.

    Parameters
    ----------
    game_state : dict with keys living_crewmates, living_impostors,
                 task_completion_pct, sabotage_active
    is_impostor : bool

    Returns
    -------
    float  ∈ [0, 1]
    """
    crew = game_state.get("living_crewmates", 0)
    imps = game_state.get("living_impostors", 0)
    task = game_state.get("task_completion_pct", 0.0)
    sab  = bool(game_state.get("sabotage_active", False))
    if is_impostor:
        return heuristic_imp_value(crew, imps, task, sab)
    return heuristic_crew_value(crew, imps, task, sab)
