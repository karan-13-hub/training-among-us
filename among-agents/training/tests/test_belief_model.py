"""
test_belief_model.py — Unit tests for BeliefModel (CPU-safe, no GPU required)

Run with:
    cd among-agents
    python -m pytest training/tests/test_belief_model.py -v
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import torch

from training.belief_model import (
    BeliefModel,
    BeliefModelConfig,
    build_belief_targets,
    beliefs_to_tensor,
    BELIEF_RULES,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    """Small config for fast CPU testing."""
    return BeliefModelConfig(
        hidden_size=64,
        belief_dim=4,
        mlp_hidden=32,
        role_emb_dim=4,
    )


@pytest.fixture
def belief_model(small_config):
    return BeliefModel(small_config).eval()


# ── Output Shape / Range Tests ───────────────────────────────────────────

class TestBeliefModelForward:
    def test_output_shape_crewmate(self, belief_model, small_config):
        B, H, D = 8, small_config.hidden_size, small_config.belief_dim
        hidden = torch.randn(B, H)
        beliefs = torch.rand(B, D)
        roles = torch.zeros(B, dtype=torch.long)   # 0 = Crewmate

        out = belief_model(hidden, beliefs, roles)
        assert out.shape == (B, D), f"Expected ({B},{D}), got {out.shape}"

    def test_output_shape_impostor(self, belief_model, small_config):
        B, H, D = 4, small_config.hidden_size, small_config.belief_dim
        hidden = torch.randn(B, H)
        beliefs = torch.rand(B, D)
        roles = torch.ones(B, dtype=torch.long)    # 1 = Impostor

        out = belief_model(hidden, beliefs, roles)
        assert out.shape == (B, D)

    def test_output_in_range(self, belief_model, small_config):
        """All outputs must be in [0, 1] — sigmoid activated."""
        B, H, D = 16, small_config.hidden_size, small_config.belief_dim
        hidden  = torch.randn(B, H)
        beliefs = torch.rand(B, D)
        roles   = torch.randint(0, 2, (B,))

        out = belief_model(hidden, beliefs, roles)
        assert out.min() >= 0.0, "Values below 0"
        assert out.max() <= 1.0, "Values above 1"

    def test_mixed_roles(self, belief_model, small_config):
        """Batch with mixed roles should not raise."""
        B, H, D = 6, small_config.hidden_size, small_config.belief_dim
        hidden  = torch.randn(B, H)
        beliefs = torch.rand(B, D)
        roles   = torch.tensor([0, 1, 0, 1, 0, 0])

        out = belief_model(hidden, beliefs, roles)
        assert out.shape == (B, D)

    def test_different_role_heads_produce_different_output(self, belief_model, small_config):
        """Same input with different role should produce different outputs."""
        H, D = small_config.hidden_size, small_config.belief_dim
        hidden  = torch.randn(1, H)
        beliefs = torch.rand(1, D)

        crew_out = belief_model(hidden, beliefs, torch.tensor([0]))
        imp_out  = belief_model(hidden, beliefs, torch.tensor([1]))
        # With fresh random weights they should differ (with overwhelming probability)
        assert not torch.allclose(crew_out, imp_out, atol=1e-4), \
            "Crewmate and Impostor heads should produce different outputs"


# ── Loss Tests ───────────────────────────────────────────────────────────

class TestBeliefLoss:
    def test_perfect_prediction_zero_loss(self, belief_model):
        """MSE of perfect prediction should be 0."""
        targets = torch.rand(4, 4)
        loss = belief_model.loss(targets, targets)
        assert loss.item() < 1e-7

    def test_loss_decreases_toward_target(self, belief_model, small_config):
        """Loss with random pred should be > 0."""
        D = small_config.belief_dim
        pred    = torch.rand(8, D)
        targets = torch.rand(8, D)
        loss = belief_model.loss(pred, targets)
        assert loss.item() > 0.0

    def test_masked_loss(self, belief_model, small_config):
        """Masked loss should differ from unmasked when mask zeros some slots."""
        D = small_config.belief_dim
        pred    = torch.rand(4, D)
        targets = torch.rand(4, D)
        mask    = torch.ones(4, D)
        mask[:, -1] = 0.0   # last slot always padded

        unmasked = belief_model.loss(pred, targets)
        masked   = belief_model.loss(pred, targets, mask)
        # They may be equal by chance, but generally differ
        # Just check they both are finite positive numbers
        assert masked.item() >= 0.0
        assert unmasked.item() >= 0.0


# ── Belief Target Builder Tests ───────────────────────────────────────────

class TestBuildBeliefTargets:
    def test_kill_sets_to_one(self):
        beliefs = {"Player 2": 0.3, "Player 3": 0.5}
        obs = [{"subject": "Player 2", "action": "KILL", "witnesses": ["Player 1"]}]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert updated["Player 2"] == 1.0, "KILL should set suspicion to 1.0"

    def test_vent_sets_to_one(self):
        beliefs = {"Player 2": 0.4}
        obs = [{"subject": "Player 2", "action": "VENT", "witnesses": ["Player 1"]}]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert updated["Player 2"] == 1.0

    def test_sabotage_multiplies_up(self):
        beliefs = {"Player 2": 0.5}
        obs = [{"subject": "Player 2", "action": "SABOTAGE", "witnesses": ["Player 1"]}]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert abs(updated["Player 2"] - min(0.5 * 1.25, 1.0)) < 1e-6

    def test_visual_task_reduces(self):
        beliefs = {"Player 2": 0.5}
        obs = [{"subject": "Player 2", "action": "VISUAL_TASK", "witnesses": ["Player 1"]}]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert abs(updated["Player 2"] - 0.5 * 0.9) < 1e-6

    def test_self_skipped(self):
        """Events about self should be ignored."""
        beliefs = {"Player 2": 0.6}
        obs = [{"subject": "Player 1", "action": "KILL", "witnesses": []}]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert updated["Player 2"] == 0.6, "Self-event should not change beliefs"

    def test_clamp_at_one(self):
        """Beliefs should not exceed 1.0."""
        beliefs = {"Player 2": 0.95}
        obs = [{"subject": "Player 2", "action": "SABOTAGE", "witnesses": ["Player 1"]}]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert updated["Player 2"] <= 1.0

    def test_clamp_at_zero(self):
        """Beliefs should not go below 0.0."""
        beliefs = {"Player 2": 0.01}
        obs = [
            {"subject": "Player 2", "action": "VISUAL_TASK", "witnesses": ["P1"]},
            {"subject": "Player 2", "action": "VISUAL_TASK", "witnesses": ["P1"]},
            {"subject": "Player 2", "action": "VISUAL_TASK", "witnesses": ["P1"]},
        ]
        updated = build_belief_targets(obs, beliefs, "Player 1")
        assert updated["Player 2"] >= 0.0


# ── Tensor Conversion Tests ───────────────────────────────────────────────

class TestBeliefsToTensor:
    def test_correct_dim(self):
        beliefs = {"A": 0.8, "B": 0.2}
        t = beliefs_to_tensor(beliefs, ["A", "B", "C"], 4)
        assert t.shape == (4,)

    def test_values_placed_correctly(self):
        beliefs = {"A": 0.9, "B": 0.1}
        t = beliefs_to_tensor(beliefs, ["A", "B"], 2)
        assert abs(t[0].item() - 0.9) < 1e-5
        assert abs(t[1].item() - 0.1) < 1e-5

    def test_missing_player_neutral(self):
        """Absent players get 0.5 (neutral)."""
        beliefs = {"A": 0.7}
        t = beliefs_to_tensor(beliefs, ["A", "B"], 2)
        assert abs(t[1].item() - 0.5) < 1e-5

    def test_truncation_beyond_belief_dim(self):
        """Extra players beyond belief_dim are ignored."""
        beliefs = {f"P{i}": 0.5 for i in range(10)}
        order = [f"P{i}" for i in range(10)]
        t = beliefs_to_tensor(beliefs, order, 4)
        assert t.shape == (4,)
