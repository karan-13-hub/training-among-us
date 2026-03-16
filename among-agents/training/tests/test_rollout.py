"""
test_rollout.py — Unit tests for RolloutCollector (CPU-safe, mock objects)

Tests:
  - Trajectory dataclass and GAE computation
  - beliefs_to_tensor / build_belief_targets helpers
  - extract_game_features

Run with:
    cd among-agents
    python -m pytest training/tests/test_rollout.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import torch
import math

from training.rollout_collector import Trajectory, StepData
from training.critic_model import extract_game_features, heuristic_value_from_state


# ── Helpers ───────────────────────────────────────────────────────────────

def make_step(t: int, reward: float, value: float, belief_dim: int = 4) -> StepData:
    """Create a minimal StepData for testing."""
    return StepData(
        timestep=t,
        player_name="Player 1",
        role="Crewmate",
        is_impostor=False,
        prompt="",
        action_text="MOVE",
        input_ids=torch.zeros(1, 5, dtype=torch.long),
        action_ids=torch.zeros(1, 2, dtype=torch.long),
        log_prob_old=-1.0,
        hidden_pool=torch.randn(64),
        belief_vec=torch.rand(belief_dim),
        belief_target=torch.rand(belief_dim),
        value=value,
        game_feat=torch.zeros(5),
        reward=reward,
        game_state={},
    )


# ── GAE Tests ─────────────────────────────────────────────────────────────

class TestGAE:
    def test_gae_shape(self):
        traj = Trajectory("Player 1", "Crewmate")
        for t in range(5):
            traj.steps.append(make_step(t, reward=1.0, value=0.5))
        traj.compute_gae(gamma=0.99, gae_lambda=0.95, final_value=0.0)
        assert traj.advantages.shape == (5,)
        assert traj.returns.shape    == (5,)

    def test_gae_terminal_zero_bootstrap(self):
        """With final_value=0 and reward=0, advantages should be −V(s)."""
        traj = Trajectory("Player 1", "Crewmate")
        # Single step: r=0, V=0.5 → δ = 0 + 0.99*0 - 0.5 = -0.5
        traj.steps.append(make_step(0, reward=0.0, value=0.5))
        traj.compute_gae(gamma=0.99, gae_lambda=0.95, final_value=0.0)
        assert abs(traj.advantages[0].item() - (-0.5)) < 1e-4

    def test_returns_equal_advantages_plus_values(self):
        """Consistency: returns_t = advantages_t + values_t."""
        traj = Trajectory("Player 1", "Impostor")
        for t in range(6):
            traj.steps.append(make_step(t, reward=float(t), value=0.5))
        traj.compute_gae(gamma=0.99, gae_lambda=0.95)
        values = traj.values_tensor
        diff = (traj.returns - (traj.advantages + values)).abs()
        assert diff.max() < 1e-5, f"returns ≠ advantages + values, max diff={diff.max()}"

    def test_empty_trajectory(self):
        """Empty trajectory should produce zero-length tensors without error."""
        traj = Trajectory("Player 1", "Crewmate")
        traj.compute_gae()
        assert traj.advantages.shape == (0,)
        assert traj.returns.shape    == (0,)

    def test_gae_with_positive_rewards(self):
        """Positive rewards should produce positive advantages."""
        traj = Trajectory("Player 1", "Crewmate")
        for t in range(4):
            traj.steps.append(make_step(t, reward=1.0, value=0.0))
        traj.compute_gae(gamma=0.99, gae_lambda=0.95, final_value=0.0)
        assert (traj.advantages > 0).all(), "All advantages should be positive for r=1, V=0"


# ── Trajectory Property Tests ─────────────────────────────────────────────

class TestTrajectoryProperties:
    def setup_method(self):
        self.traj = Trajectory("Player 1", "Crewmate")
        for t in range(3):
            self.traj.steps.append(make_step(t, 1.0, 0.5, belief_dim=4))

    def test_log_probs_old(self):
        lp = self.traj.log_probs_old
        assert lp.shape == (3,)
        assert (lp == -1.0).all()

    def test_hidden_pools(self):
        hp = self.traj.hidden_pools
        assert hp.shape == (3, 64)

    def test_belief_vecs(self):
        bv = self.traj.belief_vecs
        assert bv.shape == (3, 4)

    def test_belief_targets(self):
        bt = self.traj.belief_targets
        assert bt.shape == (3, 4)

    def test_game_feats(self):
        gf = self.traj.game_feats
        assert gf.shape == (3, 5)

    def test_values_tensor(self):
        vt = self.traj.values_tensor
        assert vt.shape == (3,)
        assert (vt == 0.5).all()


# ── extract_game_features Tests ───────────────────────────────────────────

class TestExtractGameFeatures:
    def test_shape(self):
        gs = {"living_crewmates": 3, "living_impostors": 1,
              "task_completion_pct": 50.0, "sabotage_active": False}
        f = extract_game_features(gs, is_impostor=False)
        assert f.shape == (5,)

    def test_normalisation(self):
        gs = {"living_crewmates": 10, "living_impostors": 3,
              "task_completion_pct": 100.0, "sabotage_active": True}
        f = extract_game_features(gs, is_impostor=True, max_crew=10, max_imps=3)
        assert abs(f[0].item() - 1.0) < 1e-5  # crew normalised
        assert abs(f[1].item() - 1.0) < 1e-5  # imps normalised
        assert abs(f[2].item() - 1.0) < 1e-5  # task normalised
        assert abs(f[3].item() - 1.0) < 1e-5  # sabotage active
        assert abs(f[4].item() - 1.0) < 1e-5  # is_impostor

    def test_crewmate_role_flag(self):
        gs = {"living_crewmates": 3, "living_impostors": 1,
              "task_completion_pct": 0.0, "sabotage_active": False}
        f = extract_game_features(gs, is_impostor=False)
        assert abs(f[4].item() - 0.0) < 1e-5  # not impostor

    def test_missing_keys_default_zero(self):
        """Should not crash on missing keys — defaults to 0."""
        f = extract_game_features({}, is_impostor=False)
        assert f.shape == (5,)


# ── Heuristic Value Tests ─────────────────────────────────────────────────

class TestHeuristicValue:
    def test_impostor_wins_crew_outnumbered(self):
        gs = {"living_crewmates": 1, "living_impostors": 1,
              "task_completion_pct": 0.0, "sabotage_active": False}
        v = heuristic_value_from_state(gs, is_impostor=True)
        assert v > 0.5, f"Impostor should win when numbers equal: {v}"

    def test_crew_wins_all_tasks_done(self):
        gs = {"living_crewmates": 4, "living_impostors": 1,
              "task_completion_pct": 100.0, "sabotage_active": False}
        v = heuristic_value_from_state(gs, is_impostor=False)
        assert v == 1.0

    def test_impostor_win_prob_plus_crew_win_prob_approx_one(self):
        """For a symmetric state, crew + imp ≈ 1.0 (game is zero-sum)."""
        gs = {"living_crewmates": 3, "living_impostors": 1,
              "task_completion_pct": 50.0, "sabotage_active": False}
        vc = heuristic_value_from_state(gs, is_impostor=False)
        vi = heuristic_value_from_state(gs, is_impostor=True)
        assert abs(vc + vi - 1.0) < 1e-5

    def test_value_in_range(self):
        import random
        for _ in range(50):
            gs = {
                "living_crewmates":    random.randint(0, 8),
                "living_impostors":    random.randint(0, 3),
                "task_completion_pct": random.uniform(0, 100),
                "sabotage_active":     random.random() < 0.3,
            }
            v = heuristic_value_from_state(gs, is_impostor=random.choice([True, False]))
            assert 0.0 <= v <= 1.0, f"V out of range: {v}"
