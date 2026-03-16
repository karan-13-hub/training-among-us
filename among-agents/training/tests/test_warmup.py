"""
test_warmup.py — Unit tests for WarmupTrainer (CPU-safe, no GPU required)

Run with:
    cd among-agents
    python -m pytest training/tests/test_warmup.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import torch

from training.belief_model import BeliefModel, BeliefModelConfig
from training.critic_model import CriticModel, CriticModelConfig
from training.warmup import (
    WarmupTrainer,
    WarmupConfig,
    collect_synthetic_belief_data,
    collect_synthetic_critic_data,
    BeliefDataset,
    CriticDataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_config():
    return WarmupConfig(
        warmup_games=2,
        belief_epochs=2,
        critic_epochs=2,
        belief_batch_size=16,
        critic_batch_size=16,
        belief_lr=1e-3,
        critic_lr=1e-3,
        device="cpu",
        log_every=100,
    )


@pytest.fixture
def small_belief_model():
    cfg = BeliefModelConfig(hidden_size=64, belief_dim=4, mlp_hidden=32, role_emb_dim=4)
    return BeliefModel(cfg)


@pytest.fixture
def small_critic_model():
    cfg = CriticModelConfig(hidden_size=64, mlp_hidden_1=32, mlp_hidden_2=16)
    return CriticModel(cfg)


@pytest.fixture
def trainer(small_belief_model, small_critic_model, tiny_config):
    return WarmupTrainer(small_belief_model, small_critic_model, tiny_config)


# ── Synthetic Dataset Tests ───────────────────────────────────────────────

class TestSyntheticDatasets:
    def test_belief_dataset_shapes(self):
        ds = collect_synthetic_belief_data(n_samples=50, hidden_size=64, belief_dim=4)
        assert len(ds) == 50
        hp, bv, rid, tgt = ds[0]
        assert hp.shape == (64,)
        assert bv.shape == (4,)
        assert rid.dtype == torch.long
        assert tgt.shape == (4,)

    def test_belief_dataset_targets_in_range(self):
        ds = collect_synthetic_belief_data(n_samples=100, hidden_size=64, belief_dim=4)
        for i in range(len(ds)):
            _, _, _, tgt = ds[i]
            assert tgt.min() >= 0.0
            assert tgt.max() <= 1.0

    def test_critic_dataset_shapes(self):
        ds = collect_synthetic_critic_data(n_samples=50, hidden_size=64)
        assert len(ds) == 50
        hp, gf, v = ds[0]
        assert hp.shape == (64,)
        assert gf.shape == (5,)
        assert 0.0 <= v.item() <= 1.0

    def test_critic_values_in_range(self):
        ds = collect_synthetic_critic_data(n_samples=100, hidden_size=64)
        for i in range(len(ds)):
            _, _, v = ds[i]
            assert 0.0 <= v.item() <= 1.0


# ── WarmupTrainer Tests ───────────────────────────────────────────────────

class TestWarmupTrainer:
    def test_belief_loss_decreases(self, trainer, tiny_config):
        """Belief model loss should decrease over the warmup epochs."""
        ds = collect_synthetic_belief_data(
            n_samples=200, hidden_size=64, belief_dim=4)
        losses = trainer.train_belief(ds)
        assert len(losses) == tiny_config.belief_epochs
        # With 2 epochs and lr=1e-3, loss should generally decrease
        # (may occasionally fail by chance with tiny datasets, but very rarely)
        assert losses[-1] <= losses[0] + 0.05, \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_critic_loss_decreases(self, trainer, tiny_config):
        """Critic model loss should decrease over the warmup epochs."""
        ds = collect_synthetic_critic_data(n_samples=200, hidden_size=64)
        losses = trainer.train_critic(ds)
        assert len(losses) == tiny_config.critic_epochs
        assert losses[-1] <= losses[0] + 0.05, \
            f"Critic loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_run_full_warmup(self, trainer):
        """Full WarmupTrainer.run() should complete without errors."""
        trainer.run()  # uses synthetic data internally
        assert len(trainer.belief_history) > 0
        assert len(trainer.critic_history) > 0

    def test_save_load_checkpoint(self, trainer, tmp_path, small_belief_model, small_critic_model):
        """Checkpoint round-trip should preserve weights."""
        trainer.run()  # train briefly to get non-random weights

        ckpt_dir = str(tmp_path / "warmup_ckpt")
        trainer.save_warmup_checkpoint(ckpt_dir)

        # Create new trainer and load
        new_trainer = WarmupTrainer(
            BeliefModel(BeliefModelConfig(hidden_size=64, belief_dim=4, mlp_hidden=32, role_emb_dim=4)),
            CriticModel(CriticModelConfig(hidden_size=64, mlp_hidden_1=32, mlp_hidden_2=16)),
            WarmupConfig(device="cpu"),
        )
        new_trainer.load_warmup_checkpoint(ckpt_dir)

        # Verify belief model weights match
        for (name, p_orig), (_, p_loaded) in zip(
            trainer.belief_model.named_parameters(),
            new_trainer.belief_model.named_parameters()
        ):
            assert torch.allclose(p_orig, p_loaded), f"Mismatch in {name}"

    def test_history_accumulates(self, trainer):
        """history lists should grow with each call."""
        ds_b = collect_synthetic_belief_data(n_samples=50, hidden_size=64, belief_dim=4)
        ds_c = collect_synthetic_critic_data(n_samples=50, hidden_size=64)

        trainer.train_belief(ds_b)
        trainer.train_critic(ds_c)
        trainer.train_belief(ds_b)

        assert len(trainer.belief_history) == 4   # 2+2 epochs
        assert len(trainer.critic_history) == 2   # 2 epochs
