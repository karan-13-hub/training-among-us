"""
train.py — Config-Driven RL Training Entry Point

Runs the full Among Us RL training pipeline:
    1. Warmup  — supervised pre-training of BeliefModel + CriticModel
    2. PPO     — joint optimisation of policy (LoRA) + auxiliary heads

Usage
-----
    # Single-GPU, default config
    python training/train.py

    # Single-GPU, custom YAML config
    python training/train.py --config training/train_config.yml

    # Multi-GPU (4 GPUs on one node)
    torchrun --nproc_per_node=4 training/train.py --config training/train_config.yml

    # Smoke test
    python training/train.py --smoke-test

    # Resume from PPO checkpoint
    python training/train.py --config training/train_config.yml --resume checkpoints/iter_0010

    # Skip warmup (load existing checkpoint)
    python training/train.py --config training/train_config.yml --warmup-checkpoint checkpoints/warmup

Multi-GPU notes
---------------
* Uses PyTorch DistributedDataParallel (DDP) via ``torchrun``.
* Each rank collects its own games in parallel with other ranks.
* Gradients are all-reduced across ranks after every PPO mini-batch.
* Checkpoints and metric logs are written only by rank 0.
* The policy base model (Qwen3-4B) is loaded on each rank's local GPU.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(SCRIPT_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Master Config Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Model ─────────────────────────────────────────────
    model_path: str   = "/data/kmirakho/verl/models/Qwen3-4B-Instruct-2507"

    # Comma-separated GPU indices.  Examples:
    #   "0"       → single GPU 0
    #   "0,1,3"   → 3 GPUs (indices 0, 1, and 3)
    # train.py sets CUDA_VISIBLE_DEVICES from this string.
    # Under torchrun each rank gets LOCAL_RANK 0…N-1 which maps
    # through the visible-device list to the actual GPU.
    cuda_devices: str = "0"

    # ── LoRA ──────────────────────────────────────────────
    lora_r: int             = 16
    lora_alpha: int         = 32
    lora_dropout: float     = 0.05
    lora_target_modules: Optional[List[str]] = None  # None → PEFT default

    # ── Warmup ────────────────────────────────────────────
    warmup_games: int           = 20
    warmup_belief_epochs: int   = 5
    warmup_critic_epochs: int   = 5
    warmup_belief_lr: float     = 1e-4
    warmup_critic_lr: float     = 1e-4
    warmup_batch_size: int      = 64
    warmup_synthetic_scale: int = 200

    # ── PPO ───────────────────────────────────────────────
    num_iterations: int     = 100
    games_per_iter: int     = 5
    ppo_epochs: int         = 4
    ppo_lr_policy: float    = 1e-5
    ppo_lr_heads: float     = 3e-4
    clip_eps: float         = 0.2
    gamma: float            = 0.99
    gae_lambda: float       = 0.95
    entropy_coef: float     = 0.01
    value_loss_coef: float  = 0.5
    belief_loss_coef: float = 0.3
    max_grad_norm: float    = 0.5
    target_kl: float        = 0.02
    mini_batch_size: int    = 16    # per-GPU batch size

    # ── Game ──────────────────────────────────────────────
    game_config_name: str   = "FIVE_MEMBER_GAME"
    vllm_port: int          = 8234
    max_players: int        = 9
    belief_dim: int         = 9

    # ── Multi-GPU ─────────────────────────────────────────
    ddp_backend: str        = "nccl"    # "nccl" for GPU, "gloo" for CPU/debug
    games_parallel: int     = 1         # game-collection parallelism per rank

    # ── Checkpointing ────────────────────────────────────────────────
    checkpoint_dir: str            = "./checkpoints"
    crewmate_checkpoint_dir: str   = "./checkpoints/crewmate"
    impostor_checkpoint_dir: str   = "./checkpoints/impostor"
    checkpoint_every: int          = 10
    log_dir: str                   = "./logs"

    # ── Alternating Training ───────────────────────────────────────────
    # Number of PPO iterations per phase before switching roles.
    # e.g. alternate_every=5 → train Crewmate 5 iters, Impostor 5 iters, …
    alternate_every: int = 5

    # ── Weights & Biases ─────────────────────────────────────────────────
    wandb_enabled: bool         = True
    wandb_project: str          = "among-us-rl"
    wandb_entity: Optional[str] = None   # your W&B username / team  (None = W&B default)
    wandb_run_name: Optional[str] = None # None = auto-generated by W&B
    wandb_tags: List[str]       = field(default_factory=list)
    wandb_watch_models: bool    = False  # log model gradients & params (slow)

    # ── Misc ────────────────────────────────────────────────────────────
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# YAML Config Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config_from_yaml(yaml_path: str, base: Optional[TrainConfig] = None) -> TrainConfig:
    """
    Load a TrainConfig from a YAML file, merging over *base* defaults.

    Unknown keys in the YAML are silently ignored so that config files can
    contain comments / documentation keys without breaking older code.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load YAML configs.  "
            "Install with:  pip install pyyaml"
        )

    cfg = deepcopy(base) if base is not None else TrainConfig()

    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh) or {}

    known = set(asdict(cfg).keys())
    for key, value in raw.items():
        if key not in known:
            logger.warning(f"[config] Unknown key '{key}' in {yaml_path} — ignoring")
            continue
        # YAML null → Python None
        if value == "null" or value is None:
            value = None
        setattr(cfg, key, value)

    return cfg


def save_config_to_yaml(cfg: TrainConfig, path: str) -> None:
    """Persist the resolved config alongside the run for reproducibility."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping config save.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(asdict(cfg), fh, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Distributed (DDP) Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_ddp() -> bool:
    """True when launched via torchrun / torch.distributed."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def _ddp_setup(backend: str) -> tuple:
    """
    Initialise the process group and return (rank, local_rank, world_size).
    Called only when WORLD_SIZE > 1.
    """
    import torch.distributed as dist
    dist.init_process_group(backend=backend)
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def _ddp_cleanup():
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_wrap(module, device_id: int, find_unused: bool = False):
    """Wrap *module* in DDP. Returns the wrapped module."""
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(module.to(device_id), device_ids=[device_id],
               find_unused_parameters=find_unused)


# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_game_factory(config: TrainConfig):
    """Return a callable() → AmongUs with the right settings."""
    from amongagents.envs.game import AmongUs
    from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME

    game_cfg = (FIVE_MEMBER_GAME
                if config.game_config_name == "FIVE_MEMBER_GAME"
                else SEVEN_MEMBER_GAME)

    agent_config = {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "CREWMATE_LLM_CHOICES": [config.model_path],
        "IMPOSTOR_LLM_CHOICES": [config.model_path],
    }

    if "EXPERIMENT_PATH" not in os.environ:
        os.makedirs(config.log_dir, exist_ok=True)
        os.environ["EXPERIMENT_PATH"] = config.log_dir

    _counter = [0]

    def factory():
        game = AmongUs(
            game_config=game_cfg,
            agent_config=agent_config,
            game_index=_counter[0],
        )
        _counter[0] += 1
        return game

    return factory


def build_lora_policy(config: TrainConfig, device: str):
    from training.lora_model import LoRAQwenPolicy, LoRAConfig
    lora_kwargs = dict(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    # Only override target_modules when explicitly set; otherwise use LoRAConfig default.
    if config.lora_target_modules is not None:
        lora_kwargs["target_modules"] = config.lora_target_modules
    lora_cfg = LoRAConfig(**lora_kwargs)
    return LoRAQwenPolicy(
        model_path=config.model_path,
        lora_config=lora_cfg,
        device=device,
        roles=["Crewmate", "Impostor"],   # dual named LoRA adapters
    )


def build_belief_model(config: TrainConfig, hidden_size: int, device: str):
    from training.belief_model import BeliefModel, BeliefModelConfig
    bcfg = BeliefModelConfig(hidden_size=hidden_size, belief_dim=config.belief_dim)
    return BeliefModel(bcfg).to(device)


def build_critic_model(config: TrainConfig, hidden_size: int, device: str):
    from training.critic_model import CriticModel, CriticModelConfig
    ccfg = CriticModelConfig(hidden_size=hidden_size)
    return CriticModel(ccfg).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup Phase
# ─────────────────────────────────────────────────────────────────────────────

def run_warmup(
    config: TrainConfig,
    belief_model,
    critic_model,
    device: str,
    warmup_ckpt: Optional[str] = None,
    rank: int = 0,
):
    from training.warmup import WarmupTrainer, WarmupConfig

    wcfg = WarmupConfig(
        warmup_games=config.warmup_games,
        belief_epochs=config.warmup_belief_epochs,
        critic_epochs=config.warmup_critic_epochs,
        belief_batch_size=config.warmup_batch_size,
        critic_batch_size=config.warmup_batch_size,
        belief_lr=config.warmup_belief_lr,
        critic_lr=config.warmup_critic_lr,
        device=device,
    )
    # Under DDP, unwrap if needed
    bm = belief_model.module if hasattr(belief_model, "module") else belief_model
    cm = critic_model.module if hasattr(critic_model, "module") else critic_model
    trainer = WarmupTrainer(bm, cm, wcfg)

    if warmup_ckpt and os.path.isdir(warmup_ckpt):
        logger.info(f"[rank {rank}] Loading warmup checkpoint from {warmup_ckpt}")
        trainer.load_warmup_checkpoint(warmup_ckpt)
        return trainer

    # Only rank 0 trains during warmup (data is synthetic, no game needed)
    if rank == 0:
        logger.info("=== WARMUP PHASE ===")
        trainer.run()

        # ── W&B: log warmup losses ────────────────────────────────────
        if _WANDB_AVAILABLE and config.wandb_enabled and wandb.run is not None:
            for epoch, loss in enumerate(trainer.belief_history):
                wandb.log({"warmup/belief_loss": loss, "warmup/epoch": epoch})
            for epoch, loss in enumerate(trainer.critic_history):
                wandb.log({"warmup/critic_loss": loss, "warmup/epoch": epoch})

        warmup_save_dir = os.path.join(config.checkpoint_dir, "warmup")
        trainer.save_warmup_checkpoint(warmup_save_dir)

    # All ranks wait for rank 0 to finish saving before PPO starts
    if _is_ddp():
        import torch.distributed as dist
        dist.barrier()
        if rank != 0:
            # Load the checkpoint written by rank 0
            warmup_save_dir = os.path.join(config.checkpoint_dir, "warmup")
            if os.path.isdir(warmup_save_dir):
                logger.info(f"[rank {rank}] Loading warmup checkpoint from rank 0")
                trainer.load_warmup_checkpoint(warmup_save_dir)

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# PPO Phase
# ─────────────────────────────────────────────────────────────────────────────

async def run_ppo(
    config: TrainConfig,
    policy,
    belief_model,
    critic_model,
    device: str,
    rank: int = 0,
    world_size: int = 1,
    resume_from: Optional[str] = None,
    smoke_test: bool = False,
):
    """
    Alternating PPO training loop.

    The training alternates between a **Crewmate phase** and an **Impostor phase**
    every ``config.alternate_every`` iterations.  In each phase:

    * All agents (both roles) play games and produce trajectories.
    * Only the **active role's** LoRA adapter receives gradient updates
      (the other role's adapter is frozen during that phase).
    * A role-specific checkpoint is saved to ``crewmate_checkpoint_dir`` or
      ``impostor_checkpoint_dir``.

    Both roles share the same Qwen3-4B backbone — only the LoRA adapters differ.
    """
    from training.rollout_collector import RolloutCollector, RolloutConfig
    from training.ppo_trainer import PPOTrainer, PPOConfig
    from amongagents.agent.rewards import RewardEngine

    rollout_cfg = RolloutConfig(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        belief_dim=config.belief_dim,
        device=device,
        num_games=config.games_per_iter,
    )
    ppo_cfg = PPOConfig(
        clip_eps=config.clip_eps,
        value_loss_coef=config.value_loss_coef,
        belief_loss_coef=config.belief_loss_coef,
        entropy_coef=config.entropy_coef,
        ppo_lr_policy=config.ppo_lr_policy,
        ppo_lr_heads=config.ppo_lr_heads,
        max_grad_norm=config.max_grad_norm,
        ppo_epochs=config.ppo_epochs,
        mini_batch_size=config.mini_batch_size,
        target_kl=config.target_kl,
        device=device,
        ddp=(world_size > 1),
    )

    # Unwrap DDP shells
    raw_policy       = policy.module       if hasattr(policy,       "module") else policy
    raw_belief_model = belief_model.module if hasattr(belief_model, "module") else belief_model
    raw_critic_model = critic_model.module if hasattr(critic_model, "module") else critic_model

    reward_engine = RewardEngine()
    collector = RolloutCollector(
        policy=raw_policy,
        belief_model=raw_belief_model,
        critic_model=raw_critic_model,
        reward_engine=reward_engine,
        config=rollout_cfg,
    )
    ppo_trainer = PPOTrainer(policy, belief_model, critic_model, ppo_cfg)

    # ── Role schedule ─────────────────────────────────────────────────
    # Phases alternate: Crewmate × N, Impostor × N, Crewmate × N, …
    ROLES = ["Crewmate", "Impostor"]
    ROLE_CKPT_DIRS = {
        "Crewmate": config.crewmate_checkpoint_dir,
        "Impostor":  config.impostor_checkpoint_dir,
    }

    alternate_every = 1 if smoke_test else config.alternate_every

    def _current_role(iteration: int) -> str:
        phase_idx = (iteration // alternate_every) % 2
        return ROLES[phase_idx]

    # ── Resume ─────────────────────────────────────────────────────────
    start_iter = 0
    if resume_from and os.path.isdir(resume_from):
        logger.info(f"[rank {rank}] Resuming from {resume_from}")
        ppo_trainer.load_checkpoint(resume_from)
        try:
            start_iter = int(os.path.basename(resume_from).replace("iter_", "")) + 1
        except ValueError:
            pass

    game_factory   = make_game_factory(config)
    num_iterations = 1 if smoke_test else config.num_iterations
    games_per_iter = 1 if smoke_test else config.games_per_iter

    logger.info(
        f"[rank {rank}/{world_size}] === PPO PHASE (alternating): "
        f"{num_iterations} total iterations, alternate every {alternate_every} ==="
    )
    logger.info(f"  Phase pattern: {ROLES[0]} × {alternate_every} → {ROLES[1]} × {alternate_every} → …")

    all_metrics: List[Dict] = []
    prev_role: Optional[str] = None

    for iteration in range(start_iter, start_iter + num_iterations):
        t0 = time.time()
        active_role = _current_role(iteration)

        # Announce phase change
        if active_role != prev_role:
            logger.info(
                f"\n{'='*60}\n"
                f"  SWITCHING TO [{active_role.upper()}] PHASE  (iter {iteration + 1})\n"
                f"{'='*60}"
            )
            prev_role = active_role

        # ── Collect trajectories (all agents play; both roles produce data) ──
        trajectories = await collector.collect(game_factory, num_games=games_per_iter)

        n_role = sum(1 for t in trajectories if t.role == active_role)
        logger.info(
            f"  [rank {rank}] iter {iteration + 1} [{active_role}]: "
            f"{len(trajectories)} trajs total, {n_role} for active role, "
            f"{sum(len(t.steps) for t in trajectories)} steps"
        )

        # ── PPO update — only the active role's adapter is updated ───────────
        metrics = ppo_trainer.update(trajectories, role_filter=active_role)
        if metrics:
            metrics["iteration"]   = iteration
            metrics["role"]        = active_role
            metrics["rank"]        = rank
            metrics["elapsed_sec"] = round(time.time() - t0, 1)
            all_metrics.append(metrics)

            # ── W&B: log PPO metrics (rank 0 only) ──────────────────────
            if rank == 0 and _WANDB_AVAILABLE and config.wandb_enabled and wandb.run is not None:
                role_tag = active_role.lower()  # "crewmate" or "impostor"
                wandb.log({
                    f"ppo/{role_tag}/policy_loss":  metrics.get("policy_loss", 0),
                    f"ppo/{role_tag}/value_loss":   metrics.get("value_loss", 0),
                    f"ppo/{role_tag}/belief_loss":  metrics.get("belief_loss", 0),
                    f"ppo/{role_tag}/entropy":      metrics.get("entropy", 0),
                    f"ppo/{role_tag}/kl_mean":      metrics.get("kl_mean", 0),
                    f"ppo/{role_tag}/total_loss":   metrics.get("total_loss", 0),
                    "ppo/elapsed_sec":              metrics["elapsed_sec"],
                    "ppo/active_role":              active_role,
                    "ppo/n_trajectories":           len(trajectories),
                    "ppo/n_role_trajectories":      n_role,
                }, step=iteration)

        # ── Role-specific checkpoint (rank 0 only) ──────────────────────────
        if rank == 0 and (iteration + 1) % config.checkpoint_every == 0:
            ckpt_dir = ROLE_CKPT_DIRS[active_role]
            ppo_trainer.save_role_checkpoint(active_role, ckpt_dir, iteration)

    # ── Final checkpoints for both roles ────────────────────────────────────
    if rank == 0:
        final_iter = start_iter + num_iterations - 1
        for role in ROLES:
            ckpt_dir = ROLE_CKPT_DIRS[role]
            ppo_trainer.save_role_checkpoint(role, ckpt_dir, final_iter)

        # Save metrics
        os.makedirs(config.log_dir, exist_ok=True)
        metrics_path = os.path.join(config.log_dir, "ppo_metrics.jsonl")
        with open(metrics_path, "a") as f:
            for m in all_metrics:
                f.write(json.dumps(m) + "\n")
        logger.info(f"Metrics saved → {metrics_path}")

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Among Us RL Training Pipeline")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (e.g. training/train_config.yml)")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Run a minimal 1-iteration smoke test")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Skip warmup phase entirely")
    p.add_argument("--warmup-checkpoint", type=str, default=None,
                   help="Load warmup checkpoint instead of re-training")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to PPO checkpoint dir to resume from")
    # CLI overrides (applied on top of YAML)
    p.add_argument("--model-path",      type=str, default=None)
    p.add_argument("--cuda-device",     type=str, default=None,
                   help="CUDA device index — ignored under torchrun")
    p.add_argument("--checkpoint-dir",  type=str, default=None)
    p.add_argument("--log-dir",         type=str, default=None)
    p.add_argument("--num-iterations",  type=int, default=None)
    p.add_argument("--games-per-iter",  type=int, default=None)
    p.add_argument("--mini-batch-size", type=int, default=None,
                   help="Per-GPU PPO mini-batch size")
    p.add_argument("--seed",            type=int, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    # ── 1. Build config (YAML → CLI overrides) ─────────────────────────────
    cfg = TrainConfig()
    if args.config:
        logger.info(f"Loading config from {args.config}")
        cfg = load_config_from_yaml(args.config, base=cfg)

    # CLI overrides (always win over YAML)
    if args.model_path:      cfg.model_path      = args.model_path
    if args.cuda_device:     cfg.cuda_devices    = args.cuda_device   # --cuda-device still works
    if args.checkpoint_dir:  cfg.checkpoint_dir  = args.checkpoint_dir
    if args.log_dir:         cfg.log_dir         = args.log_dir
    if args.num_iterations:  cfg.num_iterations  = args.num_iterations
    if args.games_per_iter:  cfg.games_per_iter  = args.games_per_iter
    if args.mini_batch_size: cfg.mini_batch_size = args.mini_batch_size
    if args.seed:            cfg.seed            = args.seed

    if args.smoke_test:
        logger.info("🔥 SMOKE TEST MODE — minimal settings")
        cfg.warmup_games         = 2
        cfg.warmup_belief_epochs = 2
        cfg.warmup_critic_epochs = 2
        cfg.num_iterations       = 1
        cfg.games_per_iter       = 1

    # ── 2. Parse device list and (optionally) auto-relaunch under torchrun ──
    device_list = [d.strip() for d in cfg.cuda_devices.split(",") if d.strip()]
    nproc = len(device_list)

    # Set CUDA_VISIBLE_DEVICES before any torch import so device indices are
    # remapped correctly: "0,1,3" makes LOCAL_RANK=0→GPU0, 1→GPU1, 2→GPU3.
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices

    # Auto-relaunch under torchrun when >1 GPU and not already inside torchrun.
    is_under_torchrun = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if nproc > 1 and not is_under_torchrun:
        import subprocess
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--",
        ] + sys.argv          # pass all original args (including --config)
        logger.info(
            f"Detected {nproc} GPUs ({cfg.cuda_devices}). "
            f"Re-launching under torchrun: {' '.join(cmd)}"
        )
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # ── 3. Distributed setup ───────────────────────────────────────────
    import torch
    is_ddp   = _is_ddp()
    rank     = 0
    local_rank = 0
    world_size = 1

    if is_ddp:
        rank, local_rank, world_size = _ddp_setup(cfg.ddp_backend)
        # All ranks already have the same config: torchrun passes identical
        # sys.argv to every rank, so each rank parses the same YAML + CLI.
        device = f"cuda:{local_rank}"
        logger.info(f"DDP init: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        # Single-GPU: device 0 inside the already-remapped CUDA_VISIBLE_DEVICES
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + rank)

    # ── 4. Create dirs + save resolved config ────────────────────────────
    if rank == 0:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        save_config_to_yaml(cfg, os.path.join(cfg.log_dir, "resolved_config.yml"))

        # ── W&B initialisation (rank 0 only) ─────────────────────────────
        if cfg.wandb_enabled and _WANDB_AVAILABLE:
            _tags = list(cfg.wandb_tags) + [cfg.game_config_name]
            if args.smoke_test:
                _tags.append("smoke-test")
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=cfg.wandb_run_name or None,
                tags=_tags,
                config=asdict(cfg),
                dir=cfg.log_dir,
                resume="allow",
            )
            logger.info(f"W&B run initialised → {wandb.run.url}")
        elif cfg.wandb_enabled and not _WANDB_AVAILABLE:
            logger.warning("wandb_enabled=true but `wandb` is not installed — skipping. "
                           "Install with:  pip install wandb")

    # ── 5. Build models ──────────────────────────────────────────────────────
    logger.info(f"[rank {rank}] Building models on {device} …")
    policy       = build_lora_policy(cfg, device)
    hidden_size  = policy.hidden_size
    belief_model = build_belief_model(cfg, hidden_size, device)
    critic_model = build_critic_model(cfg, hidden_size, device)

    # ── 6. Wrap models in DDP ────────────────────────────────────────────────
    if is_ddp:
        # policy: LoRA adapters are the only trainable params
        # find_unused_parameters=True because the frozen base params are not used
        policy       = _ddp_wrap(policy,       local_rank, find_unused=True)
        belief_model = _ddp_wrap(belief_model, local_rank)
        critic_model = _ddp_wrap(critic_model, local_rank)

    # ── W&B model watch (after models exist + DDP wrap; rank 0 only) ───────
    if rank == 0 and _WANDB_AVAILABLE and cfg.wandb_enabled \
            and cfg.wandb_watch_models and wandb.run is not None:
        wandb.watch(policy, log="all", log_freq=10)

    # ── 7. Warmup ────────────────────────────────────────────────────────────
    if not args.skip_warmup:
        run_warmup(
            cfg, belief_model, critic_model, device,
            warmup_ckpt=args.warmup_checkpoint,
            rank=rank,
        )
    else:
        logger.info(f"[rank {rank}] Skipping warmup phase.")

    # ── 8. PPO ──────────────────────────────────────────────────────────────────
    await run_ppo(
        cfg, policy, belief_model, critic_model, device,
        rank=rank, world_size=world_size,
        resume_from=args.resume,
        smoke_test=args.smoke_test,
    )

    if rank == 0:
        logger.info("✅ Training complete.")
        if _WANDB_AVAILABLE and cfg.wandb_enabled and wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished.")

    if is_ddp:
        _ddp_cleanup()


if __name__ == "__main__":
    asyncio.run(main())
