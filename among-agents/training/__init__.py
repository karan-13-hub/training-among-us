"""
training/ — RL Training Pipeline for Among Us with Qwen3-4B + LoRA

Modules:
  lora_model.py       — Qwen3-4B + LoRA policy wrapper
  belief_model.py     — Trainable belief head (suspicion / second-order)
  critic_model.py     — Trainable value/critic head
  warmup.py           — Supervised warmup trainer for belief + critic
  rollout_collector.py— Trajectory collection with GAE
  ppo_trainer.py      — Joint PPO update (policy + belief + critic)
  train.py            — Config-driven entry point
"""
