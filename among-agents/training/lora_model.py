"""
lora_model.py — Qwen3-4B + LoRA Policy Wrapper

Wraps the Qwen3-4B model with PEFT LoRA adapters, exposing:
  - generate()         : for rollout inference (returns text)
  - get_log_probs()    : for computing π(a|s) ratios (PPO)
  - forward()          : for training, returns (logits, hidden_states)
  - save_checkpoint()  : persist LoRA weights
  - load_checkpoint()  : restore LoRA weights
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── LoRA Configuration ────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    r: int = 16                          # LoRA rank
    lora_alpha: int = 32                 # scaling factor
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    do_sample: bool = True
    enable_thinking: bool = False    # Qwen3 thinking mode toggle


# ── LoRA Policy ────────────────────────────────────────────────────────────

class LoRAQwenPolicy(nn.Module):
    """
    Qwen3-4B + LoRA adapter policy for Among Us agents.

    The model serves as the *shared backbone* — its hidden states are
    also forwarded to the BeliefModel and CriticModel heads.

    Parameters
    ----------
    model_path : str
        Local path to the Qwen3-4B model directory.
    lora_config : LoRAConfig
        LoRA hyper-parameters.
    device : str | torch.device
        Target device ("cuda" / "cpu").
    load_in_8bit : bool
        Whether to load base model in 8-bit (bitsandbytes).
    """

    def __init__(
        self,
        model_path: str,
        lora_config: Optional[LoRAConfig] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        roles: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        roles : list of str, optional
            If provided (e.g. ``["Crewmate", "Impostor"]``), two named LoRA
            adapters are created immediately — one per role.  Call
            :meth:`set_active_role` to switch which adapter is in use.
            If *None*, a single default adapter is created (original behaviour).
        """
        super().__init__()
        self.model_path  = model_path
        self.lora_config = lora_config or LoRAConfig()
        self.device      = device
        self._active_role: Optional[str] = None

        self._load_base_model(load_in_8bit)
        self._apply_lora(roles=roles)

    # ── Initialisation ────────────────────────────────────────────────

    def _load_base_model(self, load_in_8bit: bool):
        """Load tokeniser + base causal LM."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading base model from {self.model_path} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = (
            BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=bnb_config,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        self.hidden_size = self.base_model.config.hidden_size
        logger.info(f"Base model loaded (hidden_size={self.hidden_size})")

    def _apply_lora(self, roles: Optional[List[str]] = None):
        """Wrap base model in LoRA adapters via PEFT.

        If *roles* is given, one named adapter is created per role.  The first
        role becomes the default active adapter.
        Otherwise a single default adapter is created (original behaviour).
        """
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = self.lora_config
        peft_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias=cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        if roles:
            # Create the first adapter via get_peft_model (initialises the model)
            first_role = roles[0]
            adapter_name = self._role_adapter_name(first_role)
            self.model = get_peft_model(self.base_model, peft_config,
                                        adapter_name=adapter_name)
            self._active_role = first_role
            # Add remaining adapters (share same LoRA config)
            for role in roles[1:]:
                self.add_role_adapter(role)
            # Activate the first adapter
            self.set_active_role(first_role)
        else:
            # Single default adapter — backwards-compatible
            self.model = get_peft_model(self.base_model, peft_config)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"LoRA applied: {n_trainable:,} / {n_total:,} params trainable "
            f"({100 * n_trainable / n_total:.2f}%)"
        )

    # ── Dual-Adapter (Role-Based) API ─────────────────────────────────

    @staticmethod
    def _role_adapter_name(role: str) -> str:
        """Convert role string to a valid PEFT adapter name."""
        return role.lower().replace(" ", "_")

    @property
    def active_role(self) -> Optional[str]:
        """The role whose adapter is currently active."""
        return self._active_role

    def has_role(self, role: str) -> bool:
        """True if a named adapter for *role* has been added."""
        name = self._role_adapter_name(role)
        return hasattr(self.model, "peft_config") and name in self.model.peft_config

    def add_role_adapter(self, role: str) -> None:
        """Add a new named LoRA adapter for *role* (shares the LoRA config)."""
        from peft import LoraConfig, TaskType
        if self.has_role(role):
            logger.debug(f"Adapter for role '{role}' already exists, skipping.")
            return
        cfg  = self.lora_config
        name = self._role_adapter_name(role)
        peft_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias=cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model.add_adapter(name, peft_config)
        logger.info(f"Added LoRA adapter for role '{role}' (name='{name}')")

    def set_active_role(self, role: str) -> None:
        """
        Switch to the LoRA adapter for *role*.

        Only *this role’s* adapter parameters will be ``requires_grad=True``.
        All other adapters are frozen, so the optimizer only touches the active role.
        """
        name = self._role_adapter_name(role)
        if not self.has_role(role):
            raise ValueError(f"No adapter found for role '{role}'. Call add_role_adapter() first.")
        self.model.set_adapter(name)
        self._active_role = role

        # Freeze all but the active adapter
        for pname, param in self.model.named_parameters():
            if "lora_" in pname:
                param.requires_grad = (name in pname)

        logger.debug(f"Active LoRA adapter -> '{role}' (adapter_name='{name}')")

    def trainable_parameters(self, role: Optional[str] = None) -> List:
        """
        Return trainable parameters.  When *role* is given, return ONLY that
        role’s adapter parameters (for building role-specific optimizers).
        When *role* is None, return all currently-requires_grad parameters.
        """
        if role is None:
            return [p for p in self.model.parameters() if p.requires_grad]
        name = self._role_adapter_name(role)
        return [
            p for pname, p in self.model.named_parameters()
            if "lora_" in pname and name in pname
        ]

    # ── Forward Pass ─────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids       : [B, L]
        attention_mask  : [B, L]  (optional)

        Returns
        -------
        logits          : [B, L, vocab_size]
        last_hidden     : [B, L, hidden_size]  — last transformer layer output
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = out.logits                 # [B, L, V]
        last_hidden = out.hidden_states[-1] # [B, L, H]
        return logits, last_hidden

    # ── Log-Probability Computation ───────────────────────────────────

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the sum of token log-probs for the action sequence.

        Parameters
        ----------
        input_ids   : [B, L_prompt]  — tokenised prompt (context)
        action_ids  : [B, L_action]  — tokenised action (continuation)
        attention_mask : [B, L_prompt + L_action]  (optional)

        Returns
        -------
        log_probs : [B]  — sum of log π(a_t | context)
        """
        B, L_p = input_ids.shape
        _, L_a = action_ids.shape

        # Concatenate prompt + action; labels are shifted by one
        full_ids = torch.cat([input_ids, action_ids], dim=1)  # [B, L_p+L_a]
        if attention_mask is None:
            attention_mask = torch.ones_like(full_ids)

        with torch.no_grad() if not self.training else torch.enable_grad():
            out = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

        logits = out.logits  # [B, L_p+L_a, V]
        # Align: predict action_ids using the logits at positions [L_p-1 : L_p+L_a-1]
        action_logits = logits[:, L_p - 1 : L_p + L_a - 1, :]  # [B, L_a, V]
        log_probs_token = torch.nn.functional.log_softmax(action_logits, dim=-1)
        # Gather ONLY the token positions that correspond to action_ids
        action_log_probs = log_probs_token.gather(
            dim=-1, index=action_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, L_a]
        return action_log_probs.sum(dim=-1)  # [B]

    def get_log_probs_and_hidden(
        self,
        input_ids: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like get_log_probs but also returns the last hidden state
        (averaged over prompt length) for use by Belief + Critic heads.

        Returns
        -------
        log_probs   : [B]
        hidden_pool : [B, hidden_size]  (mean pooled over prompt tokens)
        """
        B, L_p = input_ids.shape
        _, L_a = action_ids.shape

        full_ids = torch.cat([input_ids, action_ids], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(full_ids)

        out = self.model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = out.logits
        last_hidden = out.hidden_states[-1]  # [B, L_p+L_a, H]

        # Compute log probs
        action_logits = logits[:, L_p - 1 : L_p + L_a - 1, :]
        log_probs_token = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs_token.gather(
            dim=-1, index=action_ids.unsqueeze(-1)
        ).squeeze(-1)
        log_probs = action_log_probs.sum(dim=-1)  # [B]

        # Pool hidden over prompt tokens (ignore action tokens to avoid leakage)
        prompt_hidden = last_hidden[:, :L_p, :]   # [B, L_p, H]
        # Mean pool, ignoring padding if attention_mask was supplied
        prompt_mask = attention_mask[:, :L_p].unsqueeze(-1).float()
        hidden_pool = (prompt_hidden * prompt_mask).sum(dim=1) / prompt_mask.sum(dim=1).clamp(min=1)
        return log_probs, hidden_pool  # [B], [B, H]

    # ── Inference (Rollout) ───────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a text response from a prompt string.

        Returns
        -------
        text       : generated string (action text)
        input_ids  : [1, L_prompt]
        action_ids : [1, L_action]
        """
        cfg = gen_config or GenerationConfig()

        extra_kwargs = {}
        if cfg.enable_thinking is not None:
            extra_kwargs["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]            # [1, L_prompt]
        L_prompt = input_ids.shape[1]

        out_ids = self.model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )  # [1, L_prompt + L_action]

        action_ids = out_ids[:, L_prompt:]     # [1, L_action]
        text = self.tokenizer.decode(action_ids[0], skip_special_tokens=True)
        return text, input_ids, action_ids

    # ── Checkpoint I/O ────────────────────────────────────────────────

    def save_checkpoint(self, save_dir: str):
        """Save LoRA adapter weights to disk."""
    def save_checkpoint(self, save_dir: str):
        """Save active adapter weights to disk (backwards-compatible)."""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"LoRA checkpoint saved to {save_dir}")

    def save_role_checkpoint(self, role: str, save_dir: str) -> None:
        """
        Save the adapter for *role* to *save_dir* without changing which
        adapter is currently active.
        """
        prev = self._active_role
        self.set_active_role(role)
        os.makedirs(save_dir, exist_ok=True)
        adapter_name = self._role_adapter_name(role)
        self.model.save_pretrained(save_dir, selected_adapters=[adapter_name])
        if prev is not None and prev != role:
            self.set_active_role(prev)
        logger.info(f"Role-checkpoint '{role}' saved → {save_dir}")

    def load_checkpoint(self, load_dir: str):
        """Load LoRA adapter weights from disk (backwards-compatible)."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, load_dir)
        logger.info(f"LoRA checkpoint loaded from {load_dir}")

    def load_role_checkpoint(self, role: str, load_dir: str) -> None:
        """
        Load saved adapter weights for *role* from *load_dir*, injecting
        them into (or replacing) the existing named adapter.
        """
        from peft import PeftModel
        adapter_name = self._role_adapter_name(role)
        if not os.path.isdir(load_dir):
            logger.warning(f"Role-checkpoint dir '{load_dir}' not found — skipping.")
            return
        # load_adapter merges weights into the existing PEFT model
        self.model.load_adapter(load_dir, adapter_name=adapter_name)
        logger.info(f"Role-checkpoint '{role}' loaded from {load_dir}")

    # ── Helpers ─────────────────────────────────────────────────────

    def freeze_base(self):
        """Freeze all base model parameters (keep only LoRA adapters trainable)."""
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def unfreeze_lora(self):
        """Ensure active LoRA adapter parameters are trainable."""
        active_name = (
            self._role_adapter_name(self._active_role)
            if self._active_role else None
        )
        for pname, param in self.model.named_parameters():
            if "lora_" in pname:
                param.requires_grad = (active_name is None or active_name in pname)
