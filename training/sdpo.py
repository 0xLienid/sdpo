"""
SDPO (Self-Distillation Policy Optimization) Training Loop

Based on "Reinforcement Learning via Self-Distillation" (Hübotter et al.)

Key components:
- Teacher model: EMA of student weights, conditioned on feedback
- Student model: Current weights, conditioned on prompt only
- Loss: Reverse-KL divergence with Top-K distillation
- Multiple rollouts per question with importance sampling
"""

import os
import json
import copy
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from dotenv import load_dotenv

from validators.validator import Validator

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SDPOHparams:
    """Hyperparameters for SDPO training."""
    # Optimizer
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Training
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1

    # Sequence lengths
    max_prompt_length: int = 2048
    max_response_length: int = 8192

    # SDPO loss
    teacher_ema_rate: float = 0.01  # EMA update rate for teacher
    top_k_distillation: int = 20  # Top-K logits for distillation
    temperature: float = 1.0
    importance_sampling_clip: float = 2.0  # Clip ratio for importance sampling
    clip_advantages: Optional[float] = None  # Optional advantage clipping

    # Distill-on-regen: generate with teacher, then distill student towards teacher's distribution
    distill_on_regen: bool = False
    regen_temperature: float = 0.7  # Temperature for teacher regeneration

    # Include student's attempt in teacher context (alongside feedback)
    include_student_attempt: bool = False

    # Rollouts
    num_rollouts: int = 8  # Number of rollouts per question
    rollout_temperature: float = 1.0

    # Logging
    log_interval: int = 10
    validation_interval: int = 100
    save_interval: int = 500


@dataclass
class RolloutResult:
    """Result from a rollout function."""
    prompt: str
    completion: str
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor


@dataclass
class FeedbackResult:
    """Result from a feedback function."""
    feedback_text: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidatorRunConfig:
    """Runtime configuration for a validator."""
    batch_size: int = 4
    max_new_tokens: int = 1024
    max_seq_length: int = 2048


class EMATeacher:
    """
    EMA Teacher model that maintains a separate model instance.

    Unlike the old EMAModel that swapped weights in/out, this maintains
    a complete separate model that:
    1. Lives on the same device(s) as the student
    2. Can be wrapped by DDP for multi-GPU training
    3. Updates EMA weights in-place without swapping
    """

    def __init__(
        self,
        student_model: AutoModelForCausalLM,
        decay: float = 0.99,
        device: Optional[torch.device] = None,
    ):
        """
        Create an EMA teacher as a separate model instance.

        Args:
            student_model: The student model to copy architecture and initial weights from
            decay: EMA decay rate (1 - update_rate). Higher = slower updates.
            device: Device to place teacher on. If None, uses student's device.
        """
        self.decay = decay

        # Create a deep copy of the student model
        # This creates a completely separate model instance
        self.model = copy.deepcopy(student_model)

        # Move to specified device if provided
        if device is not None:
            self.model = self.model.to(device)

        # Teacher should never require gradients
        self.model.requires_grad_(False)
        self.model.eval()

    def to(self, device: torch.device) -> "EMATeacher":
        """Move teacher model to device."""
        self.model = self.model.to(device)
        return self

    @torch.no_grad()
    def update(self, student_model: AutoModelForCausalLM):
        """
        Update EMA parameters from student model.

        This should be called after optimizer.step() and only when
        gradients are synced (accelerator.sync_gradients).
        """
        student_params = dict(student_model.named_parameters())

        for name, teacher_param in self.model.named_parameters():
            if name in student_params:
                student_param = student_params[name]
                # EMA update: teacher = decay * teacher + (1 - decay) * student
                teacher_param.mul_(self.decay).add_(student_param.data, alpha=1 - self.decay)

    @torch.no_grad()
    def sync_across_processes(self, accelerator: Accelerator):
        """
        Synchronize teacher weights across all processes.

        In DDP, each process updates EMA from its local student shard.
        This syncs teacher weights so all processes have identical teachers.
        Call this after update() when using multi-GPU training.
        """
        if accelerator.num_processes > 1:
            for param in self.model.parameters():
                # Average across all processes
                torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.AVG)

    def __call__(self, *args, **kwargs):
        """Forward pass through teacher model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate with teacher model."""
        return self.model.generate(*args, **kwargs)


def build_student_messages(prompt: str, completion: str) -> List[Dict[str, str]]:
    """Build chat messages for student (prompt + completion as assistant response)."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def build_teacher_messages(
    prompt: str,
    completion: str,
    feedback: str,
    prior_solution: Optional[str],
    student_attempt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build chat messages for teacher (feedback context + prompt + completion)."""
    teacher_context_parts = []
    if student_attempt is not None:
        teacher_context_parts.append(f"## Previous Attempt\n```python\n{student_attempt}\n```")
    teacher_context_parts.append(f"## Feedback\n{feedback}")
    if prior_solution is not None:
        teacher_context_parts.append(f"## Prior Correct Solution\n```python\n{prior_solution}\n```")
    teacher_context_parts.append(f"## Original Question\n{prompt}")
    teacher_context_parts.append("Given the feedback above, re-evaluate and improve upon the following attempt:")

    return [
        {"role": "user", "content": "\n\n".join(teacher_context_parts)},
        {"role": "assistant", "content": completion},
    ]


def compute_sdpo_loss(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,  # Separate EMA teacher model
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    feedback: str,
    prior_solution: Optional[str],
    hparams: SDPOHparams,
    student_attempt: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the SDPO loss for a single example.

    Loss = Reverse-KL with Top-K distillation:
        L = sum over top-K tokens of: teacher_prob * (log(teacher_prob) - log(student_prob))

    Args:
        student_model: The student model (current weights)
        teacher_model: The teacher model (separate EMA model)
        tokenizer: The tokenizer
        prompt: Original prompt/question
        completion: Generated completion to evaluate
        feedback: Environment feedback
        prior_solution: Optional prior correct solution
        hparams: Hyperparameters

    Returns:
        loss: The SDPO loss tensor
        metrics: Dictionary of metrics for logging
    """
    student_device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length

    # Build student messages (prompt + completion as assistant)
    student_messages = build_student_messages(prompt, completion)
    student_full = tokenizer.apply_chat_template(
        student_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Build teacher messages (feedback context + prompt + completion as assistant)
    teacher_messages = build_teacher_messages(prompt, completion, feedback, prior_solution, student_attempt)
    teacher_full = tokenizer.apply_chat_template(
        teacher_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Get prompt-only portions to find where completion starts
    student_prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    teacher_context_parts = []
    if student_attempt is not None:
        teacher_context_parts.append(f"## Previous Attempt\n```python\n{student_attempt}\n```")
    teacher_context_parts.append(f"## Feedback\n{feedback}")
    if prior_solution is not None:
        teacher_context_parts.append(f"## Prior Correct Solution\n```python\n{prior_solution}\n```")
    teacher_context_parts.append(f"## Original Question\n{prompt}")
    teacher_context_parts.append("Given the feedback above, re-evaluate and improve upon the following attempt:")
    teacher_prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n\n".join(teacher_context_parts)}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    student_encoding = tokenizer(
        student_full, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(student_device)

    teacher_encoding = tokenizer(
        teacher_full, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(teacher_device)

    student_prompt_encoding = tokenizer(
        student_prompt_only, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    )
    teacher_prompt_encoding = tokenizer(
        teacher_prompt_only, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    )

    student_prompt_len = student_prompt_encoding.input_ids.shape[1]
    teacher_prompt_len = teacher_prompt_encoding.input_ids.shape[1]

    # Forward pass for student (with gradients)
    student_outputs = student_model(**student_encoding)
    student_logits = student_outputs.logits

    # Forward pass for teacher (no gradients)
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_logits = teacher_outputs.logits

    # Extract completion logits
    student_completion_len = student_encoding.input_ids.shape[1] - student_prompt_len
    teacher_completion_len = teacher_encoding.input_ids.shape[1] - teacher_prompt_len
    completion_len = min(student_completion_len, teacher_completion_len)

    if completion_len <= 0:
        return torch.tensor(0.0, device=student_device, requires_grad=True), {"loss": 0.0, "completion_tokens": 0}

    student_logits_completion = student_logits[0, student_prompt_len-1:student_prompt_len-1+completion_len, :]
    teacher_logits_completion = teacher_logits[0, teacher_prompt_len-1:teacher_prompt_len-1+completion_len, :]

    # Move teacher logits to student device for loss computation
    teacher_logits_completion = teacher_logits_completion.to(student_device)

    # Apply temperature
    student_logits_scaled = student_logits_completion / hparams.temperature
    teacher_logits_scaled = teacher_logits_completion / hparams.temperature

    # Top-K distillation: only use top-K logits from student (as per paper equation 7)
    top_k = hparams.top_k_distillation
    if top_k > 0 and top_k < student_logits_scaled.shape[-1]:
        # Get top-K indices from student distribution
        _, top_k_indices = torch.topk(student_logits_scaled, top_k, dim=-1)

        # Gather top-K logits
        student_topk_logits = torch.gather(student_logits_scaled, -1, top_k_indices)
        teacher_topk_logits = torch.gather(teacher_logits_scaled, -1, top_k_indices)

        # Compute softmax over top-K only
        student_probs = F.softmax(student_topk_logits, dim=-1)
        student_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
    else:
        student_probs = F.softmax(student_logits_scaled, dim=-1)
        student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)

    # KL(student || teacher) as per paper equation 7:
    # KL(π_θ || q_θ) = Σ π_θ(y) * log(π_θ(y) / q_θ(y))
    # = Σ student_probs * (student_log_probs - teacher_log_probs)
    # Gradients flow through student_probs and student_log_probs; teacher is stopgrad
    kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    loss = kl_per_token.mean()

    metrics = {
        "loss": loss.item(),
        "completion_tokens": completion_len,
        "kl_per_token": kl_per_token.mean().item(),
    }

    return loss, metrics


def build_teacher_regen_prompt(
    prompt: str,
    feedback: str,
    student_attempt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build chat messages for teacher regeneration (feedback context + prompt, no completion)."""
    teacher_context_parts = []
    if student_attempt is not None:
        teacher_context_parts.append(f"## Previous Attempt\n```python\n{student_attempt}\n```")
    teacher_context_parts.append(f"## Feedback\n{feedback}")
    teacher_context_parts.append(f"## Original Question\n{prompt}")
    teacher_context_parts.append("Given the feedback above, provide an improved solution:")

    return [
        {"role": "user", "content": "\n\n".join(teacher_context_parts)},
    ]


def compute_distill_on_regen_loss(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,  # Separate EMA teacher model
    tokenizer: AutoTokenizer,
    prompt: str,
    student_completion: str,
    feedback: str,
    hparams: SDPOHparams,
    student_attempt: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the distill-on-regen loss for a single example.

    Both models stay on their own sequences:
    1. Student is evaluated on its own completion S - we collect top-K logits at each position
    2. Teacher regenerates completion T with feedback, evaluated on T
    3. For each position (up to min(len(S), len(T))), get teacher's logits for student's top-K tokens
    4. Distill student toward teacher over those top-K tokens

    This keeps student on-policy (reducing forgetting) while teacher's corrections compound
    (each position conditioned on teacher's prior tokens). The distillation is gentle since
    we only shift probability mass among tokens the student already considers likely.

    Args:
        student_model: The student model (current weights)
        teacher_model: The teacher model (separate EMA model)
        tokenizer: The tokenizer
        prompt: Original prompt/question
        student_completion: The student's generated completion
        feedback: Environment feedback from the student's attempt
        hparams: Hyperparameters
        student_attempt: Optional student attempt to include in teacher context

    Returns:
        loss: The distillation loss tensor
        metrics: Dictionary of metrics for logging
    """
    student_device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    pad_token_id = tokenizer.pad_token_id

    # === Step 1: Get student's logits on its own completion ===
    student_messages = build_student_messages(prompt, student_completion)
    student_full = tokenizer.apply_chat_template(
        student_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    student_prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    student_encoding = tokenizer(
        student_full, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(student_device)

    student_prompt_encoding = tokenizer(
        student_prompt_only, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    )

    student_prompt_len = student_prompt_encoding.input_ids.shape[1]
    student_total_len = student_encoding.input_ids.shape[1]
    student_completion_len = student_total_len - student_prompt_len

    if student_completion_len <= 0:
        return torch.tensor(0.0, device=student_device, requires_grad=True), {
            "loss": 0.0,
            "completion_tokens": 0,
            "teacher_completion_len": 0,
        }

    # Forward pass for student (with gradients)
    student_outputs = student_model(**student_encoding)
    student_logits = student_outputs.logits

    # Extract student's completion logits and create mask for non-pad tokens
    # Logits at position i predict token i+1
    student_logits_completion = student_logits[0, student_prompt_len-1:student_prompt_len-1+student_completion_len, :]

    # Get the actual completion token IDs to create pad mask
    student_completion_ids = student_encoding.input_ids[0, student_prompt_len:student_total_len]
    student_non_pad_mask = (student_completion_ids != pad_token_id) if pad_token_id is not None else torch.ones_like(student_completion_ids, dtype=torch.bool)

    # === Step 2: Generate teacher's completion with feedback ===
    # Use left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    teacher_regen_messages = build_teacher_regen_prompt(prompt, feedback, student_attempt)
    teacher_regen_prompt = tokenizer.apply_chat_template(
        teacher_regen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    teacher_regen_inputs = tokenizer(
        teacher_regen_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=hparams.max_prompt_length,
        padding=False,
    ).to(teacher_device)

    teacher_gen_prompt_len = teacher_regen_inputs.input_ids.shape[1]

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": student_completion_len,  # Match student's length
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": hparams.regen_temperature > 0,
        }
        if hparams.regen_temperature > 0:
            generation_kwargs["temperature"] = hparams.regen_temperature
            generation_kwargs["top_p"] = 0.95

        teacher_output = teacher_model.generate(
            **teacher_regen_inputs,
            **generation_kwargs,
        )

    tokenizer.padding_side = original_padding_side  # Restore padding side

    # Extract teacher completion, stripping any pad tokens
    teacher_completion_ids = teacher_output[0, teacher_gen_prompt_len:]
    if pad_token_id is not None:
        pad_mask = teacher_completion_ids == pad_token_id
        if pad_mask.any():
            first_pad = pad_mask.nonzero(as_tuple=True)[0][0].item()
            teacher_completion_ids = teacher_completion_ids[:first_pad]

    teacher_completion = tokenizer.decode(teacher_completion_ids, skip_special_tokens=True)

    if len(teacher_completion_ids) == 0:
        return torch.tensor(0.0, device=student_device, requires_grad=True), {
            "loss": 0.0,
            "completion_tokens": 0,
            "teacher_completion_len": 0,
        }

    # === Step 3: Get teacher's logits on its own completion ===
    teacher_context_parts = []
    if student_attempt is not None:
        teacher_context_parts.append(f"## Previous Attempt\n```python\n{student_attempt}\n```")
    teacher_context_parts.append(f"## Feedback\n{feedback}")
    teacher_context_parts.append(f"## Original Question\n{prompt}")
    teacher_context_parts.append("Given the feedback above, provide an improved solution:")
    teacher_messages = [
        {"role": "user", "content": "\n\n".join(teacher_context_parts)},
        {"role": "assistant", "content": teacher_completion},
    ]
    teacher_full = tokenizer.apply_chat_template(
        teacher_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    teacher_encoding = tokenizer(
        teacher_full, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(teacher_device)

    teacher_total_len = teacher_encoding.input_ids.shape[1]
    teacher_completion_len = teacher_total_len - teacher_gen_prompt_len

    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_logits = teacher_outputs.logits

    # Extract teacher's completion logits
    teacher_logits_completion = teacher_logits[0, teacher_gen_prompt_len-1:teacher_gen_prompt_len-1+teacher_completion_len, :]

    # Get teacher's completion token IDs for pad mask
    teacher_completion_token_ids = teacher_encoding.input_ids[0, teacher_gen_prompt_len:teacher_total_len]
    teacher_non_pad_mask = (teacher_completion_token_ids != pad_token_id) if pad_token_id is not None else torch.ones_like(teacher_completion_token_ids, dtype=torch.bool)

    # Move teacher tensors to student device for loss computation
    teacher_logits_completion = teacher_logits_completion.to(student_device)
    teacher_non_pad_mask = teacher_non_pad_mask.to(student_device)

    # === Step 4: Compute loss over min(student_len, teacher_len) positions ===
    completion_len = min(student_completion_len, teacher_completion_len)

    if completion_len <= 0:
        return torch.tensor(0.0, device=student_device, requires_grad=True), {
            "loss": 0.0,
            "completion_tokens": 0,
            "teacher_completion_len": len(teacher_completion_ids),
        }

    # Truncate to common length
    student_logits_completion = student_logits_completion[:completion_len, :]
    teacher_logits_completion = teacher_logits_completion[:completion_len, :]
    student_non_pad_mask = student_non_pad_mask[:completion_len]
    teacher_non_pad_mask = teacher_non_pad_mask[:completion_len]

    # Combined mask: only compute loss where both have non-pad tokens
    loss_mask = student_non_pad_mask & teacher_non_pad_mask

    if not loss_mask.any():
        return torch.tensor(0.0, device=student_device, requires_grad=True), {
            "loss": 0.0,
            "completion_tokens": 0,
            "teacher_completion_len": len(teacher_completion_ids),
        }

    # Apply temperature
    student_logits_scaled = student_logits_completion / hparams.temperature
    teacher_logits_scaled = teacher_logits_completion / hparams.temperature

    # Get top-K indices from student's distribution (student stays on-policy)
    top_k = hparams.top_k_distillation
    if top_k > 0 and top_k < student_logits_scaled.shape[-1]:
        _, top_k_indices = torch.topk(student_logits_scaled, top_k, dim=-1)

        # Gather top-K logits from both models at student's top-K tokens
        student_topk_logits = torch.gather(student_logits_scaled, -1, top_k_indices)
        teacher_topk_logits = torch.gather(teacher_logits_scaled, -1, top_k_indices)

        # Softmax over top-K only
        student_probs = F.softmax(student_topk_logits, dim=-1)
        student_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
    else:
        student_probs = F.softmax(student_logits_scaled, dim=-1)
        student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)

    # KL(student || teacher) - shift student toward teacher
    kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    # Apply loss mask: only average over non-pad positions
    masked_kl = kl_per_token * loss_mask.float()
    num_valid_tokens = loss_mask.sum().item()
    loss = masked_kl.sum() / max(num_valid_tokens, 1)

    metrics = {
        "loss": loss.item(),
        "completion_tokens": num_valid_tokens,
        "kl_per_token": (masked_kl.sum() / max(num_valid_tokens, 1)).item(),
        "teacher_completion_len": len(teacher_completion_ids),
    }

    return loss, metrics


def sdpo_train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    hparams: SDPOHparams,
    rollout_fn: Callable[[AutoModelForCausalLM, AutoTokenizer, Any, int, float], List[RolloutResult]],
    get_feedback_fn: Callable[[str, str, Any], FeedbackResult],
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    include_prior_solutions: bool = False,
    verify_solution_fn: Optional[Callable[[str, str, Any], bool]] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    output_dir: str = "outputs",
    prior_solutions_store: Optional[Dict[str, str]] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    SDPO Training Loop with EMA teacher and Top-K distillation.

    Args:
        model: The language model to train
        tokenizer: The tokenizer
        dataloader: DataLoader yielding training examples
        hparams: Training hyperparameters
        rollout_fn: Function that generates N completions given (model, tokenizer, example, num_rollouts, temperature)
        get_feedback_fn: Function that gets environment feedback
        validators: List of (Validator, ValidatorRunConfig) tuples
        include_prior_solutions: Whether to include prior correct solutions in teacher context
        verify_solution_fn: Function to verify if a solution is correct
        optimizer: Optional optimizer
        scheduler: Optional LR scheduler
        output_dir: Directory to save checkpoints
        prior_solutions_store: Optional dict of prior solutions
        wandb_project: W&B project name
        wandb_run_name: W&B run name

    Returns:
        Dictionary with training metrics and history
    """
    os.makedirs(output_dir, exist_ok=True)

    if include_prior_solutions and verify_solution_fn is None:
        raise ValueError("verify_solution_fn is required when include_prior_solutions=True")

    # Initialize accelerator
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=hparams.gradient_accumulation_steps,
        adjust_scheduler=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb" if wandb_project else None,
    )

    # Initialize wandb
    if accelerator.is_main_process and wandb_project:
        accelerator.init_trackers(
            project_name=wandb_project,
            config={
                "learning_rate": hparams.learning_rate,
                "num_epochs": hparams.num_epochs,
                "gradient_accumulation_steps": hparams.gradient_accumulation_steps,
                "teacher_ema_rate": hparams.teacher_ema_rate,
                "top_k_distillation": hparams.top_k_distillation,
                "num_rollouts": hparams.num_rollouts,
                "include_prior_solutions": include_prior_solutions,
                "distill_on_regen": hparams.distill_on_regen,
                "regen_temperature": hparams.regen_temperature if hparams.distill_on_regen else None,
                "include_student_attempt": hparams.include_student_attempt,
            },
            init_kwargs={"wandb": {"name": wandb_run_name}} if wandb_run_name else {},
        )

    # Initialize optimizer
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )

    # Initialize prior solutions store
    if prior_solutions_store is None:
        prior_solutions_store = {}

    # Prepare student model with accelerator FIRST
    # This moves the model to the correct device(s) and wraps with DDP
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # Initialize EMA teacher AFTER accelerator.prepare()
    # This ensures the teacher is created from the already-prepared model
    # and lives on the same device(s) as the student
    ema_decay = 1.0 - hparams.teacher_ema_rate
    unwrapped_for_teacher = accelerator.unwrap_model(model)
    teacher = EMATeacher(unwrapped_for_teacher, decay=ema_decay)
    # Teacher is already on the right device since we copied from the prepared model

    model.train()

    # Training state
    global_step = 0
    total_loss = 0.0
    metrics_history = defaultdict(list)
    validation_history = defaultdict(list)

    # Log training config (main process only)
    if accelerator.is_main_process:
        logger.info(f"Initialized EMA teacher with decay={ema_decay}")
        logger.info(f"Teacher device: {next(teacher.model.parameters()).device}")
        logger.info(f"Starting SDPO training with {hparams.num_epochs} epochs")
        logger.info(f"Teacher EMA rate: {hparams.teacher_ema_rate}")
        logger.info(f"Top-K distillation: {hparams.top_k_distillation}")
        logger.info(f"Num rollouts: {hparams.num_rollouts}")
        if hparams.distill_on_regen:
            logger.info(f"Distill-on-regen: ENABLED (temp={hparams.regen_temperature})")
        else:
            logger.info("Distill-on-regen: disabled (using standard SDPO)")
        logger.info(f"Validators: {[v.name for v, _ in validators]}")
        logger.info(f"Using {accelerator.num_processes} GPUs")
        logger.info(f"Dataloader has {len(dataloader)} batches")

    for epoch in range(hparams.num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch + 1}/{hparams.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            batch_loss = 0.0
            batch_metrics = defaultdict(float)
            num_rollouts_processed = 0

            # Handle batched data
            if isinstance(batch, dict):
                if isinstance(batch.get('prompt', batch.get('question', batch.get('question_content'))), list):
                    batch_size = len(batch.get('prompt', batch.get('question', batch.get('question_content', []))))
                    examples = [{k: v[i] if isinstance(v, list) else v for k, v in batch.items()}
                               for i in range(batch_size)]
                else:
                    examples = [batch]
            else:
                examples = [batch]

            with accelerator.accumulate(model):
                for example in examples:
                    example_id = example.get('id', example.get('question_title', str(hash(str(example)))))
                    unwrapped_model = accelerator.unwrap_model(model)

                    # Generate multiple rollouts
                    unwrapped_model.eval()
                    rollouts = rollout_fn(
                        unwrapped_model,
                        tokenizer,
                        example,
                        hparams.num_rollouts,
                        hparams.rollout_temperature,
                    )
                    unwrapped_model.train()

                    # Process each rollout
                    for rollout in rollouts:
                        prompt = rollout.prompt
                        completion = rollout.completion

                        # Get feedback
                        feedback_result = get_feedback_fn(prompt, completion, example)
                        feedback = feedback_result.feedback_text

                        # Store correct solutions
                        if include_prior_solutions and feedback_result.success:
                            if verify_solution_fn(prompt, completion, example):
                                prior_solutions_store[example_id] = completion

                        prior_solution = prior_solutions_store.get(example_id) if include_prior_solutions else None

                        # Determine student attempt to include in teacher context
                        student_attempt = completion if hparams.include_student_attempt else None

                        # Compute loss (standard SDPO or distill-on-regen)
                        # Teacher is a separate model - no weight swapping needed
                        if hparams.distill_on_regen:
                            loss, metrics = compute_distill_on_regen_loss(
                                student_model=model,
                                teacher_model=teacher.model,  # Separate EMA teacher model
                                tokenizer=tokenizer,
                                prompt=prompt,
                                student_completion=completion,
                                feedback=feedback,
                                hparams=hparams,
                                student_attempt=student_attempt,
                            )
                        else:
                            loss, metrics = compute_sdpo_loss(
                                student_model=model,
                                teacher_model=teacher.model,  # Separate EMA teacher model
                                tokenizer=tokenizer,
                                prompt=prompt,
                                completion=completion,
                                feedback=feedback,
                                prior_solution=prior_solution,
                                hparams=hparams,
                                student_attempt=student_attempt,
                            )

                        # Backward
                        accelerator.backward(loss)

                        batch_loss += loss.item()
                        for k, v in metrics.items():
                            batch_metrics[k] += v
                        num_rollouts_processed += 1

                # Average metrics
                if num_rollouts_processed > 0:
                    batch_loss /= num_rollouts_processed
                    for k in batch_metrics:
                        batch_metrics[k] /= num_rollouts_processed

                # Gradient step
                if accelerator.sync_gradients:
                    if hparams.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                # Update EMA teacher and sync across processes
                if accelerator.sync_gradients:
                    teacher.update(accelerator.unwrap_model(model))
                    # Sync teacher weights across all GPUs to ensure consistency
                    teacher.sync_across_processes(accelerator)

            # Logging and validation
            if accelerator.sync_gradients:
                global_step += 1

                # Average loss across all processes for accurate logging
                loss_tensor = torch.tensor(batch_loss, device=accelerator.device)
                avg_loss_tensor = accelerator.reduce(loss_tensor, reduction="mean")
                avg_batch_loss = avg_loss_tensor.item()

                total_loss += avg_batch_loss
                lr = optimizer.param_groups[0]['lr']

                # Log to wandb every step (main process only)
                if accelerator.is_main_process:
                    log_dict = {
                        "train/loss": avg_batch_loss,
                        "train/lr": lr,
                        "train/kl_per_token": batch_metrics.get('kl_per_token', 0),
                        "train/completion_tokens": batch_metrics.get('completion_tokens', 0),
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    if hparams.distill_on_regen:
                        log_dict["train/teacher_completion_len"] = batch_metrics.get('teacher_completion_len', 0)
                    accelerator.log(log_dict, step=global_step)

                # Console logging (main process only)
                if accelerator.is_main_process and (global_step % hparams.log_interval == 0 or global_step == 1):
                    logged_avg_loss = total_loss / min(global_step, hparams.log_interval)
                    logger.info(f"Step {global_step} | Loss: {logged_avg_loss:.4f} | LR: {lr:.2e}")
                    metrics_history['loss'].append(logged_avg_loss)
                    metrics_history['step'].append(global_step)
                    total_loss = 0.0

                # Validation (main process only)
                if accelerator.is_main_process and (global_step % hparams.validation_interval == 0 or global_step == 1):
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.eval()

                    for validator, val_config in validators:
                        logger.info(f"Running validator: {validator.name}")
                        try:
                            score = validator.validate(
                                model=unwrapped_model,
                                tokenizer=tokenizer,
                                batch_size=val_config.batch_size,
                                max_new_tokens=val_config.max_new_tokens,
                                max_seq_length=val_config.max_seq_length,
                            )
                            logger.info(f"Validator {validator.name}: {score:.4f}")
                            validation_history[validator.name].append({
                                'step': global_step,
                                'score': score,
                            })
                            accelerator.log({f"val/{validator.name}": score}, step=global_step)

                        except Exception as e:
                            logger.error(f"Validator {validator.name} failed: {e}")

                    model.train()

                # Non-main processes wait for validation to complete
                accelerator.wait_for_everyone()

                # Save checkpoint
                if global_step % hparams.save_interval == 0 and accelerator.is_main_process:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                    if include_prior_solutions:
                        with open(os.path.join(checkpoint_dir, "prior_solutions.json"), "w") as f:
                            json.dump(prior_solutions_store, f)

                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Final validation and checkpoint (main process only)
    final_scores = {}
    if accelerator.is_main_process:
        logger.info("Running final validation")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()

        for validator, val_config in validators:
            try:
                score = validator.validate(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    batch_size=val_config.batch_size,
                    max_new_tokens=val_config.max_new_tokens,
                    max_seq_length=val_config.max_seq_length,
                )
                final_scores[validator.name] = score
                logger.info(f"Final {validator.name}: {score:.4f}")
                accelerator.log({f"val/{validator.name}": score}, step=global_step)

            except Exception as e:
                logger.error(f"Final validation for {validator.name} failed: {e}")

        # Save final checkpoint
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        if wandb_project:
            accelerator.end_training()

    # Ensure all processes wait for main to finish
    accelerator.wait_for_everyone()

    return {
        'metrics_history': dict(metrics_history),
        'validation_history': dict(validation_history),
        'final_scores': final_scores,
        'prior_solutions': prior_solutions_store if include_prior_solutions else None,
    }
