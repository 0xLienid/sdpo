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


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: AutoModelForCausalLM, decay: float = 0.99):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (1 - update_rate). Higher = slower updates.
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: AutoModelForCausalLM):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: AutoModelForCausalLM):
        """Apply EMA parameters to model (backup current params first)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: AutoModelForCausalLM):
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


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
) -> List[Dict[str, str]]:
    """Build chat messages for teacher (feedback context + prompt + completion)."""
    teacher_context_parts = [f"## Feedback\n{feedback}"]
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
    teacher_model: AutoModelForCausalLM,  # EMA model with shadow weights applied
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    feedback: str,
    prior_solution: Optional[str],
    hparams: SDPOHparams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the SDPO loss for a single example.

    Loss = Reverse-KL with Top-K distillation:
        L = sum over top-K tokens of: teacher_prob * (log(teacher_prob) - log(student_prob))

    Args:
        student_model: The student model (current weights)
        teacher_model: The teacher model (EMA weights, same object with shadow applied)
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
    device = next(student_model.parameters()).device
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length

    # Build student messages (prompt + completion as assistant)
    student_messages = build_student_messages(prompt, completion)
    student_full = tokenizer.apply_chat_template(
        student_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Build teacher messages (feedback context + prompt + completion as assistant)
    teacher_messages = build_teacher_messages(prompt, completion, feedback, prior_solution)
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
    teacher_context_parts = [f"## Feedback\n{feedback}"]
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
    ).to(device)

    teacher_encoding = tokenizer(
        teacher_full, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(device)

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
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "completion_tokens": 0}

    student_logits_completion = student_logits[0, student_prompt_len-1:student_prompt_len-1+completion_len, :]
    teacher_logits_completion = teacher_logits[0, teacher_prompt_len-1:teacher_prompt_len-1+completion_len, :]

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

    # Initialize EMA teacher
    ema_decay = 1.0 - hparams.teacher_ema_rate
    ema_teacher = EMAModel(model, decay=ema_decay)
    logger.info(f"Initialized EMA teacher with decay={ema_decay}")

    # Initialize prior solutions store
    if prior_solutions_store is None:
        prior_solutions_store = {}

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    model.train()

    # Training state
    global_step = 0
    total_loss = 0.0
    metrics_history = defaultdict(list)
    validation_history = defaultdict(list)

    logger.info(f"Starting SDPO training with {hparams.num_epochs} epochs")
    logger.info(f"Teacher EMA rate: {hparams.teacher_ema_rate}")
    logger.info(f"Top-K distillation: {hparams.top_k_distillation}")
    logger.info(f"Num rollouts: {hparams.num_rollouts}")
    logger.info(f"Validators: {[v.name for v, _ in validators]}")
    logger.info(f"Using {accelerator.num_processes} GPUs")
    logger.info(f"Dataloader has {len(dataloader)} batches")

    for epoch in range(hparams.num_epochs):
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

                        # Apply EMA weights to get teacher
                        ema_teacher.apply_shadow(unwrapped_model)

                        # Compute loss
                        loss, metrics = compute_sdpo_loss(
                            student_model=model,
                            teacher_model=unwrapped_model,  # Has EMA weights applied
                            tokenizer=tokenizer,
                            prompt=prompt,
                            completion=completion,
                            feedback=feedback,
                            prior_solution=prior_solution,
                            hparams=hparams,
                        )

                        # Restore student weights
                        ema_teacher.restore(unwrapped_model)

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

                total_loss += batch_loss

                # Gradient step
                if accelerator.sync_gradients:
                    if hparams.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                # Update EMA teacher
                if accelerator.sync_gradients:
                    ema_teacher.update(accelerator.unwrap_model(model))

            # Logging and validation
            if accelerator.sync_gradients:
                global_step += 1
                lr = optimizer.param_groups[0]['lr']

                # Log to wandb every step
                if accelerator.is_main_process:
                    accelerator.log({
                        "train/loss": batch_loss,
                        "train/lr": lr,
                        "train/kl_per_token": batch_metrics.get('kl_per_token', 0),
                        "train/completion_tokens": batch_metrics.get('completion_tokens', 0),
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }, step=global_step)

                # Console logging
                if global_step % hparams.log_interval == 0 or global_step == 1:
                    avg_loss = total_loss / min(global_step, hparams.log_interval)
                    logger.info(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    metrics_history['loss'].append(avg_loss)
                    metrics_history['step'].append(global_step)
                    total_loss = 0.0

                # Validation
                if global_step % hparams.validation_interval == 0 or global_step == 1:
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

                            if accelerator.is_main_process:
                                accelerator.log({f"val/{validator.name}": score}, step=global_step)

                        except Exception as e:
                            logger.error(f"Validator {validator.name} failed: {e}")

                    model.train()

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

    # Final validation
    logger.info("Running final validation")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    final_scores = {}
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

            if accelerator.is_main_process:
                accelerator.log({f"val/{validator.name}": score}, step=global_step)

        except Exception as e:
            logger.error(f"Final validation for {validator.name} failed: {e}")

    # Save final checkpoint
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    if accelerator.is_main_process and wandb_project:
        accelerator.end_training()

    return {
        'metrics_history': dict(metrics_history),
        'validation_history': dict(validation_history),
        'final_scores': final_scores,
        'prior_solutions': prior_solutions_store if include_prior_solutions else None,
    }
