"""
Convergence Speed Experiment

Measures how different SDPO teacher configurations affect training convergence
by tracking pass@k on private test cases every few gradient steps.

Six conditions:
  1. code_only          — teacher sees question + extracted code + feedback
  2. with_thinking      — teacher sees question + full completion + feedback
  3. entropy_weighted   — same as code_only, KL weighted by student entropy per ventile
  4. reasoning_augmented — reasoning in assistant turn (prepended to completion)
  5. reasoning_in_prompt — reasoning in user prompt, code-only attempt
  6. reasoning_in_prompt_thinking — reasoning in user prompt, full completion attempt

Usage:
    uv run python -m analysis.experiment_convergence
    uv run python -m analysis.experiment_convergence --conditions code_only entropy_weighted
    uv run python -m analysis.experiment_convergence --num-problems 3 --num-rollouts 2 --num-epochs 1 --eval-k 2 --eval-interval 2
"""

import argparse
import copy
import json
import logging
import os
import random
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.utils import bin_into_ventiles
from data_modules.livecodebench.code_execution import (
    extract_python_code,
    run_test_cases,
)
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import (
    EMATeacher,
    SDPOHparams,
    build_student_messages,
)

logger = logging.getLogger(__name__)

ALL_CONDITIONS = [
    "code_only", "with_thinking", "entropy_weighted",
    "reasoning_augmented",
    "reasoning_in_prompt", "reasoning_in_prompt_thinking",
]


# ---------------------------------------------------------------------------
# Pass@k evaluation
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n=total samples, c=correct, k=budget."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def evaluate_pass_at_k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: LiveCodeBenchDataset,
    k: int = 4,
    n: int = 16,
    temperature: float = 0.8,
    max_new_tokens: int = 4096,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate pass@k using private test cases.

    Returns (mean_pass_at_k, mean_sample_pass_rate, per_problem_details).
    """
    if n < k:
        raise ValueError(f"eval n must be >= k, got n={n}, k={k}")

    model.eval()
    gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
    if gc_was_enabled:
        model.gradient_checkpointing_disable()

    per_problem = []

    for idx in range(len(dataset)):
        example = dataset[idx]
        title = example.get("question_title", f"Problem {idx}")

        # Generate n completions, then estimate pass@k.
        rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=n, temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Get private test cases
        private_tests = dataset.get_private_test_cases(idx)

        # Test each completion
        correct = 0
        for rollout in rollouts:
            code = extract_python_code(rollout.completion)
            all_passed, _ = run_test_cases(
                code, private_tests, timeout_seconds=10)
            if all_passed:
                correct += 1

        score = pass_at_k(len(rollouts), correct, k)
        sample_pass_rate = correct / \
            len(rollouts) if len(rollouts) > 0 else 0.0
        per_problem.append({
            "idx": idx, "title": title,
            "correct": correct,
            "total": len(rollouts),
            "pass_at_k": score,
            "sample_pass_rate": sample_pass_rate,
        })

    if gc_was_enabled:
        model.gradient_checkpointing_enable()
    model.train()

    mean_score = sum(p["pass_at_k"] for p in per_problem) / \
        len(per_problem) if per_problem else 0.0
    mean_sample_rate = sum(p["sample_pass_rate"] for p in per_problem) / \
        len(per_problem) if per_problem else 0.0
    return mean_score, mean_sample_rate, per_problem


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducible dataset sampling and decoding."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Helpers for batched completion extraction and loss
# ---------------------------------------------------------------------------

def _extract_completion_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: List[int],
    completion_lens: List[int],
    max_comp_len: int,
) -> torch.Tensor:
    """
    Extract aligned completion logits from batched forward-pass output.

    Slices each sequence's completion region and pads to max_comp_len via
    torch.cat + torch.stack so autograd flows through to the original logits.

    Args:
        logits: (batch, max_seq_len, vocab) from model forward pass.
        attention_mask: (batch, max_seq_len) from tokenizer encoding.
        prompt_lens: Per-element prompt token count.
        completion_lens: Per-element completion token count (0 for skip).
        max_comp_len: Target second dimension for the output tensor.

    Returns:
        (batch, max_comp_len, vocab) with zero-padding beyond each element's
        completion length. Padded positions should be masked in the loss.
    """
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]
    device = logits.device

    comp_list = []
    for i in range(batch_size):
        cl = completion_lens[i]
        if cl > 0:
            start = prompt_lens[i] - 1
            comp = logits[i, start:start + cl]  # (cl, vocab)
            if cl < max_comp_len:
                comp = torch.cat(
                    [comp, comp.new_zeros(max_comp_len - cl, vocab_size)], dim=0)
        else:
            comp = logits.new_zeros(max_comp_len, vocab_size)
        comp_list.append(comp)

    return torch.stack(comp_list)  # (batch, max_comp_len, vocab)


def _batched_topk_kl_loss(
    student_comp: torch.Tensor,
    teacher_comp: torch.Tensor,
    comp_mask: torch.Tensor,
    top_k: int,
    temperature: float,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, int]:
    """
    Batched top-K KL loss on aligned completion logits.

    Args:
        student_comp: (batch, max_comp, vocab) — requires grad.
        teacher_comp: (batch, max_comp, vocab) — detached / no grad.
        comp_mask: (batch, max_comp) — 1 for real positions, 0 for padding.
        top_k: Number of student top-K tokens.
        temperature: Softmax temperature.
        weights: Optional (batch, max_comp) per-token weights.

    Returns:
        (loss, avg_kl, total_tokens)
    """
    total_tokens = int(comp_mask.sum().item())
    if total_tokens == 0:
        device = student_comp.device
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    s_scaled = student_comp / temperature
    t_scaled = teacher_comp / temperature

    # Batched top-K on student logits — (batch, max_comp, top_k)
    _, topk_idx = torch.topk(s_scaled, top_k, dim=-1)
    s_topk = torch.gather(s_scaled, -1, topk_idx)
    t_topk = torch.gather(t_scaled, -1, topk_idx)
    del s_scaled, t_scaled

    s_probs = F.softmax(s_topk, dim=-1)
    s_log_probs = F.log_softmax(s_topk, dim=-1)
    t_log_probs = F.log_softmax(t_topk, dim=-1)
    kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # (batch, max_comp)

    if weights is not None:
        loss = (kl * weights * comp_mask).sum() / comp_mask.sum()
    else:
        loss = (kl * comp_mask).sum() / comp_mask.sum()

    avg_kl = (kl.detach() * comp_mask).sum().item() / total_tokens
    return loss, avg_kl, total_tokens


def _generate_reasoning_batched(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    student_codes: List[str],
    feedbacks: List[str],
    max_prompt_length: int,
    teacher_device: torch.device
) -> List[str]:
    """
    Batched reasoning generation using left-padded inputs.

    Returns list of decoded reasoning strings.
    """
    batch_size = len(prompts)
    formatted = []
    for i in range(batch_size):
        rp = (
            f"## Question\n{prompts[i]}\n\n"
            f"## Student Code\n```python\n{student_codes[i]}\n```\n\n"
            f"## Feedback\n{feedbacks[i]}\n\n"
            "Analyze the student's attempt based on the feedback. If the "
            "student was correct simply say so. If the student's code was "
            "incorrect identify where the student went wrong, and how they "
            "can fix it.\n\n/no_think"
        )
        formatted.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": rp}],
            tokenize=False, add_generation_prompt=True))

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoding = tokenizer(
        formatted, return_tensors="pt", truncation=True,
        max_length=max_prompt_length + 512, padding=True,
    ).to(teacher_device)
    tokenizer.padding_side = orig_pad_side

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=1024,
            do_sample=True, temperature=0.7, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_lens = encoding.attention_mask.sum(dim=1).tolist()
    reasoning_texts = []
    for i, in_len in enumerate(input_lens):
        reasoning_ids = outputs[i, int(in_len):]
        reasoning_texts.append(
            tokenizer.decode(reasoning_ids, skip_special_tokens=True)
            .replace("<think>", "")
            .replace("</think>", "")
            .strip()
        )
    del outputs, encoding
    return reasoning_texts


def _build_teacher_context_ablation_style(
    prompt: str,
    feedback: str,
    attempt_text: Optional[str] = None,
    attempt_is_full_completion: bool = False,
    reasoning_text: Optional[str] = None,
) -> str:
    """Mirror teacher prompt structure used in experiment_2_5."""
    parts = [f"## Question\n{prompt}"]

    if attempt_text is not None:
        if attempt_is_full_completion:
            parts.append(f"## Previous Attempt (including reasoning)\n{attempt_text}")
        else:
            parts.append(f"## Previous Attempt\n```python\n{attempt_text}\n```")

    parts.append(f"## Feedback (from environment) for the previous attempt\n{feedback}")

    if reasoning_text is not None:
        parts.append(
            f"## Analysis of Previous Attempt and Guidance for Improvement\n{reasoning_text}"
        )

    parts.append("Correctly solve the original question.")
    return "\n\n".join(parts)


def compute_sdpo_loss_batched_ablation_style(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    feedbacks: List[str],
    hparams: SDPOHparams,
    student_attempts: Optional[List[Optional[str]]] = None,
    attempt_is_full_completion: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Batched SDPO KL loss with teacher prompts matching experiment_2_5."""
    batch_size = len(prompts)
    device = next(student_model.parameters()).device
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    teacher_device = next(teacher_model.parameters()).device
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    top_k = hparams.top_k_distillation
    temperature = hparams.temperature

    if student_attempts is None:
        student_attempts = [None] * batch_size

    student_fulls, teacher_fulls = [], []
    student_prompt_lens, teacher_prompt_lens = [], []

    for i in range(batch_size):
        student_messages = build_student_messages(prompts[i], completions[i])
        student_fulls.append(tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False))
        student_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[i]}],
            tokenize=False, add_generation_prompt=True)
        student_prompt_lens.append(len(tokenizer(
            student_prompt_only, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

        teacher_context = _build_teacher_context_ablation_style(
            prompt=prompts[i],
            feedback=feedbacks[i],
            attempt_text=student_attempts[i],
            attempt_is_full_completion=attempt_is_full_completion,
        )
        teacher_messages = [
            {"role": "user", "content": teacher_context},
            {"role": "assistant", "content": completions[i]},
        ]
        teacher_fulls.append(tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False))
        teacher_prompt_only = tokenizer.apply_chat_template(
            [teacher_messages[0]], tokenize=False, add_generation_prompt=True)
        teacher_prompt_lens.append(len(tokenizer(
            teacher_prompt_only, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    student_encoding = tokenizer(
        student_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(device)
    teacher_encoding = tokenizer(
        teacher_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(teacher_device)
    tokenizer.padding_side = orig_pad_side

    completion_lens = []
    for i in range(batch_size):
        s_seq = (student_encoding.attention_mask[i] == 1).sum().item()
        t_seq = (teacher_encoding.attention_mask[i] == 1).sum().item()
        completion_lens.append(max(0, min(
            s_seq - student_prompt_lens[i],
            t_seq - teacher_prompt_lens[i])))

    max_comp_len = max(completion_lens) if any(c > 0 for c in completion_lens) else 1
    total_tokens = sum(completion_lens)
    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    comp_mask = torch.zeros(batch_size, max_comp_len, device=device)
    for i in range(batch_size):
        comp_mask[i, :completion_lens[i]] = 1.0

    student_outputs = student_model(**student_encoding)
    student_comp = _extract_completion_logits(
        student_outputs.logits, student_encoding.attention_mask,
        student_prompt_lens, completion_lens, max_comp_len)
    del student_outputs

    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_comp = _extract_completion_logits(
            teacher_outputs.logits, teacher_encoding.attention_mask,
            teacher_prompt_lens, completion_lens, max_comp_len)
        del teacher_outputs
        teacher_comp = teacher_comp.to(device)

    loss, avg_kl, total_tokens = _batched_topk_kl_loss(
        student_comp, teacher_comp, comp_mask, top_k, temperature)

    return loss, {"loss": loss.item(), "completion_tokens": total_tokens, "kl_per_token": avg_kl}


# ---------------------------------------------------------------------------
# Condition 3: Entropy-weighted SDPO loss
# ---------------------------------------------------------------------------

def compute_entropy_weighted_loss(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    feedbacks: List[str],
    hparams: SDPOHparams,
    student_attempts: List[Optional[str]],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SDPO loss with per-token KL weighted by student entropy per ventile.

    Batched forward passes, batched top-K / KL, masked loss.
    Entropy is computed on full-vocab student logits (detached) before
    top-K reduction, then used to build per-token weights.
    """
    batch_size = len(prompts)
    device = next(student_model.parameters()).device
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    teacher_device = next(teacher_model.parameters()).device
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    top_k = hparams.top_k_distillation
    temperature = hparams.temperature

    if student_attempts is None:
        student_attempts = [None] * batch_size

    # Build sequences
    student_fulls, teacher_fulls = [], []
    student_prompt_lens, teacher_prompt_lens = [], []

    for i in range(batch_size):
        student_messages = build_student_messages(prompts[i], completions[i])
        student_fulls.append(tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False))

        student_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[i]}],
            tokenize=False, add_generation_prompt=True)
        student_prompt_lens.append(len(tokenizer(
            student_prompt_only, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

        teacher_context = _build_teacher_context_ablation_style(
            prompt=prompts[i],
            feedback=feedbacks[i],
            attempt_text=student_attempts[i],
            attempt_is_full_completion=False,
        )
        teacher_messages = [
            {"role": "user", "content": teacher_context},
            {"role": "assistant", "content": completions[i]},
        ]
        teacher_fulls.append(tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False))

        teacher_prompt_only = tokenizer.apply_chat_template(
            [teacher_messages[0]], tokenize=False, add_generation_prompt=True)
        teacher_prompt_lens.append(len(tokenizer(
            teacher_prompt_only, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

    # Batch tokenize
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    student_encoding = tokenizer(
        student_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(device)
    teacher_encoding = tokenizer(
        teacher_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(teacher_device)
    tokenizer.padding_side = orig_pad_side

    # Compute completion lengths
    completion_lens = []
    for i in range(batch_size):
        s_seq = (student_encoding.attention_mask[i] == 1).sum().item()
        t_seq = (teacher_encoding.attention_mask[i] == 1).sum().item()
        completion_lens.append(max(0, min(
            s_seq - student_prompt_lens[i],
            t_seq - teacher_prompt_lens[i])))

    max_comp_len = max(completion_lens) if any(c > 0 for c in completion_lens) else 1
    total_tokens = sum(completion_lens)
    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    comp_mask = torch.zeros(batch_size, max_comp_len, device=device)
    for i in range(batch_size):
        comp_mask[i, :completion_lens[i]] = 1.0

    # Batched student forward
    student_outputs = student_model(**student_encoding)
    student_logits = student_outputs.logits
    del student_outputs

    # Extract aligned completion logits — (batch, max_comp, vocab)
    student_comp = _extract_completion_logits(
        student_logits, student_encoding.attention_mask,
        student_prompt_lens, completion_lens, max_comp_len)
    del student_logits

    # Compute entropy on full-vocab logits (detached) for weighting
    with torch.no_grad():
        ent_scaled = student_comp.detach() / temperature
        ent_probs = F.softmax(ent_scaled, dim=-1)
        ent_log_probs = F.log_softmax(ent_scaled, dim=-1)
        entropy = -(ent_probs * ent_log_probs).sum(dim=-1)  # (batch, max_comp)
        del ent_scaled, ent_probs, ent_log_probs

    # Build per-token entropy weights (detached, no grad)
    entropy_weights = torch.ones(batch_size, max_comp_len, device=device)
    for i in range(batch_size):
        cl = completion_lens[i]
        if cl <= 0:
            continue
        ventile_means = bin_into_ventiles(entropy[i, :cl].cpu())
        total_entropy = sum(v for v in ventile_means if v == v)
        if total_entropy > 0:
            for v in range(20):
                v_start = int(v * cl / 20)
                v_end = int((v + 1) * cl / 20)
                if v_start < v_end and ventile_means[v] == ventile_means[v]:
                    entropy_weights[i, v_start:v_end] = (
                        ventile_means[v] / total_entropy) * 20.0

    # Batched teacher forward
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_logits = teacher_outputs.logits
        del teacher_outputs

        teacher_comp = _extract_completion_logits(
            teacher_logits, teacher_encoding.attention_mask,
            teacher_prompt_lens, completion_lens, max_comp_len)
        del teacher_logits
        teacher_comp = teacher_comp.to(device)

    # Batched top-K KL with entropy weights
    loss, avg_kl, total_tokens = _batched_topk_kl_loss(
        student_comp, teacher_comp, comp_mask, top_k, temperature,
        weights=entropy_weights)

    return loss, {"loss": loss.item(), "completion_tokens": total_tokens, "kl_per_token": avg_kl}


# ---------------------------------------------------------------------------
# Condition 4: Reasoning-augmented SDPO loss (reasoning in assistant turn)
# ---------------------------------------------------------------------------

def compute_reasoning_augmented_loss(
    student_model: AutoModelForCausalLM,
    teacher_model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    feedbacks: List[str],
    student_attempts: List[str],
    hparams: SDPOHparams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Teacher generates reasoning, prepended to the completion in the assistant turn.

    Batched reasoning generation (left-padded), batched forward passes,
    batched top-K KL with completion mask.
    """
    device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    batch_size = len(prompts)
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    top_k = hparams.top_k_distillation
    temperature = hparams.temperature

    # Step 1: Batched reasoning generation
    reasoning_texts = _generate_reasoning_batched(
        teacher_model, tokenizer, prompts, student_attempts, feedbacks,
        hparams.max_prompt_length, teacher_device)

    # Step 2: Build all sequences.
    # We bypass apply_chat_template for the assistant content to match the
    # reasoning-in-response logic in experiment_2_5 and avoid </think> mangling.
    student_fulls, teacher_fulls = [], []
    student_prompt_lens, teacher_prefix_lens = [], []

    for i in range(batch_size):
        teacher_context = _build_teacher_context_ablation_style(
            prompt=prompts[i],
            feedback=feedbacks[i],
            attempt_text=student_attempts[i],
            attempt_is_full_completion=True,
        )
        teacher_user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_context}],
            tokenize=False, add_generation_prompt=True,
        )
        teacher_full_text = (
            teacher_user_text + f"\n{reasoning_texts[i]}\n\n{completions[i]}\n<im_end>"
        )
        teacher_prefix_text = teacher_user_text + f"\n{reasoning_texts[i]}"
        teacher_fulls.append(teacher_full_text)
        teacher_prefix_lens.append(len(tokenizer(
            teacher_prefix_text, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

        student_user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[i]}],
            tokenize=False, add_generation_prompt=True,
        )
        student_full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompts[i]},
                {"role": "assistant", "content": completions[i]},
            ],
            tokenize=False, add_generation_prompt=False,
        )
        student_fulls.append(student_full_text)
        student_prompt_lens.append(len(tokenizer(
            student_user_text, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

    # Step 3: Batch tokenize (right-padded)
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    student_encoding = tokenizer(
        student_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(device)
    teacher_encoding = tokenizer(
        teacher_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(teacher_device)
    tokenizer.padding_side = orig_pad_side

    # Step 4: Compute completion lengths
    completion_lens = []
    for i in range(batch_size):
        s_seq = (student_encoding.attention_mask[i] == 1).sum().item()
        t_seq = (teacher_encoding.attention_mask[i] == 1).sum().item()
        completion_lens.append(max(0, min(
            s_seq - student_prompt_lens[i],
            t_seq - teacher_prefix_lens[i])))

    max_comp_len = max(completion_lens) if any(c > 0 for c in completion_lens) else 1
    total_tokens = sum(completion_lens)
    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    comp_mask = torch.zeros(batch_size, max_comp_len, device=device)
    for i in range(batch_size):
        comp_mask[i, :completion_lens[i]] = 1.0

    # Step 5: Batched student forward + extraction
    student_outputs = student_model(**student_encoding)
    student_comp = _extract_completion_logits(
        student_outputs.logits, student_encoding.attention_mask,
        student_prompt_lens, completion_lens, max_comp_len)
    del student_outputs

    # Step 6: Batched teacher forward + extraction
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_comp = _extract_completion_logits(
            teacher_outputs.logits, teacher_encoding.attention_mask,
            teacher_prefix_lens, completion_lens, max_comp_len)
        del teacher_outputs
        teacher_comp = teacher_comp.to(device)

    # Step 7: Batched loss
    loss, avg_kl, total_tokens = _batched_topk_kl_loss(
        student_comp, teacher_comp, comp_mask, top_k, temperature)

    return loss, {"loss": loss.item(), "completion_tokens": total_tokens, "kl_per_token": avg_kl}


# ---------------------------------------------------------------------------
# Conditions 5-6: Reasoning in user prompt SDPO loss
# ---------------------------------------------------------------------------

def compute_reasoning_in_prompt_loss(
    student_model: AutoModelForCausalLM,
    teacher_model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    feedbacks: List[str],
    student_codes: List[str],
    hparams: SDPOHparams,
    teacher_attempts: Optional[List[str]] = None,
    attempt_is_full_completion: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reasoning placed in the user prompt so both sides go through
    apply_chat_template uniformly.

    Batched reasoning generation (left-padded), batched forward passes,
    batched top-K KL with completion mask.
    """
    device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    batch_size = len(prompts)
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    top_k = hparams.top_k_distillation
    temperature = hparams.temperature

    # Step 1: Batched reasoning generation
    reasoning_texts = _generate_reasoning_batched(
        teacher_model, tokenizer, prompts, student_codes, feedbacks,
        hparams.max_prompt_length, teacher_device)

    # Step 2: Build all sequences
    student_fulls, teacher_fulls = [], []
    student_prompt_lens, teacher_prompt_lens = [], []

    for i in range(batch_size):
        attempt = (teacher_attempts[i]
                   if teacher_attempts else student_codes[i])
        teacher_context = _build_teacher_context_ablation_style(
            prompt=prompts[i],
            feedback=feedbacks[i],
            attempt_text=attempt,
            attempt_is_full_completion=attempt_is_full_completion,
            reasoning_text=reasoning_texts[i],
        )
        teacher_msgs = [
            {"role": "user", "content": teacher_context},
            {"role": "assistant", "content": completions[i]},
        ]
        teacher_fulls.append(tokenizer.apply_chat_template(
            teacher_msgs, tokenize=False, add_generation_prompt=False))
        teacher_prompt_text = tokenizer.apply_chat_template(
            [teacher_msgs[0]], tokenize=False, add_generation_prompt=True)
        teacher_prompt_lens.append(len(tokenizer(
            teacher_prompt_text, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

        student_msgs = [
            {"role": "user", "content": prompts[i]},
            {"role": "assistant", "content": completions[i]},
        ]
        student_fulls.append(tokenizer.apply_chat_template(
            student_msgs, tokenize=False, add_generation_prompt=False))
        student_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[i]}],
            tokenize=False, add_generation_prompt=True)
        student_prompt_lens.append(len(tokenizer(
            student_prompt_text, truncation=True, max_length=max_seq_length,
            padding=False).input_ids))

    # Step 3: Batch tokenize (right-padded)
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    student_encoding = tokenizer(
        student_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(device)
    teacher_encoding = tokenizer(
        teacher_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(teacher_device)
    tokenizer.padding_side = orig_pad_side

    # Step 4: Compute completion lengths
    completion_lens = []
    for i in range(batch_size):
        s_seq = (student_encoding.attention_mask[i] == 1).sum().item()
        t_seq = (teacher_encoding.attention_mask[i] == 1).sum().item()
        completion_lens.append(max(0, min(
            s_seq - student_prompt_lens[i],
            t_seq - teacher_prompt_lens[i])))

    max_comp_len = max(completion_lens) if any(c > 0 for c in completion_lens) else 1
    total_tokens = sum(completion_lens)
    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "loss": 0.0, "completion_tokens": 0, "kl_per_token": 0.0}

    comp_mask = torch.zeros(batch_size, max_comp_len, device=device)
    for i in range(batch_size):
        comp_mask[i, :completion_lens[i]] = 1.0

    # Step 5: Batched student forward + extraction
    student_outputs = student_model(**student_encoding)
    student_comp = _extract_completion_logits(
        student_outputs.logits, student_encoding.attention_mask,
        student_prompt_lens, completion_lens, max_comp_len)
    del student_outputs

    # Step 6: Batched teacher forward + extraction
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_comp = _extract_completion_logits(
            teacher_outputs.logits, teacher_encoding.attention_mask,
            teacher_prompt_lens, completion_lens, max_comp_len)
        del teacher_outputs
        teacher_comp = teacher_comp.to(device)

    # Step 7: Batched loss
    loss, avg_kl, total_tokens = _batched_topk_kl_loss(
        student_comp, teacher_comp, comp_mask, top_k, temperature)

    return loss, {"loss": loss.item(), "completion_tokens": total_tokens, "kl_per_token": avg_kl}


# ---------------------------------------------------------------------------
# Training loop for a single condition
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    model_name: str,
    dataset: LiveCodeBenchDataset,
    num_rollouts: int = 4,
    num_epochs: int = 3,
    eval_k: int = 4,
    eval_n: int = 16,
    eval_temperature: float = 0.8,
    eval_interval: int = 5,
    learning_rate: float = 1e-6,
    max_new_tokens: int = 4096,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Train one condition and return its convergence curve."""

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition}")
    print(f"{'=' * 60}")

    # Load fresh model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    model = model.cuda()
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create EMA teacher
    teacher = EMATeacher(model, decay=0.99)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Hparams for loss computation — sequence lengths depend on condition
    # with_thinking: teacher prompt has full completion (~4096) + question/feedback
    # reasoning_augmented: teacher response has reasoning (~1024) + completion (~4096)
    # reasoning_in_prompt: teacher prompt has question + code + feedback + reasoning
    # reasoning_in_prompt_thinking: teacher prompt has full completion + feedback + reasoning
    if condition == "with_thinking":
        # full completion + question/feedback/instruction
        max_prompt = max_new_tokens + 2048
        max_response = max_new_tokens
    elif condition == "reasoning_augmented":
        max_prompt = max_new_tokens + 2048  # question + full completion + feedback
        max_response = max_new_tokens + 1024  # reasoning + completion
    elif condition == "reasoning_in_prompt":
        max_prompt = max_new_tokens + 1024  # question + code + feedback + reasoning
        max_response = max_new_tokens
    elif condition == "reasoning_in_prompt_thinking":
        max_prompt = max_new_tokens + 3072  # full completion + feedback + reasoning
        max_response = max_new_tokens
    else:
        max_prompt = max_new_tokens  # question + extracted code + feedback
        max_response = max_new_tokens

    hparams = SDPOHparams(
        learning_rate=learning_rate,
        top_k_distillation=top_k,
        temperature=1.0,
        max_prompt_length=max_prompt,
        max_response_length=max_response,
        num_rollouts=num_rollouts,
    )

    # Convergence tracking
    curve: List[Dict[str, Any]] = []
    global_step = 0

    # Initial evaluation (step 0)
    print("  Evaluating at step 0...")
    score, sample_rate, details = evaluate_pass_at_k(
        model, tokenizer, dataset, k=eval_k, n=eval_n,
        temperature=eval_temperature, max_new_tokens=max_new_tokens)
    curve.append({
        "step": 0,
        "pass_at_k": score,
        "avg_pass_rate": sample_rate,
        "loss": None
    })
    print(
        f"  Step 0 | pass@{eval_k} (n={eval_n}) = {score:.4f} "
        f"| avg_pass_rate={sample_rate:.4f}"
    )

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")

        for prob_idx in range(len(dataset)):
            example = dataset[prob_idx]
            question = example.get(
                "question_content", example.get("question", ""))
            title = example.get("question_title", f"Problem {prob_idx}")

            # Generate rollouts
            model.eval()
            model.gradient_checkpointing_disable()
            rollouts = livecodebench_rollout(
                model, tokenizer, example,
                num_rollouts=num_rollouts, temperature=1.0,
                max_new_tokens=max_new_tokens,
            )
            model.gradient_checkpointing_enable()
            model.train()

            # Get feedback
            # Use raw question text, NOT rollout.prompt (which is already
            # chat-template-formatted and would get double-wrapped)
            prompts_list = []
            completions_list = []
            feedbacks_list = []
            student_attempts_list = []
            student_codes_list = []

            for rollout in rollouts:
                fb = get_environment_feedback(
                    prompt=rollout.prompt, completion=rollout.completion, example=example)
                prompts_list.append(question)
                completions_list.append(rollout.completion)
                feedbacks_list.append(fb.feedback_text)
                student_codes_list.append(
                    extract_python_code(rollout.completion))

            # Build student_attempts based on condition
            if condition in ("code_only", "entropy_weighted",
                             "reasoning_in_prompt"):
                student_attempts_list = student_codes_list
            elif condition in ("with_thinking", "reasoning_augmented", "reasoning_in_prompt_thinking"):
                student_attempts_list = completions_list
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Compute loss
            if condition in ("code_only", "with_thinking"):
                loss, metrics = compute_sdpo_loss_batched_ablation_style(
                    student_model=model,
                    teacher_model=teacher.model,
                    tokenizer=tokenizer,
                    prompts=prompts_list,
                    completions=completions_list,
                    feedbacks=feedbacks_list,
                    hparams=hparams,
                    student_attempts=student_attempts_list,
                    attempt_is_full_completion=(condition == "with_thinking"),
                )
            elif condition == "entropy_weighted":
                loss, metrics = compute_entropy_weighted_loss(
                    student_model=model,
                    teacher_model=teacher.model,
                    tokenizer=tokenizer,
                    prompts=prompts_list,
                    completions=completions_list,
                    feedbacks=feedbacks_list,
                    hparams=hparams,
                    student_attempts=student_attempts_list,
                )
            elif condition == "reasoning_augmented":
                loss, metrics = compute_reasoning_augmented_loss(
                    student_model=model,
                    teacher_model=teacher.model,
                    tokenizer=tokenizer,
                    prompts=prompts_list,
                    completions=completions_list,
                    feedbacks=feedbacks_list,
                    student_attempts=student_attempts_list,
                    hparams=hparams,
                )
            elif condition == "reasoning_in_prompt":
                loss, metrics = compute_reasoning_in_prompt_loss(
                    student_model=model,
                    teacher_model=teacher.model,
                    tokenizer=tokenizer,
                    prompts=prompts_list,
                    completions=completions_list,
                    feedbacks=feedbacks_list,
                    student_codes=student_codes_list,
                    hparams=hparams,
                )
            elif condition == "reasoning_in_prompt_thinking":
                loss, metrics = compute_reasoning_in_prompt_loss(
                    student_model=model,
                    teacher_model=teacher.model,
                    tokenizer=tokenizer,
                    prompts=prompts_list,
                    completions=completions_list,
                    feedbacks=feedbacks_list,
                    student_codes=student_codes_list,
                    hparams=hparams,
                    teacher_attempts=completions_list,
                    attempt_is_full_completion=True,
                )

            # Backward + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # EMA update
            teacher.update(model)

            global_step += 1
            step_loss = metrics["loss"]
            print(
                f"    Step {global_step} | {title} | loss={step_loss:.6f} "
                f"| kl={metrics.get('kl_per_token', 0):.6f} "
                f"| tokens={metrics.get('completion_tokens', 0)}"
            )

            # Periodic evaluation
            if global_step % eval_interval == 0:
                print(f"  Evaluating at step {global_step}...")
                score, sample_rate, details = evaluate_pass_at_k(
                    model, tokenizer, dataset, k=eval_k, n=eval_n,
                    temperature=eval_temperature, max_new_tokens=max_new_tokens)
                curve.append(
                    {
                        "step": global_step,
                        "pass_at_k": score,
                        "avg_pass_rate": sample_rate,
                        "loss": step_loss
                    }
                )
                print(
                    f"  Step {global_step} | pass@{eval_k} (n={eval_n}) = {score:.4f} "
                    f"| avg_pass_rate={sample_rate:.4f}"
                )

    # Final evaluation if not already done
    if not curve or curve[-1]["step"] != global_step:
        print(f"  Final evaluation at step {global_step}...")
        score, sample_rate, details = evaluate_pass_at_k(
            model, tokenizer, dataset, k=eval_k, n=eval_n,
            temperature=eval_temperature, max_new_tokens=max_new_tokens)
        curve.append(
            {
                "step": global_step,
                "pass_at_k": score,
                "avg_pass_rate": sample_rate,
                "loss": step_loss
            }
        )
        print(
            f"  Step {global_step} | pass@{eval_k} (n={eval_n}) = {score:.4f} "
            f"| avg_pass_rate={sample_rate:.4f}"
        )

    # Clean up
    del model, teacher, optimizer
    torch.cuda.empty_cache()

    return {"condition": condition, "curve": curve}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    conditions: List[str],
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 15,
    num_rollouts: int = 4,
    num_epochs: int = 3,
    eval_k: int = 4,
    eval_n: int = 16,
    eval_temperature: float = 0.8,
    eval_interval: int = 5,
    learning_rate: float = 1e-6,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    output_dir: str = "analysis/results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run convergence experiment across multiple conditions."""

    set_global_seed(seed)

    print("=" * 60)
    print("Convergence Speed Experiment")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(
        f"Problems: {num_problems} | Rollouts: {num_rollouts} | Epochs: {num_epochs}")
    print(
        f"Eval: pass@{eval_k} with n={eval_n}, temp={eval_temperature} "
        f"every {eval_interval} steps"
    )
    print(f"Seed: {seed}")
    print(f"Conditions: {conditions}")
    print("=" * 60)

    # Load dataset once (shared across conditions)
    print("Loading dataset...")
    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems\n")

    all_results = {}
    for condition in conditions:
        result = run_condition(
            condition=condition,
            model_name=model_name,
            dataset=dataset,
            num_rollouts=num_rollouts,
            num_epochs=num_epochs,
            eval_k=eval_k,
            eval_n=eval_n,
            eval_temperature=eval_temperature,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
        all_results[condition] = result

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Step':<8}", end="")
    for c in conditions:
        print(f" {c:<22}", end="")
    print()
    print("-" * (8 + 23 * len(conditions)))

    # Collect all step numbers
    all_steps = sorted(set(
        pt["step"] for r in all_results.values() for pt in r["curve"]
    ))
    for step in all_steps:
        print(f"{step:<8}", end="")
        for c in conditions:
            curve = all_results[c]["curve"]
            match = [pt for pt in curve if pt["step"] == step]
            if match:
                print(f" {match[0]['pass_at_k']:<22.4f}", end="")
            else:
                print(f" {'—':<22}", end="")
        print()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "num_epochs": num_epochs,
            "eval_k": eval_k,
            "eval_n": eval_n,
            "eval_temperature": eval_temperature,
            "eval_interval": eval_interval,
            "learning_rate": learning_rate,
            "top_k": top_k,
            "seed": seed,
            "conditions": conditions,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {c: r["curve"] for c, r in all_results.items()},
    }
    json_path = os.path.join(output_dir, "experiment_convergence.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {
            "code_only": "#2196F3",
            "with_thinking": "#FF5722",
            "entropy_weighted": "#4CAF50",
            "reasoning_augmented": "#9C27B0",
            "reasoning_in_prompt": "#FF9800",
            "reasoning_in_prompt_thinking": "#795548",
        }

        for c in conditions:
            curve = all_results[c]["curve"]
            steps = [pt["step"] for pt in curve]
            scores = [pt["pass_at_k"] for pt in curve]
            ax.plot(steps, scores, marker="o", markersize=4, label=c,
                    color=colors.get(c, None))

        ax.set_xlabel("Gradient Step")
        ax.set_ylabel(f"pass@{eval_k}")
        ax.set_title("SDPO Convergence: Teacher Prompt Ablation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "experiment_convergence.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convergence Speed Experiment")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="Which conditions to run")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--eval-k", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=16)
    parser.add_argument("--eval-temperature", type=float, default=0.8)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_experiment(
        conditions=args.conditions,
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        num_epochs=args.num_epochs,
        eval_k=args.eval_k,
        eval_n=args.eval_n,
        eval_temperature=args.eval_temperature,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        output_dir=args.output_dir,
        seed=args.seed,
    )
