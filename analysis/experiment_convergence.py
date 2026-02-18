"""
Convergence Speed Experiment

Measures how different SDPO teacher configurations affect training convergence
by tracking pass@k on private test cases every few gradient steps.

Four conditions:
  1. code_only          — teacher sees question + extracted code + feedback
  2. with_thinking      — teacher sees question + full completion + feedback
  3. entropy_weighted   — same as code_only, KL weighted by student entropy per ventile
  4. reasoning_augmented — teacher reasons about corrections first, then evaluates completion

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
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional, Tuple

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
    build_teacher_messages,
    compute_sdpo_loss_batched,
)

logger = logging.getLogger(__name__)

ALL_CONDITIONS = ["code_only", "with_thinking", "entropy_weighted", "reasoning_augmented"]


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
    k: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate pass@k using private test cases.

    Returns (mean_pass_at_k, per_problem_details).
    """
    model.eval()
    gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
    if gc_was_enabled:
        model.gradient_checkpointing_disable()

    per_problem = []

    for idx in range(len(dataset)):
        example = dataset[idx]
        title = example.get("question_title", f"Problem {idx}")

        # Generate k completions
        rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=k, temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Get private test cases
        private_tests = dataset.get_private_test_cases(idx)

        # Test each completion
        correct = 0
        for rollout in rollouts:
            code = extract_python_code(rollout.completion)
            all_passed, _ = run_test_cases(code, private_tests, timeout_seconds=10)
            if all_passed:
                correct += 1

        score = pass_at_k(len(rollouts), correct, k)
        per_problem.append({
            "idx": idx, "title": title,
            "correct": correct, "total": len(rollouts), "pass_at_k": score,
        })

    if gc_was_enabled:
        model.gradient_checkpointing_enable()
    model.train()

    mean_score = sum(p["pass_at_k"] for p in per_problem) / len(per_problem) if per_problem else 0.0
    return mean_score, per_problem


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

    Same forward pass structure as compute_sdpo_loss_batched, but:
    1. Computes student entropy from full-vocab logits before top-K reduction
    2. Bins positions into ventiles
    3. Weights per-ventile by entropy share (normalized so uniform entropy = no change)
    """
    batch_size = len(prompts)
    device = next(student_model.parameters()).device
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "completion_tokens": 0}

    teacher_device = next(teacher_model.parameters()).device
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    top_k = hparams.top_k_distillation

    if student_attempts is None:
        student_attempts = [None] * batch_size

    # Build sequences
    student_fulls, teacher_fulls = [], []
    student_prompt_lens, teacher_prompt_lens = [], []

    for i in range(batch_size):
        student_messages = build_student_messages(prompts[i], completions[i])
        student_full = tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False)
        student_fulls.append(student_full)

        student_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[i]}],
            tokenize=False, add_generation_prompt=True)
        student_prompt_lens.append(len(tokenizer(
            student_prompt_only, truncation=True, max_length=max_seq_length, padding=False).input_ids))

        teacher_messages = build_teacher_messages(
            prompts[i], completions[i], feedbacks[i], None, student_attempts[i])
        teacher_full = tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False)
        teacher_fulls.append(teacher_full)

        teacher_prompt_only = tokenizer.apply_chat_template(
            [teacher_messages[0]], tokenize=False, add_generation_prompt=True)
        teacher_prompt_lens.append(len(tokenizer(
            teacher_prompt_only, truncation=True, max_length=max_seq_length, padding=False).input_ids))

    # Tokenize
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    student_encoding = tokenizer(
        student_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(device)
    teacher_encoding = tokenizer(
        teacher_fulls, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=True).to(teacher_device)

    tokenizer.padding_side = orig_pad_side

    # Student forward
    student_outputs = student_model(**student_encoding)
    student_logits = student_outputs.logits
    del student_outputs

    # Extract per-sequence: entropy (full vocab) and top-K logits
    student_topk_list, top_k_indices_list, completion_lens = [], [], []
    entropy_per_seq = []  # list of 1-D tensors (completion_len,)

    for i in range(batch_size):
        student_seq_len = (student_encoding.attention_mask[i] == 1).sum().item()
        teacher_seq_len = (teacher_encoding.attention_mask[i] == 1).sum().item()
        s_comp_len = student_seq_len - student_prompt_lens[i]
        t_comp_len = teacher_seq_len - teacher_prompt_lens[i]
        comp_len = min(s_comp_len, t_comp_len)

        if comp_len <= 0:
            completion_lens.append(0)
            student_topk_list.append(None)
            top_k_indices_list.append(None)
            entropy_per_seq.append(None)
            continue

        completion_lens.append(comp_len)
        start = student_prompt_lens[i] - 1
        logits_comp = student_logits[i, start:start + comp_len, :]

        # Compute entropy from full vocab BEFORE reducing to top-K
        probs_full = F.softmax(logits_comp / hparams.temperature, dim=-1)
        log_probs_full = F.log_softmax(logits_comp / hparams.temperature, dim=-1)
        entropy = -(probs_full * log_probs_full).sum(dim=-1)  # (comp_len,)
        entropy_per_seq.append(entropy.detach())

        # Top-K reduction
        scaled = logits_comp / hparams.temperature
        _, topk_idx = torch.topk(scaled, top_k, dim=-1)
        student_topk_list.append(torch.gather(scaled, -1, topk_idx))
        top_k_indices_list.append(topk_idx)

    del student_logits

    # Teacher forward
    with torch.no_grad():
        teacher_outputs = teacher_model(**teacher_encoding)
        teacher_logits = teacher_outputs.logits
        del teacher_outputs

        teacher_topk_list = []
        for i in range(batch_size):
            if completion_lens[i] <= 0:
                teacher_topk_list.append(None)
                continue
            start = teacher_prompt_lens[i] - 1
            t_logits = teacher_logits[i, start:start + completion_lens[i], :].to(device)
            t_scaled = t_logits / hparams.temperature
            if top_k_indices_list[i] is not None:
                teacher_topk_list.append(torch.gather(t_scaled, -1, top_k_indices_list[i]))
            else:
                teacher_topk_list.append(t_scaled)
        del teacher_logits

    # Compute entropy-weighted loss
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_tokens = 0
    total_kl = 0.0

    for i in range(batch_size):
        if completion_lens[i] <= 0:
            continue

        s_topk = student_topk_list[i]
        t_topk = teacher_topk_list[i]

        s_probs = F.softmax(s_topk, dim=-1)
        s_log_probs = F.log_softmax(s_topk, dim=-1)
        t_log_probs = F.log_softmax(t_topk, dim=-1)
        kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)

        # Compute ventile weights from entropy
        entropy = entropy_per_seq[i]
        ventile_means = bin_into_ventiles(entropy)  # list of 20 floats
        total_entropy = sum(v for v in ventile_means if v == v)  # skip NaN
        if total_entropy > 0:
            n_tokens = completion_lens[i]
            weights = torch.ones(n_tokens, device=device)
            for v in range(20):
                v_start = int(v * n_tokens / 20)
                v_end = int((v + 1) * n_tokens / 20)
                if v_start < v_end and ventile_means[v] == ventile_means[v]:
                    w = (ventile_means[v] / total_entropy) * 20.0
                    weights[v_start:v_end] = w
            weighted_kl = kl_per_token * weights
        else:
            weighted_kl = kl_per_token

        total_loss = total_loss + weighted_kl.sum()
        total_tokens += completion_lens[i]
        total_kl += kl_per_token.sum().item()

    if total_tokens > 0:
        loss = total_loss / total_tokens
        avg_kl = total_kl / total_tokens
    else:
        loss = total_loss
        avg_kl = 0.0

    return loss, {"loss": loss.item(), "completion_tokens": total_tokens, "kl_per_token": avg_kl}


# ---------------------------------------------------------------------------
# Condition 4: Reasoning-augmented SDPO loss
# ---------------------------------------------------------------------------

def compute_reasoning_augmented_loss(
    student_model: AutoModelForCausalLM,
    teacher_model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    feedbacks: List[str],
    student_codes: List[str],
    hparams: SDPOHparams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Teacher generates reasoning about corrections, then evaluates the student's
    completion with that reasoning as additional context in the user prompt.

    Teacher: model(question + code + feedback + reasoning | completion)
    Student: model(question | completion)
    Distill on the completion tokens only.

    Reasoning is placed in the user prompt (not the assistant turn) so that
    both student and teacher go through apply_chat_template uniformly,
    ensuring identical completion tokenization on both sides.
    """
    device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    batch_size = len(prompts)
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "completion_tokens": 0}

    max_seq_length = hparams.max_prompt_length + hparams.max_response_length
    top_k = hparams.top_k_distillation

    # Process each rollout sequentially (variable-length reasoning makes batching complex)
    all_student_topk = []
    all_teacher_topk = []
    all_top_k_indices = []
    all_comp_lens = []

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for i in range(batch_size):
        # --- Step 1: Generate reasoning with teacher ---
        reasoning_prompt = (
            f"Analyze this student's attempt and explain what corrections are needed.\n\n"
            f"## Question\n{prompts[i]}\n\n"
            f"## Student Code\n```python\n{student_codes[i]}\n```\n\n"
            f"## Feedback\n{feedbacks[i]}\n\n/no_think"
        )
        reasoning_messages = [{"role": "user", "content": reasoning_prompt}]
        reasoning_text_formatted = tokenizer.apply_chat_template(
            reasoning_messages, tokenize=False, add_generation_prompt=True)
        reasoning_inputs = tokenizer(
            reasoning_text_formatted, return_tensors="pt", truncation=True,
            max_length=hparams.max_prompt_length + 512, padding=False).to(teacher_device)

        with torch.no_grad():
            reasoning_output = teacher_model.generate(
                **reasoning_inputs,
                max_new_tokens=512,
                do_sample=True, temperature=0.7, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        reasoning_ids = reasoning_output[0, reasoning_inputs.input_ids.shape[1]:]
        reasoning_text = tokenizer.decode(reasoning_ids, skip_special_tokens=True)
        del reasoning_output, reasoning_inputs

        # --- Step 2: Build teacher messages with reasoning in user prompt ---
        teacher_context = (
            f"## Question\n{prompts[i]}\n\n"
            f"## Previous Attempt\n```python\n{student_codes[i]}\n```\n\n"
            f"## Feedback (from environment) for the previous attempt\n{feedbacks[i]}\n\n"
            f"## Analysis of Previous Attempt and Guidance for Improvement\n{reasoning_text}\n\n"
            f"Use the context above to inform your approach, but treat your attempt "
            f"as an attempt from scratch. Do not reference the previous attempt in "
            f"your solution, just use it and its feedback as guidance. Write a correct "
            f"solution to the question. Put your code in a ```python{{code}}``` block."
        )
        teacher_msgs = [
            {"role": "user", "content": teacher_context},
            {"role": "assistant", "content": completions[i]},
        ]
        teacher_full_text = tokenizer.apply_chat_template(
            teacher_msgs, tokenize=False, add_generation_prompt=False)
        teacher_prompt_text = tokenizer.apply_chat_template(
            [teacher_msgs[0]], tokenize=False, add_generation_prompt=True)

        # --- Step 3: Build student messages (standard) ---
        student_msgs = [
            {"role": "user", "content": prompts[i]},
            {"role": "assistant", "content": completions[i]},
        ]
        student_full_text = tokenizer.apply_chat_template(
            student_msgs, tokenize=False, add_generation_prompt=False)
        student_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[i]}],
            tokenize=False, add_generation_prompt=True)

        # Tokenize
        student_enc = tokenizer(
            student_full_text, return_tensors="pt", truncation=True,
            max_length=max_seq_length, padding=False).to(device)
        teacher_enc = tokenizer(
            teacher_full_text, return_tensors="pt", truncation=True,
            max_length=max_seq_length, padding=False).to(teacher_device)

        student_prompt_len = len(tokenizer(
            student_prompt_text, truncation=True, max_length=max_seq_length,
            padding=False).input_ids)
        teacher_prompt_len = len(tokenizer(
            teacher_prompt_text, truncation=True, max_length=max_seq_length,
            padding=False).input_ids)

        # Completion lengths (subtract 1 for trailing <|im_end|> in template output)
        s_seq_len = student_enc.input_ids.shape[1]
        t_seq_len = teacher_enc.input_ids.shape[1]
        s_comp_len = s_seq_len - student_prompt_len
        t_comp_len = t_seq_len - teacher_prompt_len
        comp_len = min(s_comp_len, t_comp_len)

        if comp_len <= 0:
            logger.warning(
                f"  Rollout {i}: comp_len={comp_len} (s_seq={s_seq_len}, "
                f"s_prompt={student_prompt_len}, t_seq={t_seq_len}, "
                f"t_prompt={teacher_prompt_len}, reasoning_len={len(reasoning_text)})"
            )
            all_comp_lens.append(0)
            all_student_topk.append(None)
            all_teacher_topk.append(None)
            all_top_k_indices.append(None)
            continue

        all_comp_lens.append(comp_len)

        # --- Step 4: Forward passes and top-K extraction ---
        student_out = student_model(**student_enc)
        s_logits = student_out.logits[0, student_prompt_len - 1:student_prompt_len - 1 + comp_len, :]
        s_scaled = s_logits / hparams.temperature
        del student_out

        _, topk_idx = torch.topk(s_scaled, top_k, dim=-1)
        s_topk = torch.gather(s_scaled, -1, topk_idx)
        all_student_topk.append(s_topk)
        all_top_k_indices.append(topk_idx)
        del s_logits, s_scaled

        with torch.no_grad():
            teacher_out = teacher_model(**teacher_enc)
            t_logits = teacher_out.logits[0, teacher_prompt_len - 1:teacher_prompt_len - 1 + comp_len, :]
            t_logits = t_logits.to(device)
            t_scaled = t_logits / hparams.temperature
            t_topk = torch.gather(t_scaled, -1, topk_idx)
            all_teacher_topk.append(t_topk)
            del teacher_out, t_logits, t_scaled

        del student_enc, teacher_enc

    tokenizer.padding_side = orig_pad_side

    # Compute loss
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_tokens = 0
    total_kl = 0.0

    for i in range(batch_size):
        if all_comp_lens[i] <= 0:
            continue

        s_probs = F.softmax(all_student_topk[i], dim=-1)
        s_log_probs = F.log_softmax(all_student_topk[i], dim=-1)
        t_log_probs = F.log_softmax(all_teacher_topk[i], dim=-1)
        kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)

        total_loss = total_loss + kl_per_token.sum()
        total_tokens += all_comp_lens[i]
        total_kl += kl_per_token.sum().item()

    if total_tokens > 0:
        loss = total_loss / total_tokens
        avg_kl = total_kl / total_tokens
    else:
        loss = total_loss
        avg_kl = 0.0

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
    eval_k: int = 8,
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
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    model = model.cuda()
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create EMA teacher
    teacher = EMATeacher(model, decay=0.99)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Hparams for loss computation — sequence lengths depend on condition
    # with_thinking: teacher prompt contains the full completion (~4096 tokens)
    #   plus question + feedback, so prompt can be ~6000 tokens
    # reasoning_augmented: teacher prompt contains question + code + feedback +
    #   reasoning (~512 tokens), so prompt needs ~1024 extra
    if condition == "with_thinking":
        max_prompt = max_new_tokens + 2048  # full completion + question/feedback/instruction
        max_response = max_new_tokens
    elif condition == "reasoning_augmented":
        max_prompt = max_new_tokens + 1024  # question + code + feedback + reasoning
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
    score, details = evaluate_pass_at_k(
        model, tokenizer, dataset, k=eval_k, max_new_tokens=max_new_tokens)
    curve.append({"step": 0, "pass_at_k": score, "loss": None})
    print(f"  Step 0 | pass@{eval_k} = {score:.4f}")

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")

        for prob_idx in range(len(dataset)):
            example = dataset[prob_idx]
            question = example.get("question_content", example.get("question", ""))
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
                student_codes_list.append(extract_python_code(rollout.completion))

            # Build student_attempts based on condition
            if condition == "code_only" or condition == "entropy_weighted":
                student_attempts_list = student_codes_list
            elif condition == "with_thinking":
                student_attempts_list = completions_list
            elif condition == "reasoning_augmented":
                student_attempts_list = student_codes_list  # used for teacher context
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Compute loss
            if condition in ("code_only", "with_thinking"):
                loss, metrics = compute_sdpo_loss_batched(
                    student_model=model,
                    teacher_model=teacher.model,
                    tokenizer=tokenizer,
                    prompts=prompts_list,
                    completions=completions_list,
                    feedbacks=feedbacks_list,
                    prior_solutions=[None] * len(prompts_list),
                    hparams=hparams,
                    student_attempts=student_attempts_list,
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
                    student_codes=student_codes_list,
                    hparams=hparams,
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
                score, details = evaluate_pass_at_k(
                    model, tokenizer, dataset, k=eval_k, max_new_tokens=max_new_tokens)
                curve.append({"step": global_step, "pass_at_k": score, "loss": step_loss})
                print(f"  Step {global_step} | pass@{eval_k} = {score:.4f}")

    # Final evaluation if not already done
    if not curve or curve[-1]["step"] != global_step:
        print(f"  Final evaluation at step {global_step}...")
        score, details = evaluate_pass_at_k(
            model, tokenizer, dataset, k=eval_k, max_new_tokens=max_new_tokens)
        curve.append({"step": global_step, "pass_at_k": score, "loss": step_loss})
        print(f"  Step {global_step} | pass@{eval_k} = {score:.4f}")

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
    eval_k: int = 8,
    eval_interval: int = 5,
    learning_rate: float = 1e-6,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run convergence experiment across multiple conditions."""

    print("=" * 60)
    print("Convergence Speed Experiment")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts: {num_rollouts} | Epochs: {num_epochs}")
    print(f"Eval: pass@{eval_k} every {eval_interval} steps")
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
            "eval_interval": eval_interval,
            "learning_rate": learning_rate,
            "top_k": top_k,
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
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_experiment(
        conditions=args.conditions,
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        num_epochs=args.num_epochs,
        eval_k=args.eval_k,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
