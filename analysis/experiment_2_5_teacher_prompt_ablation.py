import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.experiment_1_reward_on_regen import compute_reward
from analysis.utils import (
    bin_into_ventiles,
    compute_full_kl_per_position,
    compute_topk_kl_per_position,
    get_standard_completion_logits_completion_ids_and_mask
)
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout

logger = logging.getLogger(__name__)

ATTEMPT_MODES = [
    "code_only", "no_attempt", "with_thinking",
    "reasoning_augmented", "reasoning_augmented_thinking",
    "reasoning_in_response_thinking",
]


def build_teacher_messages_ablation(
    prompt: str,
    completion: str,
    feedback: str,
    attempt_mode: str,
    student_code: Optional[str] = None,
    full_completion: Optional[str] = None,
    reasoning_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build teacher messages with different prior-attempt configurations.

    attempt_mode:
      - "code_only": includes extracted python code as Previous Attempt
      - "no_attempt": omits the Previous Attempt section entirely
      - "with_thinking": includes the full completion (thinking + code)
      - "reasoning_augmented": code_only + model-generated reasoning in user prompt
      - "reasoning_augmented_thinking": with_thinking + model-generated reasoning
      - "reasoning_in_response_thinking": handled separately (reasoning in assistant turn)
    """
    parts = [f"## Question\n{prompt}"]

    if attempt_mode in ("code_only", "reasoning_augmented") and student_code is not None:
        parts.append(
            f"## Previous Attempt\n```python\n{student_code}\n```"
        )
    elif attempt_mode in ("with_thinking", "reasoning_augmented_thinking") and full_completion is not None:
        parts.append(
            f"## Previous Attempt (including reasoning)\n{full_completion}"
        )
    # no_attempt: skip the Previous Attempt section

    parts.append(
        f"## Feedback (from environment) for the previous attempt\n{feedback}"
    )

    if attempt_mode.startswith("reasoning_augmented") and reasoning_text is not None:
        parts.append(
            f"## Analysis of Previous Attempt and Guidance for Improvement\n{reasoning_text}"
        )

    parts.append(
        "Correctly solve the original question."
    )

    if attempt_mode == "reasoning_in_response_thinking":
        completion = f"{reasoning_text}\n\n{completion}"

    return [
        {"role": "user", "content": "\n\n".join(parts)},
        {"role": "assistant", "content": completion},
    ]


def _generate_reasoning(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    student_code: str,
    feedback_text: str,
    max_seq_length: int = 10240,
) -> str:
    """Generate reasoning about corrections using /no_think."""
    reasoning_prompt = (
        f"## Question\n{question}\n\n"
        f"## Student Code\n```python\n{student_code}\n```\n\n"
        f"## Feedback\n{feedback_text}\n\n"
        f"Analyze the student's attempt based on the feedback. If the student was correct simply say so. If the student's code was incorrect identify where the student went wrong, and how they can fix it.\n\n/no_think"
    )
    reasoning_messages = [{"role": "user", "content": reasoning_prompt}]
    reasoning_input_text = tokenizer.apply_chat_template(
        reasoning_messages, tokenize=False, add_generation_prompt=True)
    reasoning_inputs = tokenizer(
        reasoning_input_text, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False).to(model.device)

    with torch.no_grad():
        reasoning_output = model.generate(
            **reasoning_inputs,
            max_new_tokens=1024,
            do_sample=True, temperature=0.7, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reasoning_ids = reasoning_output[0, reasoning_inputs.input_ids.shape[1]:]
    reasoning_text = tokenizer.decode(reasoning_ids, skip_special_tokens=True)
    del reasoning_output, reasoning_inputs
    return reasoning_text.replace("<think>", "").replace("</think>", "").strip()


def _compute_reasoning_in_response_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    feedback_text: str,
    full_completion: str,
    reasoning_text: str,
    max_seq_length: int = 10240,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute student and teacher logits with reasoning in the assistant turn.

    The teacher sees question + full completion + feedback in the user prompt,
    then reasoning + completion in the assistant turn. Both sides bypass
    apply_chat_template for assistant content (Qwen3's template would mangle
    the combined reasoning + completion which contains multiple </think> tags).

    Returns:
        (student_logits, teacher_logits, completion_token_ids)
    """
    # Build student sequence (bypass template for assistant content)
    student_user_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    student_full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion}
        ],
        tokenize=False, add_generation_prompt=False)

    # Build teacher sequence (bypass template for assistant content)
    teacher_context = (
        f"## Question\n{question}\n\n"
        f"## Previous Attempt (including reasoning)\n{full_completion}\n\n"
        f"## Feedback (from environment) for the previous attempt\n{feedback_text}\n\n"
        "Correctly solve the original question."
    )
    teacher_user_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": teacher_context}],
        tokenize=False, add_generation_prompt=True)

    teacher_full_text = teacher_user_text + \
        f"\n{reasoning_text}\n\n{completion}\n<im_end>"
    teacher_prefix_text = teacher_user_text + f"\n{reasoning_text}"

    # Tokenize
    student_enc = tokenizer(
        student_full_text, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False).to(model.device)
    student_prompt_len = len(tokenizer(
        student_user_text, truncation=True, max_length=max_seq_length,
        padding=False).input_ids)

    teacher_enc = tokenizer(
        teacher_full_text, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False).to(model.device)
    teacher_prefix_len = len(tokenizer(
        teacher_prefix_text, truncation=True, max_length=max_seq_length,
        padding=False).input_ids)

    # Completion token IDs (from student side, raw tokenization)
    s_seq_len = student_enc.input_ids.shape[1]
    comp_ids = student_enc.input_ids[0, student_prompt_len:s_seq_len]

    # Forward passes and extract completion logits
    with torch.no_grad():
        student_out = model(**student_enc)
        student_logits = student_out.logits[0,
                                            student_prompt_len - 1: s_seq_len - 1, :]
        del student_out

        teacher_out = model(**teacher_enc)
        t_seq_len = teacher_enc.input_ids.shape[1]
        teacher_logits = teacher_out.logits[0,
                                            teacher_prefix_len - 1: t_seq_len - 1, :]
        del teacher_out

    del student_enc, teacher_enc

    return student_logits, teacher_logits, comp_ids


def _pad_seq_dim(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    pad_t = target_len - x.shape[1]

    if pad_t <= 0:
        return x

    if x.dim() == 3:
        return F.pad(x, (0, 0, 0, pad_t), value=pad_value)
    
    if x.dim() == 2:
        return F.pad(x, (0, pad_t), value=pad_value)
    
    raise ValueError(f"Unsupported dimension: {x.dim()}")


def get_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    comp_ids: torch.Tensor,
    mask: torch.Tensor,
    k: int = 20,
    disagreement_threshold: float = -1.0
) -> Dict[str, Any]:
    """
    Compute metrics for a given student logits, teacher logits, and completion token IDs. We then accumulate these metrics 
    over ventiles

    We are calculating the following metrics:
    - KL(student || teacher) over the student's top-k tokens per position, averaged per ventile
    - Disagreement rate, i.e. the fraction of tokens where the teacher assigns lower probability than the student to the actual completion token, averaged per ventile
    - Student entropy, i.e. the entropy of the student's logits per position, averaged per ventile
    - Teacher entropy, i.e. the entropy of the teacher's logits per position, averaged per ventile
    """
    target_len = max(student_logits.shape[1], teacher_logits.shape[1], comp_ids.shape[1], mask.shape[1])
    student_logits = _pad_seq_dim(student_logits, target_len)
    teacher_logits = _pad_seq_dim(teacher_logits, target_len)
    comp_ids = _pad_seq_dim(comp_ids, target_len)
    mask = _pad_seq_dim(mask, target_len, pad_value=False)

    student_top_k_logits, student_top_k_indices = torch.topk(
        student_logits, k, dim=-1)
    teacher_logits_at_top_k_indices = torch.gather(
        teacher_logits, -1, student_top_k_indices)

    s_probs = F.softmax(student_top_k_logits, dim=-1)
    s_log_probs = F.log_softmax(student_top_k_logits, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits_at_top_k_indices, dim=-1)

    per_token_kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)

    s_log_probs = F.log_softmax(student_logits, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits, dim=-1)
    token_ids = comp_ids.to(torch.int64)
    t_selected_log_probs = torch.gather(
        t_log_probs, -1, token_ids.unsqueeze(-1))
    s_selected_log_probs = torch.gather(
        s_log_probs, -1, token_ids.unsqueeze(-1))
    logp_diff = (t_selected_log_probs - s_selected_log_probs).squeeze(-1)
    disagreement = (logp_diff < disagreement_threshold).float()

    student_entropy = -(F.softmax(student_top_k_logits, dim=-1)
                        * F.log_softmax(student_top_k_logits, dim=-1)).sum(dim=-1)
    teacher_entropy = -(F.softmax(teacher_logits_at_top_k_indices, dim=-1)
                        * F.log_softmax(teacher_logits_at_top_k_indices, dim=-1)).sum(dim=-1)

    per_token_kl = per_token_kl * mask
    disagreement = disagreement * mask
    student_entropy = student_entropy * mask
    teacher_entropy = teacher_entropy * mask

    return {
        "kl_ventiles": bin_into_ventiles(per_token_kl.detach().cpu(), mask.detach().cpu()),
        "disagreement_ventiles": bin_into_ventiles(disagreement.detach().cpu(), mask.detach().cpu()),
        "student_entropy_ventiles": bin_into_ventiles(student_entropy.detach().cpu(), mask.detach().cpu()),
        "teacher_entropy_ventiles": bin_into_ventiles(teacher_entropy.detach().cpu(), mask.detach().cpu()),
        "seq_len": mask.sum(dim=-1).to(torch.int64),
    }


def compute_ablation_kls(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    feedback_text: str,
    student_code: str,
    full_completion: str,
    top_k: int = 20,
    temperature: float = 1.0,
    full_dist: bool = False,
    max_seq_length: int = 10240,
) -> Dict[str, Any]:
    """
    Compute KL(student || teacher) under all six teacher prompt conditions.

    The student side is always model(completion | question) — identical across
    conditions. The teacher side varies by what context precedes the completion.

    Most conditions go through apply_chat_template uniformly. The exception is
    reasoning_in_response_thinking, which places reasoning in the assistant turn
    and must bypass apply_chat_template (Qwen3's </think> handling would mangle
    the combined reasoning + completion). For that condition, both student and
    teacher bypass the template to keep completion tokens aligned.

    Returns dict mapping attempt_mode -> {kl, ventiles, seq_len}.
    """
    student_msgs = [{"role": "user", "content": question}]
    student_logits, prompt_len = get_completion_logits(
        model, tokenizer, student_msgs, completion,
    )

    # Get completion token IDs (for disagreement computation)
    # These come from the template-processed student sequence.
    student_full_msgs = student_msgs + \
        [{"role": "assistant", "content": completion}]
    student_full_text = tokenizer.apply_chat_template(
        student_full_msgs, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(
        student_full_text, truncation=True, max_length=max_seq_length, padding=False,
    ).input_ids
    template_comp_ids = torch.tensor(
        full_ids[prompt_len:], device=student_logits.device)

    kl_fn = compute_full_kl_per_position if full_dist else compute_topk_kl_per_position

    # Generate reasoning once, reused for all reasoning conditions
    reasoning_text = None
    if any("reasoning" in m for m in ATTEMPT_MODES):
        reasoning_text = _generate_reasoning(
            model, tokenizer, question, student_code, feedback_text, max_seq_length,
        )

    results = {}
    for mode in ATTEMPT_MODES:
        if mode == "reasoning_in_response_thinking":
            # Reasoning goes in assistant turn — must bypass apply_chat_template
            # for both sides to keep completion tokens aligned.
            s_logits, teacher_logits, raw_comp_ids = _compute_reasoning_in_response_logits(
                model, tokenizer, question, completion, feedback_text,
                full_completion, reasoning_text, max_seq_length,
            )
            comp_ids = raw_comp_ids
        else:
            teacher_full = build_teacher_messages_ablation(
                prompt=question, completion=completion,
                feedback=feedback_text, attempt_mode=mode,
                student_code=student_code, full_completion=full_completion,
                reasoning_text=reasoning_text,
            )
            teacher_msgs = [teacher_full[0]]  # just the user turn

            teacher_logits, _ = get_completion_logits(
                model, tokenizer, teacher_msgs, completion,
            )
            s_logits = student_logits
            comp_ids = template_comp_ids

        if full_dist:
            kl = kl_fn(s_logits, teacher_logits, temperature)
        else:
            kl = kl_fn(s_logits, teacher_logits, top_k, temperature)

        # Compute disagreement rate: fraction of tokens where teacher assigns
        # lower probability than the student to the actual completion token.
        min_len = len(kl)
        if min_len > 0:
            token_len = min(min_len, len(comp_ids))
            s_log_probs = F.log_softmax(s_logits[:token_len], dim=-1)
            t_log_probs = F.log_softmax(teacher_logits[:token_len], dim=-1)
            idx = torch.arange(token_len, device=s_logits.device)
            token_ids = comp_ids[:token_len]
            logp_diff = t_log_probs[idx, token_ids] - \
                s_log_probs[idx, token_ids]
            disagreement = (logp_diff < 0).float()
            dis_ventiles = bin_into_ventiles(disagreement.detach().cpu())
            mean_dis = disagreement.mean().item()
        else:
            dis_ventiles = [float("nan")] * 20
            mean_dis = float("nan")

        results[mode] = {
            "kl": kl.detach().cpu(),
            "ventiles": bin_into_ventiles(kl.detach().cpu()),
            "disagreement_ventiles": dis_ventiles,
            "seq_len": min_len,
            "mean_kl": kl.mean().item() if min_len > 0 else float("nan"),
            "mean_disagreement": mean_dis,
        }

    return results


def aggregate_ventiles(
    ventile_lists: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """Compute mean and stderr for a list of ventile vectors."""
    n = len(ventile_lists)
    if n == 0:
        return [float("nan")] * 20, [float("nan")] * 20

    means = [sum(d[i] for d in ventile_lists) / n for i in range(20)]
    stderrs = []
    for i in range(20):
        vals = [d[i] for d in ventile_lists]
        var = sum((v - means[i]) ** 2 for v in vals) / \
            (n - 1) if n > 1 else 0.0
        stderrs.append(math.sqrt(var / n) if n > 0 else 0.0)
    return means, stderrs


def print_comparison_table(
    title: str,
    values: List[float],
    count: int,
    stderrs: Optional[List[float]] = None,
) -> None:
    """Print a formatted comparison table for a single metric series."""
    print(f"\n{title} (n={count})")
    if stderrs is None:
        if len(values) == 0:
            print("No values available.")
            return
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / \
            (len(values) - 1) if len(values) > 1 else 0.0
        stderr = math.sqrt(var / len(values))
        print(f"{'Mean':<10} {'StdErr':<10}")
        print("-" * 22)
        print(f"{mean:<10.4f} {stderr:<10.4f}")
        return

    if len(values) != len(stderrs):
        raise ValueError(
            f"values/stderrs length mismatch: {len(values)} vs {len(stderrs)}")

    print(f"{'Ventile':<10} {'Mean±StdErr':<18}")
    print("-" * 30)
    for d, (mean, stderr) in enumerate(zip(values, stderrs)):
        pct = f"{d * 5}-{(d + 1) * 5}%"
        print(f"{pct:<10} {mean:>7.4f}±{stderr:<8.4f}")


def run_experiment_2_5(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 8192,
    top_k: int = 20,
    full_dist: bool = False,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 2.5: Teacher Prompt Ablation."""

    kl_mode = "full-dist" if full_dist else f"top-{top_k}"
    print("=" * 70)
    print("Experiment 2.5: Teacher Prompt Ablation — KL Distribution")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(
        f"Problems: {num_problems} | Rollouts: {num_rollouts} | KL mode: {kl_mode}")
    print(f"Conditions: {', '.join(ATTEMPT_MODES)}")
    print("=" * 70)
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems\n")

    # Collect per-rollout records
    problem_results: List[Dict[str, Any]] = []

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        question = example.get("question_content", example.get("question", ""))
        print(f"Problem {prob_idx} ({title})...")

        # Generate original rollouts
        print(f"  Generating {num_rollouts} rollouts...")
        rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Get feedback for each rollout
        feedbacks = []
        rewards = []
        for rollout in rollouts:
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            feedbacks.append(fb)
            rewards.append(compute_reward(fb))

            # if rewards[-1] == 0.0:
            #     print(rollout.completion)
            #     raise ValueError("Incorrect rollout")

        student_user_messages = [
            f"Answer the following question, please keep your reasoning concise, and put your code in a ```python{{code}}``` block:\n\n{question}"] * num_rollouts
        student_assistant_messages = [
            rollout.completion for rollout in rollouts]

        teacher_user_messages = [
            f"## Question\n{question}\n\n## Previous Attempt\n{rollout.completion}\n\n## Feedback (from environment) for the previous attempt\n{fb.feedback_text}\nCorrectly solve the original question." for rollout, fb in zip(rollouts, feedbacks)]
        teacher_assistant_messages = [
            rollout.completion for rollout in rollouts]

        student_logits, student_completion_ids, student_mask = get_standard_completion_logits_completion_ids_and_mask(
            model, tokenizer, student_user_messages, student_assistant_messages,
        )
        teacher_logits, _, teacher_mask = get_standard_completion_logits_completion_ids_and_mask(
            model, tokenizer, teacher_user_messages, teacher_assistant_messages,
        )

        metrics = get_metrics(
            student_logits, teacher_logits, student_completion_ids, student_mask
        )

        problem_results.extend([{
            "problem_idx": prob_idx,
            "rollout_idx": r_idx,
            "correct": rewards[r_idx] == 1.0,
            "kl_ventiles": metrics["kl_ventiles"][r_idx],
            "disagreement_ventiles": metrics["disagreement_ventiles"][r_idx],
            "student_entropy_ventiles": metrics["student_entropy_ventiles"][r_idx],
            "teacher_entropy_ventiles": metrics["teacher_entropy_ventiles"][r_idx],
            "seq_len": metrics["seq_len"][r_idx],
        } for r_idx in range(num_rollouts)])

    # ------------------------------------------------------------------
    # Aggregate by correctness strata
    # ------------------------------------------------------------------
    strata = {
        "all": problem_results,
        "incorrect": [r for r in problem_results if not r["correct"]],
        "correct": [r for r in problem_results if r["correct"]],
    }

    summary: Dict[str, Any] = {}
    print("=" * 70)
    for stratum_name, records in strata.items():
        if not records:
            summary[stratum_name] = {"count": 0}
            continue

        kl_means, kl_stderrs = aggregate_ventiles(
            [r["kl_ventiles"] for r in records])
        disagreement_means, disagreement_stderrs = aggregate_ventiles(
            [r["disagreement_ventiles"] for r in records])
        student_entropy_means, student_entropy_stderrs = aggregate_ventiles(
            [r["student_entropy_ventiles"] for r in records])
        teacher_entropy_means, teacher_entropy_stderrs = aggregate_ventiles(
            [r["teacher_entropy_ventiles"] for r in records])
        seq_lens = [r["seq_len"] for r in records]

        summary[stratum_name] = {
            "count": len(records),
            "kl_means": kl_means,
            "kl_stderrs": kl_stderrs,
            "disagreement_means": disagreement_means,
            "disagreement_stderrs": disagreement_stderrs,
            "student_entropy_means": student_entropy_means,
            "student_entropy_stderrs": student_entropy_stderrs,
            "teacher_entropy_means": teacher_entropy_means,
            "teacher_entropy_stderrs": teacher_entropy_stderrs,
            "seq_lens": seq_lens,
        }

        print_comparison_table(
            f"KL — {stratum_name}",
            kl_means, len(records), kl_stderrs,
        )
        print_comparison_table(
            f"Disagreement Rate — {stratum_name}",
            disagreement_means, len(records), disagreement_stderrs,
        )
        print_comparison_table(
            f"Student Entropy — {stratum_name}",
            student_entropy_means, len(records), student_entropy_stderrs,
        )
        print_comparison_table(
            f"Teacher Entropy — {stratum_name}",
            teacher_entropy_means, len(records), teacher_entropy_stderrs,
        )
        print_comparison_table(
            f"Sequence Length — {stratum_name}",
            seq_lens, len(records),
        )

    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2.5: Teacher Prompt Ablation — KL Distribution"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--full-dist", action="store_true",
                        help="Use full vocabulary KL instead of top-K")
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_experiment_2_5(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        full_dist=args.full_dist,
        output_dir=args.output_dir,
    )
