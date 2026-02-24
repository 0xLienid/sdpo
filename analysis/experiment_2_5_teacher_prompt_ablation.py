"""
Experiment 2.5: Teacher Prompt Ablation — KL Distribution

Compares KL(student || teacher) across sequence positions under six teacher
prompt configurations:

  1. code_only   — teacher sees question + extracted code + feedback (current SDPO)
  2. no_attempt  — teacher sees question + feedback only (no prior attempt)
  3. with_thinking — teacher sees question + full completion (thinking + code) + feedback
  4. reasoning_augmented — teacher sees question + extracted code + feedback +
     model-generated reasoning about corrections (via /no_think) in the user prompt
  5. reasoning_augmented_thinking — same as 4, but with full completion instead of
     extracted code as the previous attempt
  6. reasoning_in_response_thinking — like 5, but the reasoning is placed in the
     assistant turn (prepended to the completion) instead of the user prompt

Tests whether including the student's prior attempt in the teacher prompt
contributes to the positional KL degradation ("prefix corruption").
"""

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
    get_completion_logits,
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

    teacher_full_text = teacher_user_text + f"\n{reasoning_text}\n\n{completion}\n<im_end>"
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
    mode_means: Dict[str, List[float]],
    mode_stderrs: Dict[str, List[float]],
    count: int,
) -> None:
    """Print a formatted comparison table across attempt modes."""
    print(f"\n{title} (n={count})")
    header = f"{'Ventile':<10}"
    for mode in ATTEMPT_MODES:
        header += f" {mode:<18}"
    print(header)
    print("-" * (10 + 19 * len(ATTEMPT_MODES)))
    for d in range(20):
        pct = f"{d * 5}-{(d + 1) * 5}%"
        row = f"{pct:<10}"
        for mode in ATTEMPT_MODES:
            m = mode_means[mode][d]
            se = mode_stderrs[mode][d]
            row += f" {m:>7.4f}±{se:<8.4f}"
        print(row)


def run_experiment_2_5(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
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
    rollout_records: List[Dict[str, Any]] = []
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

        # Compute ablation KLs for each rollout
        prob_rollout_records = []
        for r_idx, (rollout, fb) in enumerate(zip(rollouts, feedbacks)):
            student_code = extract_python_code(rollout.completion)
            correct = rewards[r_idx] == 1.0

            print(
                f"  Rollout {r_idx}: reward={rewards[r_idx]:.2f} "
                f"{'correct' if correct else 'incorrect'} — computing KL ({len(ATTEMPT_MODES)} conditions)..."
            )

            kl_results = compute_ablation_kls(
                model, tokenizer,
                question=question,
                completion=rollout.completion,
                feedback_text=fb.feedback_text,
                student_code=student_code,
                full_completion=rollout.completion,
                top_k=top_k,
                full_dist=full_dist,
            )

            # Log mean KLs and disagreement
            kl_summary = " | ".join(
                f"{mode}: KL={kl_results[mode]['mean_kl']:.4f} "
                f"dis={kl_results[mode]['mean_disagreement']:.1%}"
                for mode in ATTEMPT_MODES
            )
            print(f"    {kl_summary}")

            record = {
                "problem_idx": prob_idx,
                "rollout_idx": r_idx,
                "reward": rewards[r_idx],
                "correct": correct,
                "ventiles": {
                    mode: kl_results[mode]["ventiles"]
                    for mode in ATTEMPT_MODES
                },
                "disagreement_ventiles": {
                    mode: kl_results[mode]["disagreement_ventiles"]
                    for mode in ATTEMPT_MODES
                },
                "mean_kl": {
                    mode: kl_results[mode]["mean_kl"]
                    for mode in ATTEMPT_MODES
                },
                "mean_disagreement": {
                    mode: kl_results[mode]["mean_disagreement"]
                    for mode in ATTEMPT_MODES
                },
                "seq_len": kl_results["code_only"]["seq_len"],
            }
            rollout_records.append(record)
            prob_rollout_records.append(record)

        problem_results.append({
            "problem_idx": prob_idx,
            "question_title": title,
            "mean_reward": sum(rewards) / len(rewards),
            "rollouts": prob_rollout_records,
        })
        print()

    # ------------------------------------------------------------------
    # Aggregate by correctness strata
    # ------------------------------------------------------------------
    strata = {
        "all": rollout_records,
        "incorrect": [r for r in rollout_records if not r["correct"]],
        "correct": [r for r in rollout_records if r["correct"]],
    }

    summary: Dict[str, Any] = {}
    print("=" * 70)
    for stratum_name, records in strata.items():
        if not records:
            summary[stratum_name] = {"count": 0}
            continue

        mode_means = {}
        mode_stderrs = {}
        dis_mode_means = {}
        dis_mode_stderrs = {}
        for mode in ATTEMPT_MODES:
            ventile_lists = [r["ventiles"][mode] for r in records]
            means, stderrs = aggregate_ventiles(ventile_lists)
            mode_means[mode] = means
            mode_stderrs[mode] = stderrs

            dis_lists = [r["disagreement_ventiles"][mode] for r in records]
            d_means, d_stderrs = aggregate_ventiles(dis_lists)
            dis_mode_means[mode] = d_means
            dis_mode_stderrs[mode] = d_stderrs

        summary[stratum_name] = {
            "count": len(records),
            "per_mode": {
                mode: {
                    "mean_kl_per_ventile": mode_means[mode],
                    "stderr_kl_per_ventile": mode_stderrs[mode],
                    "grand_mean_kl": sum(
                        r["mean_kl"][mode] for r in records
                    ) / len(records),
                    "mean_disagreement_per_ventile": dis_mode_means[mode],
                    "stderr_disagreement_per_ventile": dis_mode_stderrs[mode],
                    "grand_mean_disagreement": sum(
                        r["mean_disagreement"][mode] for r in records
                    ) / len(records),
                }
                for mode in ATTEMPT_MODES
            },
        }

        print_comparison_table(
            f"KL — {stratum_name}",
            mode_means, mode_stderrs, len(records),
        )
        print_comparison_table(
            f"Disagreement Rate — {stratum_name}",
            dis_mode_means, dis_mode_stderrs, len(records),
        )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Print slope analysis: compare first-half vs second-half
    # ------------------------------------------------------------------
    for metric, key in [("KL", "mean_kl_per_ventile"),
                        ("Disagreement", "mean_disagreement_per_ventile")]:
        print(
            f"\nSlope Analysis — {metric} (second-half mean - first-half mean):")
        print(
            f"{'Stratum':<15} {'Mode':<30} {'First Half':<12} {'Second Half':<12} {'Slope':<10}")
        print("-" * 79)
        for stratum_name, data in summary.items():
            if data.get("count", 0) == 0:
                continue
            for mode in ATTEMPT_MODES:
                ventiles = data["per_mode"][mode][key]
                first_half = sum(ventiles[:10]) / 10
                second_half = sum(ventiles[10:]) / 10
                slope = second_half - first_half
                fmt = ".4f" if metric == "KL" else ".1%"
                print(
                    f"{stratum_name:<15} {mode:<30} "
                    f"{first_half:<12{fmt}} {second_half:<12{fmt}} "
                    f"{slope:<10{fmt}}"
                )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    suffix = "_full_dist" if full_dist else ""
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "full_dist": full_dist,
            "timestamp": datetime.now().isoformat(),
        },
        "problems": problem_results,
        "summary": summary,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"experiment_2_5{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ------------------------------------------------------------------
    # Plot: one row per stratum, one column per attempt mode
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = {
            k: v for k, v in summary.items() if v.get("count", 0) > 0
        }
        num_rows = len(plot_strata)
        fig, axes = plt.subplots(
            num_rows, len(ATTEMPT_MODES),
            figsize=(7 * len(ATTEMPT_MODES), 5 * num_rows),
            squeeze=False,
        )
        ventile_labels = [f"{d * 5}-{(d + 1) * 5}%" for d in range(20)]
        x = range(20)

        for row, (stratum_name, data) in enumerate(plot_strata.items()):
            for col, mode in enumerate(ATTEMPT_MODES):
                means = data["per_mode"][mode]["mean_kl_per_ventile"]
                stderrs = data["per_mode"][mode]["stderr_kl_per_ventile"]

                axes[row][col].bar(x, means, yerr=stderrs,
                                   capsize=2, alpha=0.8)
                axes[row][col].set_xticks(x)
                axes[row][col].set_xticklabels(
                    ventile_labels, rotation=45, ha="right", fontsize=7,
                )
                axes[row][col].set_title(
                    f"{mode} — {stratum_name} (n={data['count']})"
                )
                axes[row][col].set_ylabel("KL(student || teacher)")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"experiment_2_5{suffix}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")

        # Overlay plot: all modes on one axis, per stratum
        fig2, axes2 = plt.subplots(
            1, num_rows, figsize=(8 * num_rows, 5), squeeze=False,
        )
        colors = {
            "code_only": "#2196F3", "no_attempt": "#4CAF50",
            "with_thinking": "#FF5722", "reasoning_augmented": "#9C27B0",
            "reasoning_augmented_thinking": "#FF9800",
            "reasoning_in_response_thinking": "#795548",
        }

        for col, (stratum_name, data) in enumerate(plot_strata.items()):
            ax = axes2[0][col]
            for mode in ATTEMPT_MODES:
                means = data["per_mode"][mode]["mean_kl_per_ventile"]
                stderrs = data["per_mode"][mode]["stderr_kl_per_ventile"]
                ax.errorbar(
                    x, means, yerr=stderrs, label=mode,
                    color=colors[mode], capsize=2, marker="o", markersize=3,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(ventile_labels, rotation=45,
                               ha="right", fontsize=7)
            ax.set_title(
                f"KL by Teacher Prompt — {stratum_name} (n={data['count']})")
            ax.set_ylabel("KL(student || teacher)")
            ax.set_xlabel("Position ventile")
            ax.legend()

        plt.tight_layout()
        overlay_path = os.path.join(
            output_dir, f"experiment_2_5_overlay{suffix}.png")
        plt.savefig(overlay_path, dpi=150)
        plt.close()
        print(f"Overlay plot saved to: {overlay_path}")

        # Overlay plot: disagreement rate by teacher prompt, per stratum
        fig3, axes3 = plt.subplots(
            1, num_rows, figsize=(8 * num_rows, 5), squeeze=False,
        )

        for col, (stratum_name, data) in enumerate(plot_strata.items()):
            ax = axes3[0][col]
            for mode in ATTEMPT_MODES:
                means = data["per_mode"][mode]["mean_disagreement_per_ventile"]
                stderrs = data["per_mode"][mode]["stderr_disagreement_per_ventile"]
                ax.errorbar(
                    x, means, yerr=stderrs, label=mode,
                    color=colors[mode], capsize=2, marker="o", markersize=3,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(ventile_labels, rotation=45,
                               ha="right", fontsize=7)
            ax.set_title(
                f"Disagreement Rate by Teacher Prompt — {stratum_name} (n={data['count']})")
            ax.set_ylabel("Disagreement rate")
            ax.set_xlabel("Position ventile")
            ax.legend()

        plt.tight_layout()
        dis_overlay_path = os.path.join(
            output_dir, f"experiment_2_5_disagreement_overlay{suffix}.png")
        plt.savefig(dis_overlay_path, dpi=150)
        plt.close()
        print(f"Disagreement overlay plot saved to: {dis_overlay_path}")

    except ImportError:
        print("matplotlib not available, skipping plots")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2.5: Teacher Prompt Ablation — KL Distribution"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
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
