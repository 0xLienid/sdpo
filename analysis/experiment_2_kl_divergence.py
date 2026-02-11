"""
Experiment 2: KL Divergence Curve Over Position

Shows that the gap between the standard SDPO teacher signal and the
regenerated teacher signal grows across the sequence, quantifying how
much correction signal is lost when the teacher is forced to evaluate
the student's own (potentially flawed) trajectory.
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.experiment_1_reward_on_regen import (
    compute_reward,
    generate_from_message_batches,
)
from analysis.utils import (
    bin_into_ventiles,
    compute_topk_kl_per_position,
    get_completion_logits,
)
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import build_teacher_messages, build_teacher_regen_prompt

logger = logging.getLogger(__name__)


def aggregate_deciles(
    decile_lists: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """Compute mean and stderr for a list of decile vectors."""
    n = len(decile_lists)
    if n == 0:
        return [float("nan")] * 10, [float("nan")] * 10

    means = [sum(d[i] for d in decile_lists) / n for i in range(10)]
    stderrs = []
    for i in range(10):
        vals = [d[i] for d in decile_lists]
        var = sum((v - means[i]) ** 2 for v in vals) / \
            (n - 1) if n > 1 else 0.0
        stderrs.append(math.sqrt(var / n) if n > 0 else 0.0)
    return means, stderrs


def compute_rollout_kls(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    feedback_text: str,
    regen_completion: str,
    student_code: str,
    top_k: int = 20,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute per-position standard KL, regen KL, and delta-KL for one rollout pair.

    Returns dict with 'standard_kl', 'regen_kl', 'delta_kl' (all 1-D CPU tensors)
    and 'deciles' dict with 'standard', 'regen', 'delta' (lists of 10 floats).
    """
    # Student: model(completion | problem)
    student_msgs = [{"role": "user", "content": question}]

    # Standard teacher: model(completion | problem + student_attempt + feedback)
    std_teacher_full = build_teacher_messages(
        prompt=question, completion=completion,
        feedback=feedback_text, prior_solution=None, student_attempt=student_code,
    )
    std_teacher_msgs = [std_teacher_full[0]]

    # Regen teacher: model(regen_completion | problem + student_attempt + feedback)
    regen_teacher_full = build_teacher_messages(
        prompt=question, completion=regen_completion,
        feedback=feedback_text, prior_solution=None,
        student_attempt=student_code,
    )
    regen_teacher_msgs = [regen_teacher_full[0]]

    # Forward passes
    student_logits, _ = get_completion_logits(
        model, tokenizer, student_msgs, completion,
    )
    std_teacher_logits, _ = get_completion_logits(
        model, tokenizer, std_teacher_msgs, completion,
    )
    regen_teacher_logits, _ = get_completion_logits(
        model, tokenizer, regen_teacher_msgs, regen_completion,
    )

    # Per-position KL
    standard_kl = compute_topk_kl_per_position(
        student_logits, std_teacher_logits, top_k, temperature,
    )
    regen_kl = compute_topk_kl_per_position(
        student_logits, regen_teacher_logits, top_k, temperature,
    )

    # Truncate to min length for delta
    min_len = min(len(standard_kl), len(regen_kl))
    standard_kl_trunc = standard_kl[:min_len]
    regen_kl_trunc = regen_kl[:min_len]
    delta_kl = regen_kl_trunc - standard_kl_trunc

    return {
        "standard_kl": standard_kl_trunc.detach().cpu(),
        "regen_kl": regen_kl_trunc.detach().cpu(),
        "delta_kl": delta_kl.detach().cpu(),
        "seq_len": min_len,
        "deciles": {
            "standard": bin_into_ventiles(standard_kl_trunc.detach().cpu()),
            "regen": bin_into_ventiles(regen_kl_trunc.detach().cpu()),
            "delta": bin_into_ventiles(delta_kl.detach().cpu()),
        },
    }


def print_decile_table(
    title: str,
    mean_standard: List[float],
    mean_regen: List[float],
    mean_delta: List[float],
    stderr_delta: List[float],
    count: int,
) -> None:
    """Print a formatted decile summary table."""
    print(f"\n{title} (n={count})")
    print(f"{'Decile':<10} {'Std KL':<12} {'Regen KL':<12} {'Delta KL':<12} {'Stderr':<10}")
    print("-" * 56)
    for d in range(10):
        pct = f"{d * 10}-{(d + 1) * 10}%"
        print(
            f"{pct:<10} {mean_standard[d]:<12.4f} {mean_regen[d]:<12.4f} "
            f"{mean_delta[d]:<12.4f} {stderr_delta[d]:<10.4f}"
        )


def run_experiment_2(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 2: KL Divergence Curve Over Position."""

    print("=" * 60)
    print("Experiment 2: KL Divergence Curve Over Position")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(
        f"Problems: {num_problems} | Rollouts: {num_rollouts} | Top-K: {top_k}")
    print("=" * 60)
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

    # Collect per-rollout results, tagged by correctness category
    rollout_records: List[Dict[str, Any]] = []
    problem_results: List[Dict[str, Any]] = []

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        question = example.get("question_content", example.get("question", ""))
        print(f"Problem {prob_idx} ({title})...")

        # Generate original rollouts
        print(f"  Generating {num_rollouts} original rollouts...")
        original_rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Collect feedback
        feedbacks = []
        original_rewards = []
        regen_message_batches = []
        for rollout in original_rollouts:
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            feedbacks.append(fb)
            original_rewards.append(compute_reward(fb))

            regen_message_batches.append(build_teacher_regen_prompt(
                prompt=rollout.prompt,
                feedback=fb.feedback_text,
                student_attempt=extract_python_code(rollout.completion),
            ))

        # Generate regen rollouts (batched)
        print(f"  Generating {num_rollouts} regen rollouts...")
        regen_rollouts = generate_from_message_batches(
            model=model, tokenizer=tokenizer,
            message_batches=regen_message_batches,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )

        regen_rewards = []
        for regen_rollout in regen_rollouts:
            regen_fb = get_environment_feedback(
                prompt=question, completion=regen_rollout.completion,
                example=example,
            )
            regen_rewards.append(compute_reward(regen_fb))

        # Compute KL for each rollout pair
        rollout_kl_results = []
        for r_idx, (rollout, fb, regen_rollout) in enumerate(
            zip(original_rollouts, feedbacks, regen_rollouts)
        ):
            orig_correct = original_rewards[r_idx] == 1.0
            regen_correct = regen_rewards[r_idx] == 1.0

            if not orig_correct and regen_correct:
                category = "incorrect_to_correct"
            else:
                category = "other"

            print(
                f"  Rollout {r_idx}: orig={original_rewards[r_idx]:.2f} "
                f"regen={regen_rewards[r_idx]:.2f} [{category}] computing KL..."
            )
            kl_result = compute_rollout_kls(
                model, tokenizer,
                question=question,
                completion=rollout.completion,
                feedback_text=fb.feedback_text,
                regen_completion=regen_rollout.completion,
                student_code=extract_python_code(rollout.completion),
                top_k=top_k,
            )

            record = {
                "problem_idx": prob_idx,
                "rollout_idx": r_idx,
                "original_reward": original_rewards[r_idx],
                "regen_reward": regen_rewards[r_idx],
                "original_correct": orig_correct,
                "regen_correct": regen_correct,
                "category": category,
                "seq_len": kl_result["seq_len"],
                "deciles": kl_result["deciles"],
            }
            rollout_records.append(record)
            rollout_kl_results.append(record)

        problem_results.append({
            "problem_idx": prob_idx,
            "question_title": title,
            "mean_original_reward": sum(original_rewards) / len(original_rewards),
            "mean_regen_reward": sum(regen_rewards) / len(regen_rewards),
            "rollouts": rollout_kl_results,
        })
        print()

    # ------------------------------------------------------------------
    # Aggregate: all rollouts, incorrect→correct, other
    # ------------------------------------------------------------------
    strata = {
        "all": rollout_records,
        "incorrect_to_correct": [r for r in rollout_records if r["category"] == "incorrect_to_correct"],
        "other": [r for r in rollout_records if r["category"] == "other"],
    }

    summary: Dict[str, Any] = {}
    print("=" * 60)
    for stratum_name, records in strata.items():
        if not records:
            summary[stratum_name] = {"count": 0}
            continue

        delta_dec = [r["deciles"]["delta"] for r in records]
        std_dec = [r["deciles"]["standard"] for r in records]
        regen_dec = [r["deciles"]["regen"] for r in records]

        mean_std, _ = aggregate_deciles(std_dec)
        mean_regen, _ = aggregate_deciles(regen_dec)
        mean_delta, stderr_delta = aggregate_deciles(delta_dec)

        summary[stratum_name] = {
            "count": len(records),
            "mean_standard_kl_per_decile": mean_std,
            "mean_regen_kl_per_decile": mean_regen,
            "mean_delta_kl_per_decile": mean_delta,
            "stderr_delta_kl_per_decile": stderr_delta,
        }

        print_decile_table(
            f"SUMMARY — {stratum_name}",
            mean_std, mean_regen, mean_delta, stderr_delta,
            len(records),
        )
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "timestamp": datetime.now().isoformat(),
        },
        "problems": problem_results,
        "summary": summary,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_2.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # ------------------------------------------------------------------
    # Plot: one row per stratum, 3 columns (standard, regen, delta)
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = {k: v for k,
                       v in summary.items() if v.get("count", 0) > 0}
        num_rows = len(plot_strata)
        fig, axes = plt.subplots(num_rows, 3, figsize=(
            18, 5 * num_rows), squeeze=False)
        decile_labels = [f"{d * 10}-{(d + 1) * 10}%" for d in range(10)]
        x = range(10)

        for row, (stratum_name, data) in enumerate(plot_strata.items()):
            ms = data["mean_standard_kl_per_decile"]
            mr = data["mean_regen_kl_per_decile"]
            md = data["mean_delta_kl_per_decile"]
            se = data["stderr_delta_kl_per_decile"]

            axes[row][0].bar(x, ms)
            axes[row][0].set_xticks(x)
            axes[row][0].set_xticklabels(
                decile_labels, rotation=45, ha="right")
            axes[row][0].set_title(
                f"Standard KL — {stratum_name} (n={data['count']})")
            axes[row][0].set_ylabel("KL Divergence")

            axes[row][1].bar(x, mr)
            axes[row][1].set_xticks(x)
            axes[row][1].set_xticklabels(
                decile_labels, rotation=45, ha="right")
            axes[row][1].set_title(f"Regen KL — {stratum_name}")
            axes[row][1].set_ylabel("KL Divergence")

            axes[row][2].bar(x, md, yerr=se, capsize=3)
            axes[row][2].set_xticks(x)
            axes[row][2].set_xticklabels(
                decile_labels, rotation=45, ha="right")
            axes[row][2].set_title(f"Delta KL — {stratum_name}")
            axes[row][2].set_ylabel("Delta KL")
            axes[row][2].axhline(
                y=0, color="gray", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "experiment_2.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2: KL Divergence Curve Over Position"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_experiment_2(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
