"""
Experiment 3: KL Convergence Over Position Across Training Steps

Shows that earlier token positions reduce their delta-KL faster than
later tokens under standard SDPO training, confirming that the model
learns from clean early signal but struggles with corrupted late signal.
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.experiment_1_reward_on_regen import (
    compute_reward,
    generate_from_message_batches,
)
from analysis.experiment_2_kl_divergence import aggregate_ventiles, compute_rollout_kls
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import SDPOHparams, build_teacher_regen_prompt, compute_sdpo_loss_batched

logger = logging.getLogger(__name__)

STRATA = ["all", "incorrect_to_correct", "other"]


def measure_kl_ventiles(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute KL per ventile for all rollout pairs, stratified by correctness.

    Returns a dict keyed by stratum name ("all", "incorrect_to_correct", "other"),
    each containing mean/stderr ventile vectors and peak_standard_ventile.
    """
    per_rollout: List[Dict[str, Any]] = []

    for item in rollout_data:
        kl_result = compute_rollout_kls(
            model, tokenizer,
            question=item["question"],
            completion=item["completion"],
            feedback_text=item["feedback_text"],
            regen_completion=item["regen_completion"],
            student_code=item["student_code"],
            top_k=top_k,
        )
        per_rollout.append({
            "category": item["category"],
            "delta": kl_result["ventiles"]["delta"],
            "standard": kl_result["ventiles"]["standard"],
            "regen": kl_result["ventiles"]["regen"],
        })

    results: Dict[str, Dict[str, Any]] = {}
    for stratum in STRATA:
        if stratum == "all":
            subset = per_rollout
        else:
            subset = [r for r in per_rollout if r["category"] == stratum]

        if not subset:
            results[stratum] = {
                "count": 0,
                "mean_delta": [float("nan")] * 20,
                "stderr_delta": [float("nan")] * 20,
                "mean_standard": [float("nan")] * 20,
                "stderr_standard": [float("nan")] * 20,
                "mean_regen": [float("nan")] * 20,
                "peak_standard_ventile": -1,
            }
            continue

        mean_delta, stderr_delta = aggregate_ventiles([r["delta"] for r in subset])
        mean_std, stderr_std = aggregate_ventiles([r["standard"] for r in subset])
        mean_regen, _ = aggregate_ventiles([r["regen"] for r in subset])

        peak_ventile = max(range(20), key=lambda i: mean_std[i])

        results[stratum] = {
            "count": len(subset),
            "mean_delta": mean_delta,
            "stderr_delta": stderr_delta,
            "mean_standard": mean_std,
            "stderr_standard": stderr_std,
            "mean_regen": mean_regen,
            "peak_standard_ventile": peak_ventile,
        }

    return results


def sdpo_training_step(
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    hparams: SDPOHparams,
) -> float:
    """
    Perform one SDPO training step over all rollouts (gradient accumulation).

    Uses the same model as both student and teacher: the student forward pass
    runs with gradients, and the teacher forward pass runs under torch.no_grad().
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    n = len(rollout_data)

    for item in rollout_data:
        loss, _ = compute_sdpo_loss_batched(
            student_model=model,
            teacher_model=model,
            tokenizer=tokenizer,
            prompts=[item["question"]],
            completions=[item["completion"]],
            feedbacks=[item["feedback_text"]],
            prior_solutions=[None],
            hparams=hparams,
            student_attempts=[item["student_code"]],
        )
        # Accumulate gradients, averaging across rollouts
        (loss / n).backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)
    optimizer.step()

    return total_loss / n


def run_experiment_3(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 4,
    num_training_steps: int = 10,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    learning_rate: float = 1e-6,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 3: KL Convergence Over Position Across Training Steps."""

    print("=" * 60)
    print("Experiment 3: KL Convergence Over Position")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts: {num_rollouts}")
    print(f"Training steps: {num_training_steps} | LR: {learning_rate}")
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems\n")

    # -------------------------------------------------------------------
    # Phase 1: Generate all rollouts, feedback, and regen rollouts (frozen)
    # -------------------------------------------------------------------
    print("Generating rollouts and collecting feedback...")
    rollout_data: List[Dict[str, Any]] = []

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        question = example.get("question_content", example.get("question", ""))
        print(f"  Problem {prob_idx} ({title}): generating {num_rollouts} rollouts...")

        original_rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

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

        print(f"  Problem {prob_idx}: generating {num_rollouts} regen rollouts...")
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

        for r_idx, (rollout, fb, regen_rollout) in enumerate(
            zip(original_rollouts, feedbacks, regen_rollouts)
        ):
            orig_correct = original_rewards[r_idx] == 1.0
            regen_correct = regen_rewards[r_idx] == 1.0
            category = "incorrect_to_correct" if (not orig_correct and regen_correct) else "other"

            rollout_data.append({
                "question": question,
                "completion": rollout.completion,
                "feedback_text": fb.feedback_text,
                "regen_completion": regen_rollout.completion,
                "student_code": extract_python_code(rollout.completion),
                "category": category,
                "original_reward": original_rewards[r_idx],
                "regen_reward": regen_rewards[r_idx],
            })

    counts = {s: sum(1 for r in rollout_data if r["category"] == s) for s in ["incorrect_to_correct", "other"]}
    print(f"\nTotal rollout pairs: {len(rollout_data)}")
    print(f"  incorrect_to_correct: {counts['incorrect_to_correct']}")
    print(f"  other: {counts['other']}\n")

    # -------------------------------------------------------------------
    # Phase 2: Setup optimizer and SDPO hyperparameters
    # -------------------------------------------------------------------
    hparams = SDPOHparams(
        learning_rate=learning_rate,
        top_k_distillation=top_k,
        temperature=1.0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hparams.weight_decay,
    )

    # -------------------------------------------------------------------
    # Phase 3: Training loop with measurement at each step
    # -------------------------------------------------------------------
    steps_data: List[Dict[str, Any]] = []

    for step in range(num_training_steps + 1):
        # Measure KL (eval mode, no dropout)
        model.eval()
        print(f"Step {step}: measuring KL ventiles...")
        stratified = measure_kl_ventiles(model, tokenizer, rollout_data, top_k=top_k)

        step_record: Dict[str, Any] = {"step": step}
        for stratum in STRATA:
            data = stratified[stratum]
            step_record[stratum] = {
                "count": data["count"],
                "mean_delta_per_ventile": data["mean_delta"],
                "stderr_delta_per_ventile": data["stderr_delta"],
                "mean_standard_per_ventile": data["mean_standard"],
                "stderr_standard_per_ventile": data["stderr_standard"],
                "mean_regen_per_ventile": data["mean_regen"],
                "peak_standard_ventile": data["peak_standard_ventile"],
            }

        steps_data.append(step_record)

        for stratum in STRATA:
            data = stratified[stratum]
            if data["count"] == 0:
                continue
            peak_v = data["peak_standard_ventile"]
            peak_pct = f"{peak_v * 5}-{(peak_v + 1) * 5}%" if peak_v >= 0 else "N/A"
            peak_val = data["mean_standard"][peak_v] if peak_v >= 0 else float("nan")
            delta_str = " ".join(f"{v:.4f}" for v in data["mean_delta"])
            print(f"  {stratum} (n={data['count']}): delta=[{delta_str}]")
            print(f"    peak std KL: ventile {peak_pct} = {peak_val:.4f}")

        # Train (skip after last measurement)
        if step < num_training_steps:
            print(f"Step {step}: training...")
            loss = sdpo_training_step(
                model, optimizer, tokenizer, rollout_data, hparams,
            )
            print(f"  Loss: {loss:.6f}")

        print()

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    for stratum in STRATA:
        if steps_data[0][stratum]["count"] == 0:
            continue
        print("=" * 80)
        print(f"SUMMARY — {stratum} (n={steps_data[0][stratum]['count']})")

        # Delta KL table
        print("-" * 80)
        print("Delta KL per ventile:")
        header = f"{'Step':<6}" + "".join(f"{'V' + str(d):<8}" for d in range(20))
        print(header)
        print("-" * len(header))
        for rec in steps_data:
            line = f"{rec['step']:<6}" + "".join(
                f"{v:<8.4f}" for v in rec[stratum]["mean_delta_per_ventile"]
            )
            print(line)

        # Standard KL table
        print()
        print("Standard KL per ventile:")
        header = f"{'Step':<6}" + "".join(f"{'V' + str(d):<8}" for d in range(20)) + f"  {'Peak':<8}"
        print(header)
        print("-" * len(header))
        for rec in steps_data:
            peak_v = rec[stratum]["peak_standard_ventile"]
            peak_label = f"V{peak_v}" if peak_v >= 0 else "N/A"
            line = f"{rec['step']:<6}" + "".join(
                f"{v:<8.4f}" for v in rec[stratum]["mean_standard_per_ventile"]
            ) + f"  {peak_label:<8}"
            print(line)

    print("=" * 80)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "num_training_steps": num_training_steps,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
        },
        "strata_counts": counts,
        "steps": steps_data,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_3.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # -------------------------------------------------------------------
    # Plot: 2 rows (delta KL, standard KL) x N columns (one per stratum)
    # -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = [s for s in STRATA if steps_data[0][s]["count"] > 0]
        num_cols = len(plot_strata)
        fig, axes = plt.subplots(
            2, num_cols, figsize=(10 * num_cols, 12), squeeze=False,
        )
        steps_x = [rec["step"] for rec in steps_data]
        cmap = plt.cm.viridis

        for col, stratum in enumerate(plot_strata):
            n = steps_data[0][stratum]["count"]

            # Row 0: Delta KL per ventile
            ax = axes[0][col]
            for d in range(20):
                means = [rec[stratum]["mean_delta_per_ventile"][d] for rec in steps_data]
                stderrs = [rec[stratum]["stderr_delta_per_ventile"][d] for rec in steps_data]
                color = cmap(d / 19)
                label = f"{d * 5}-{(d + 1) * 5}%"
                ax.plot(steps_x, means, color=color, label=label, linewidth=1.5)
                ax.fill_between(
                    steps_x,
                    [m - s for m, s in zip(means, stderrs)],
                    [m + s for m, s in zip(means, stderrs)],
                    color=color, alpha=0.15,
                )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Delta KL (Regen - Standard)")
            ax.set_title(f"Delta KL — {stratum} (n={n})")
            ax.legend(
                title="Position Ventile", bbox_to_anchor=(1.02, 1), loc="upper left",
                fontsize="x-small", ncol=2,
            )
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

            # Row 1: Standard KL per ventile
            ax = axes[1][col]
            for d in range(20):
                means = [rec[stratum]["mean_standard_per_ventile"][d] for rec in steps_data]
                stderrs = [rec[stratum]["stderr_standard_per_ventile"][d] for rec in steps_data]
                color = cmap(d / 19)
                label = f"{d * 5}-{(d + 1) * 5}%"
                ax.plot(steps_x, means, color=color, label=label, linewidth=1.5)
                ax.fill_between(
                    steps_x,
                    [m - s for m, s in zip(means, stderrs)],
                    [m + s for m, s in zip(means, stderrs)],
                    color=color, alpha=0.15,
                )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Standard KL")
            ax.set_title(f"Standard KL — {stratum} (n={n})")
            ax.legend(
                title="Position Ventile", bbox_to_anchor=(1.02, 1), loc="upper left",
                fontsize="x-small", ncol=2,
            )

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "experiment_3.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3: KL Convergence Over Position Across Training Steps"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--num-training-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_experiment_3(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        num_training_steps=args.num_training_steps,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
