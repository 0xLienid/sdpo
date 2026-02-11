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
    generate_from_message_batches,
)
from analysis.experiment_2_kl_divergence import compute_rollout_kls
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import SDPOHparams, build_teacher_regen_prompt, compute_sdpo_loss_batched

logger = logging.getLogger(__name__)


def measure_delta_kl_deciles(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Compute delta-KL per decile for all rollout pairs using the current model weights.

    Returns dict with per-rollout deciles and aggregated mean/stderr.
    """
    all_delta_deciles: List[List[float]] = []
    all_standard_deciles: List[List[float]] = []
    all_regen_deciles: List[List[float]] = []

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
        all_delta_deciles.append(kl_result["deciles"]["delta"])
        all_standard_deciles.append(kl_result["deciles"]["standard"])
        all_regen_deciles.append(kl_result["deciles"]["regen"])

    n = len(all_delta_deciles)
    mean_delta = [sum(d[i] for d in all_delta_deciles) / n for i in range(10)]
    mean_standard = [sum(d[i] for d in all_standard_deciles) / n for i in range(10)]
    mean_regen = [sum(d[i] for d in all_regen_deciles) / n for i in range(10)]

    stderr_delta = []
    for i in range(10):
        vals = [d[i] for d in all_delta_deciles]
        var = sum((v - mean_delta[i]) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        stderr_delta.append(math.sqrt(var / n) if n > 0 else 0.0)

    return {
        "mean_delta": mean_delta,
        "mean_standard": mean_standard,
        "mean_regen": mean_regen,
        "stderr_delta": stderr_delta,
        "per_rollout_delta": all_delta_deciles,
    }


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
        regen_message_batches = []
        for rollout in original_rollouts:
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            feedbacks.append(fb)
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

        for rollout, fb, regen_rollout in zip(original_rollouts, feedbacks, regen_rollouts):
            rollout_data.append({
                "question": question,
                "completion": rollout.completion,
                "feedback_text": fb.feedback_text,
                "regen_completion": regen_rollout.completion,
                "student_code": extract_python_code(rollout.completion),
            })

    print(f"\nTotal rollout pairs: {len(rollout_data)}\n")

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
        # Measure delta-KL (eval mode, no dropout)
        model.eval()
        print(f"Step {step}: measuring delta-KL...")
        metrics = measure_delta_kl_deciles(model, tokenizer, rollout_data, top_k=top_k)

        step_record = {
            "step": step,
            "mean_delta_per_decile": metrics["mean_delta"],
            "mean_standard_per_decile": metrics["mean_standard"],
            "mean_regen_per_decile": metrics["mean_regen"],
            "stderr_delta_per_decile": metrics["stderr_delta"],
        }
        steps_data.append(step_record)

        delta_summary = " ".join(f"{v:.4f}" for v in metrics["mean_delta"])
        print(f"  Delta-KL by decile: [{delta_summary}]")

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
    print("=" * 60)
    print("SUMMARY: Delta-KL per Decile Over Training Steps")
    print("=" * 60)
    header = f"{'Step':<6}" + "".join(f"{'D' + str(d):<10}" for d in range(10))
    print(header)
    print("-" * len(header))
    for rec in steps_data:
        line = f"{rec['step']:<6}" + "".join(
            f"{v:<10.4f}" for v in rec["mean_delta_per_decile"]
        )
        print(line)
    print("=" * 60)

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
        "steps": steps_data,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_3.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # -------------------------------------------------------------------
    # Plot: one line per decile, x = step, y = delta-KL
    # -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))
        steps = [rec["step"] for rec in steps_data]
        cmap = plt.cm.viridis

        for d in range(10):
            means = [rec["mean_delta_per_decile"][d] for rec in steps_data]
            stderrs = [rec["stderr_delta_per_decile"][d] for rec in steps_data]
            color = cmap(d / 9)
            label = f"{d * 10}-{(d + 1) * 10}%"
            ax.plot(steps, means, color=color, label=label, linewidth=1.5)
            ax.fill_between(
                steps,
                [m - s for m, s in zip(means, stderrs)],
                [m + s for m, s in zip(means, stderrs)],
                color=color, alpha=0.15,
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Delta KL (Regen - Standard)")
        ax.set_title("Delta-KL Convergence by Decile Over SDPO Training")
        ax.legend(title="Position Decile", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
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
