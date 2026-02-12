"""
Experiment 3: Standard KL Convergence Over Position Across Training Steps

Tracks how the standard SDPO teacher-student KL (same completion, teacher has
feedback context) evolves per-ventile across training steps using an EMA teacher.
Shows where in the sequence the KL is highest and how it converges.
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

from analysis.experiment_1_reward_on_regen import compute_reward
from analysis.utils import (
    bin_into_ventiles,
    compute_topk_kl_per_position,
    get_completion_logits,
)
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import (
    EMATeacher,
    SDPOHparams,
    build_teacher_messages,
    compute_sdpo_loss_batched,
)

logger = logging.getLogger(__name__)

NUM_VENTILES = 20
STRATA = ["all", "incorrect", "correct"]


def aggregate_ventiles(
    ventile_lists: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """Compute mean and stderr for a list of ventile vectors."""
    n = len(ventile_lists)
    if n == 0:
        return [float("nan")] * NUM_VENTILES, [float("nan")] * NUM_VENTILES

    means = [sum(d[i] for d in ventile_lists) / n for i in range(NUM_VENTILES)]
    stderrs = []
    for i in range(NUM_VENTILES):
        vals = [d[i] for d in ventile_lists]
        var = sum((v - means[i]) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        stderrs.append(math.sqrt(var / n) if n > 0 else 0.0)
    return means, stderrs


def compute_standard_kl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    feedback_text: str,
    student_code: str,
    top_k: int = 20,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute per-position standard KL for one rollout.

    Standard KL = KL(student || teacher) where both evaluate the same completion,
    student is conditioned on problem only, teacher on problem + feedback.

    Returns dict with 'kl' (1-D CPU tensor), 'ventiles' (list of 20 floats),
    and 'seq_len'.
    """
    student_msgs = [{"role": "user", "content": question}]

    teacher_full = build_teacher_messages(
        prompt=question, completion=completion,
        feedback=feedback_text, prior_solution=None,
        student_attempt=student_code,
    )
    teacher_msgs = [teacher_full[0]]

    student_logits, _ = get_completion_logits(
        model, tokenizer, student_msgs, completion,
    )
    teacher_logits, _ = get_completion_logits(
        model, tokenizer, teacher_msgs, completion,
    )

    kl = compute_topk_kl_per_position(
        student_logits, teacher_logits, top_k, temperature,
    )

    return {
        "kl": kl.detach().cpu(),
        "ventiles": bin_into_ventiles(kl.detach().cpu()),
        "seq_len": len(kl),
    }


def measure_standard_kl_ventiles(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute standard KL per ventile for all rollouts, stratified by correctness.

    Returns a dict keyed by stratum name, each containing mean/stderr ventile
    vectors, peak ventile index, and mean KL.
    """
    per_rollout: List[Dict[str, Any]] = []

    for item in rollout_data:
        kl_result = compute_standard_kl(
            model, tokenizer,
            question=item["question"],
            completion=item["completion"],
            feedback_text=item["feedback_text"],
            student_code=item["student_code"],
            top_k=top_k,
        )
        per_rollout.append({
            "category": item["category"],
            "ventiles": kl_result["ventiles"],
            "mean_kl": kl_result["kl"].mean().item() if kl_result["seq_len"] > 0 else float("nan"),
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
                "mean_ventiles": [float("nan")] * NUM_VENTILES,
                "stderr_ventiles": [float("nan")] * NUM_VENTILES,
                "peak_ventile": -1,
                "mean_kl": float("nan"),
            }
            continue

        mean_v, stderr_v = aggregate_ventiles([r["ventiles"] for r in subset])
        peak_ventile = max(range(NUM_VENTILES), key=lambda i: mean_v[i])
        mean_kl = sum(r["mean_kl"] for r in subset) / len(subset)

        results[stratum] = {
            "count": len(subset),
            "mean_ventiles": mean_v,
            "stderr_ventiles": stderr_v,
            "peak_ventile": peak_ventile,
            "mean_kl": mean_kl,
        }

    return results


def sdpo_training_step(
    model: AutoModelForCausalLM,
    teacher: EMATeacher,
    optimizer: torch.optim.Optimizer,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    hparams: SDPOHparams,
) -> float:
    """
    Perform one SDPO training step over all rollouts (gradient accumulation).

    Uses a separate EMA teacher model (matching the real sdpo_train loop).
    After the optimizer step, updates the teacher via EMA.
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    n = len(rollout_data)

    for item in rollout_data:
        loss, _ = compute_sdpo_loss_batched(
            student_model=model,
            teacher_model=teacher.model,
            tokenizer=tokenizer,
            prompts=[item["question"]],
            completions=[item["completion"]],
            feedbacks=[item["feedback_text"]],
            prior_solutions=[None],
            hparams=hparams,
            student_attempts=[item["student_code"]],
        )
        (loss / n).backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)
    optimizer.step()

    teacher.update(model)

    return total_loss / n


def run_experiment_3(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 15,
    num_rollouts: int = 4,
    num_training_steps: int = 10,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    learning_rate: float = 1e-5,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 3: Standard KL Convergence Over Position Across Training Steps."""

    print("=" * 60)
    print("Experiment 3: Standard KL Over Position")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts: {num_rollouts}")
    print(f"Training steps: {num_training_steps} | LR: {learning_rate}")
    print(f"Top-K: {top_k}")
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
    # Phase 1: Generate rollouts and collect feedback
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

        for rollout in original_rollouts:
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(fb)
            category = "correct" if reward == 1.0 else "incorrect"

            rollout_data.append({
                "question": question,
                "completion": rollout.completion,
                "feedback_text": fb.feedback_text,
                "student_code": extract_python_code(rollout.completion),
                "category": category,
                "reward": reward,
            })

    counts = {s: sum(1 for r in rollout_data if r["category"] == s) for s in ["correct", "incorrect"]}
    print(f"\nTotal rollouts: {len(rollout_data)}")
    print(f"  correct: {counts['correct']}")
    print(f"  incorrect: {counts['incorrect']}\n")

    # -------------------------------------------------------------------
    # Phase 2: Setup optimizer, EMA teacher, and SDPO hyperparameters
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

    ema_decay = 1.0 - hparams.teacher_ema_rate
    print(f"Creating EMA teacher (decay={ema_decay})...")
    teacher = EMATeacher(model, decay=ema_decay)

    # -------------------------------------------------------------------
    # Phase 3: Training loop with measurement at each step
    # -------------------------------------------------------------------
    steps_data: List[Dict[str, Any]] = []

    for step in range(num_training_steps + 1):
        model.eval()
        print(f"Step {step}: measuring standard KL...")
        stratified = measure_standard_kl_ventiles(
            model, tokenizer, rollout_data, top_k=top_k,
        )

        step_record: Dict[str, Any] = {"step": step}
        for stratum in STRATA:
            data = stratified[stratum]
            step_record[stratum] = {
                "count": data["count"],
                "mean_ventiles": data["mean_ventiles"],
                "stderr_ventiles": data["stderr_ventiles"],
                "peak_ventile": data["peak_ventile"],
                "mean_kl": data["mean_kl"],
            }

        steps_data.append(step_record)

        for stratum in STRATA:
            data = stratified[stratum]
            if data["count"] == 0:
                continue
            peak_v = data["peak_ventile"]
            peak_pct = f"{peak_v * 5}-{(peak_v + 1) * 5}%" if peak_v >= 0 else "N/A"
            peak_val = data["mean_ventiles"][peak_v] if peak_v >= 0 else float("nan")
            print(f"  {stratum} (n={data['count']}): "
                  f"mean_kl={data['mean_kl']:.4f}  "
                  f"peak={peak_pct} ({peak_val:.4f})")

        # Train (skip after last measurement)
        if step < num_training_steps:
            print(f"Step {step}: training...")
            loss = sdpo_training_step(
                model, teacher, optimizer, tokenizer, rollout_data, hparams,
            )
            print(f"  Loss: {loss:.6f}")

        print()

    # -------------------------------------------------------------------
    # Compute percentage-change metrics from raw ventile data
    # -------------------------------------------------------------------
    for stratum in STRATA:
        if steps_data[0][stratum]["count"] == 0:
            continue

        baseline = steps_data[0][stratum]["mean_ventiles"]
        baseline_mean_kl = steps_data[0][stratum]["mean_kl"]

        for i, rec in enumerate(steps_data):
            s = rec[stratum]

            # Cumulative % change from baseline: (KL_t - KL_0) / KL_0
            s["cumulative_pct_ventiles"] = [
                ((s["mean_ventiles"][d] - baseline[d]) / baseline[d] * 100.0
                 if baseline[d] != 0 else float("nan"))
                for d in range(NUM_VENTILES)
            ]
            s["cumulative_pct_mean_kl"] = (
                (s["mean_kl"] - baseline_mean_kl) / baseline_mean_kl * 100.0
                if baseline_mean_kl != 0 else float("nan")
            )

            # Step-over-step % change: (KL_t - KL_{t-1}) / KL_{t-1}
            if i == 0:
                s["step_pct_ventiles"] = [0.0] * NUM_VENTILES
                s["step_pct_mean_kl"] = 0.0
            else:
                prev = steps_data[i - 1][stratum]
                s["step_pct_ventiles"] = [
                    ((s["mean_ventiles"][d] - prev["mean_ventiles"][d])
                     / prev["mean_ventiles"][d] * 100.0
                     if prev["mean_ventiles"][d] != 0 else float("nan"))
                    for d in range(NUM_VENTILES)
                ]
                s["step_pct_mean_kl"] = (
                    (s["mean_kl"] - prev["mean_kl"]) / prev["mean_kl"] * 100.0
                    if prev["mean_kl"] != 0 else float("nan")
                )

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    for stratum in STRATA:
        if steps_data[0][stratum]["count"] == 0:
            continue
        print("=" * 80)
        print(f"SUMMARY — {stratum} (n={steps_data[0][stratum]['count']})")

        # Raw KL table
        print("-" * 80)
        print("Raw KL:")
        header = (f"{'Step':<6}{'Loss':<10}"
                  + "".join(f"{'V' + str(d):<8}" for d in range(NUM_VENTILES))
                  + f"  {'Peak':<8}{'Mean':<8}")
        print(header)
        print("-" * len(header))
        for rec in steps_data:
            s = rec[stratum]
            peak_v = s["peak_ventile"]
            peak_label = f"V{peak_v}" if peak_v >= 0 else "N/A"
            loss_str = f"{rec.get('loss', 0.0):<10.6f}" if "loss" in rec else f"{'':<10}"
            line = (f"{rec['step']:<6}{loss_str}"
                    + "".join(f"{v:<8.4f}" for v in s["mean_ventiles"])
                    + f"  {peak_label:<8}{s['mean_kl']:<8.4f}")
            print(line)

        # Cumulative % change table
        print()
        print("Cumulative % change from baseline:")
        header = (f"{'Step':<6}"
                  + "".join(f"{'V' + str(d):<8}" for d in range(NUM_VENTILES))
                  + f"  {'Mean':<8}")
        print(header)
        print("-" * len(header))
        for rec in steps_data:
            s = rec[stratum]
            line = (f"{rec['step']:<6}"
                    + "".join(f"{v:<8.1f}" for v in s["cumulative_pct_ventiles"])
                    + f"  {s['cumulative_pct_mean_kl']:<8.1f}")
            print(line)

        # Step-over-step % change table
        print()
        print("Step-over-step % change:")
        print(header)
        print("-" * len(header))
        for rec in steps_data:
            s = rec[stratum]
            line = (f"{rec['step']:<6}"
                    + "".join(f"{v:<8.1f}" for v in s["step_pct_ventiles"])
                    + f"  {s['step_pct_mean_kl']:<8.1f}")
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
            "ema_decay": ema_decay,
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
    # Plot: 3 rows (raw KL, cumulative %, step-over-step %) x strata cols
    # -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = [s for s in STRATA if steps_data[0][s]["count"] > 0]
        num_cols = len(plot_strata)
        row_configs = [
            ("mean_ventiles", "Standard KL (nats/token)", "stderr_ventiles"),
            ("cumulative_pct_ventiles", "Cumulative % change from baseline", None),
            ("step_pct_ventiles", "Step-over-step % change", None),
        ]
        num_rows = len(row_configs)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows),
            squeeze=False,
        )
        steps_x = [rec["step"] for rec in steps_data]
        cmap = plt.cm.viridis

        for row, (key, ylabel, stderr_key) in enumerate(row_configs):
            for col, stratum in enumerate(plot_strata):
                ax = axes[row][col]
                n = steps_data[0][stratum]["count"]

                for d in range(NUM_VENTILES):
                    values = [rec[stratum][key][d] for rec in steps_data]
                    color = cmap(d / (NUM_VENTILES - 1))
                    label = f"{d * 5}-{(d + 1) * 5}%"
                    ax.plot(steps_x, values, color=color, label=label, linewidth=1.5)

                    if stderr_key is not None:
                        stderrs = [rec[stratum][stderr_key][d] for rec in steps_data]
                        ax.fill_between(
                            steps_x,
                            [v - s for v, s in zip(values, stderrs)],
                            [v + s for v, s in zip(values, stderrs)],
                            color=color, alpha=0.15,
                        )

                # Add zero line for percentage-change rows
                if row > 0:
                    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

                ax.set_xlabel("Training Step")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{stratum} (n={n})")
                if col == num_cols - 1:
                    ax.legend(
                        title="Position Ventile", bbox_to_anchor=(1.02, 1),
                        loc="upper left", fontsize="x-small", ncol=2,
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
        description="Experiment 3: Standard KL Convergence Over Position"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--num-training-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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
