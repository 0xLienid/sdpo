"""
Experiment 3: KL Distribution Shift Across Iterative SDPO Generations

For each problem independently, runs an iterative generate-train loop:
  1. Generate rollouts with the current model
  2. Get environment feedback
  3. Measure per-ventile KL(student || teacher) on those rollouts
  4. Train for a few inner steps on those rollouts
  5. Repeat from (1)

The primary signal is how the KL distribution across ventiles shifts over
successive generations.  If SDPO is working, the peak KL should migrate later
in the sequence as the model learns to get early tokens right.  If prefix
corruption dominates, the peak stays stuck early.
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
    num_outer_steps: int = 5,
    num_inner_steps: int = 3,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    learning_rate: float = 1e-5,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 3: KL Distribution Shift Across Iterative SDPO Generations.

    For each problem independently:
      - Restore model to initial weights
      - For each outer step (generation round):
        1. Generate fresh rollouts, get feedback, measure KL ventiles
        2. Train for num_inner_steps on those rollouts
      - Take a final measurement after the last training round

    The x-axis of the output is "generation" (outer step), not gradient step.
    Percentage-change metrics are computed per-rollout-generation before
    aggregation across problems.
    """

    print("=" * 60)
    print("Experiment 3: KL Shift Across Generations (per-problem)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts/gen: {num_rollouts}")
    print(f"Outer steps (generations): {num_outer_steps}")
    print(f"Inner steps (training/gen): {num_inner_steps}")
    print(f"LR: {learning_rate} | Top-K: {top_k}")
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

    # Save initial weights — restored before each problem
    print("Saving initial model weights...")
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    hparams = SDPOHparams(
        learning_rate=learning_rate,
        top_k_distillation=top_k,
        temperature=1.0,
    )
    ema_decay = 1.0 - hparams.teacher_ema_rate

    # -------------------------------------------------------------------
    # Per-problem generate-train loop
    # -------------------------------------------------------------------
    # Per generation we collect one "snapshot" per rollout.  A snapshot has
    # the ventiles, mean_kl, category, and reward for that rollout.
    # all_snapshots[gen_idx] is a list of snapshots across all problems.
    all_snapshots: List[List[Dict[str, Any]]] = [
        [] for _ in range(num_outer_steps + 1)
    ]

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        question = example.get("question_content", example.get("question", ""))

        # Restore model to initial weights
        model.load_state_dict({k: v.clone() for k, v in initial_state.items()})
        model.eval()

        print(f"Problem {prob_idx}/{len(dataset) - 1} ({title})")

        # Create fresh EMA teacher and optimizer for this problem
        teacher = EMATeacher(model, decay=ema_decay)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=hparams.weight_decay,
        )

        for gen in range(num_outer_steps):
            # ----- Generate rollouts -----
            model.eval()
            rollouts = livecodebench_rollout(
                model, tokenizer, example,
                num_rollouts=num_rollouts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            # ----- Feedback + KL measurement -----
            gen_rollout_data: List[Dict[str, Any]] = []
            for rollout in rollouts:
                fb = get_environment_feedback(
                    prompt=rollout.prompt, completion=rollout.completion,
                    example=example,
                )
                reward = compute_reward(fb)
                category = "correct" if reward == 1.0 else "incorrect"
                student_code = extract_python_code(rollout.completion)

                kl_result = compute_standard_kl(
                    model, tokenizer,
                    question=question,
                    completion=rollout.completion,
                    feedback_text=fb.feedback_text,
                    student_code=student_code,
                    top_k=top_k,
                )

                gen_rollout_data.append({
                    "question": question,
                    "completion": rollout.completion,
                    "feedback_text": fb.feedback_text,
                    "student_code": student_code,
                    "category": category,
                    "reward": reward,
                })

                all_snapshots[gen].append({
                    "problem_idx": prob_idx,
                    "category": category,
                    "reward": reward,
                    "ventiles": kl_result["ventiles"],
                    "mean_kl": (
                        kl_result["kl"].mean().item()
                        if kl_result["seq_len"] > 0
                        else float("nan")
                    ),
                })

            n_correct = sum(
                1 for r in gen_rollout_data if r["category"] == "correct"
            )
            mean_kl = sum(
                s["mean_kl"] for s in all_snapshots[gen][-num_rollouts:]
            ) / num_rollouts
            print(
                f"  Gen {gen}: {n_correct}/{num_rollouts} correct, "
                f"avg_kl={mean_kl:.4f}",
                end="",
            )

            # ----- Inner training steps -----
            for inner in range(num_inner_steps):
                loss = sdpo_training_step(
                    model, teacher, optimizer, tokenizer,
                    gen_rollout_data, hparams,
                )
            print(f"  loss={loss:.6f}")

        # ----- Final measurement after last training round -----
        model.eval()
        final_rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        n_correct_final = 0
        for rollout in final_rollouts:
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(fb)
            category = "correct" if reward == 1.0 else "incorrect"
            student_code = extract_python_code(rollout.completion)
            if category == "correct":
                n_correct_final += 1

            kl_result = compute_standard_kl(
                model, tokenizer,
                question=question,
                completion=rollout.completion,
                feedback_text=fb.feedback_text,
                student_code=student_code,
                top_k=top_k,
            )

            all_snapshots[num_outer_steps].append({
                "problem_idx": prob_idx,
                "category": category,
                "reward": reward,
                "ventiles": kl_result["ventiles"],
                "mean_kl": (
                    kl_result["kl"].mean().item()
                    if kl_result["seq_len"] > 0
                    else float("nan")
                ),
            })

        mean_kl_final = sum(
            s["mean_kl"] for s in all_snapshots[num_outer_steps][-num_rollouts:]
        ) / num_rollouts
        print(
            f"  Gen {num_outer_steps} (final): "
            f"{n_correct_final}/{num_rollouts} correct, "
            f"avg_kl={mean_kl_final:.4f}"
        )
        print()

    # -------------------------------------------------------------------
    # Aggregate per generation across all problems/rollouts
    # -------------------------------------------------------------------
    total_counts = {
        "correct": sum(1 for s in all_snapshots[0] if s["category"] == "correct"),
        "incorrect": sum(1 for s in all_snapshots[0] if s["category"] == "incorrect"),
    }

    num_gens = num_outer_steps + 1  # includes final measurement
    gens_data: List[Dict[str, Any]] = []

    for gen in range(num_gens):
        gen_record: Dict[str, Any] = {"generation": gen}

        for stratum in STRATA:
            if stratum == "all":
                subset = all_snapshots[gen]
            else:
                subset = [s for s in all_snapshots[gen] if s["category"] == stratum]

            if not subset:
                gen_record[stratum] = {
                    "count": 0,
                    "mean_ventiles": [float("nan")] * NUM_VENTILES,
                    "stderr_ventiles": [float("nan")] * NUM_VENTILES,
                    "peak_ventile": -1,
                    "mean_kl": float("nan"),
                    "mean_reward": float("nan"),
                }
                continue

            raw_vs = [s["ventiles"] for s in subset]
            mean_v, stderr_v = aggregate_ventiles(raw_vs)
            peak_ventile = max(range(NUM_VENTILES), key=lambda i: mean_v[i])
            mean_kl = sum(s["mean_kl"] for s in subset) / len(subset)
            mean_reward = sum(s["reward"] for s in subset) / len(subset)

            gen_record[stratum] = {
                "count": len(subset),
                "mean_ventiles": mean_v,
                "stderr_ventiles": stderr_v,
                "peak_ventile": peak_ventile,
                "mean_kl": mean_kl,
                "mean_reward": mean_reward,
            }

        gens_data.append(gen_record)

    # -------------------------------------------------------------------
    # Compute percentage-change metrics from generation 0 baseline
    # -------------------------------------------------------------------
    for stratum in STRATA:
        if gens_data[0][stratum]["count"] == 0:
            continue

        baseline = gens_data[0][stratum]["mean_ventiles"]
        baseline_kl = gens_data[0][stratum]["mean_kl"]

        for i, rec in enumerate(gens_data):
            s = rec[stratum]

            # Cumulative % change from gen-0 baseline
            s["cumulative_pct_ventiles"] = [
                ((s["mean_ventiles"][d] - baseline[d]) / baseline[d] * 100.0
                 if baseline[d] != 0 else float("nan"))
                for d in range(NUM_VENTILES)
            ]
            s["cumulative_pct_mean_kl"] = (
                (s["mean_kl"] - baseline_kl) / baseline_kl * 100.0
                if baseline_kl != 0 else float("nan")
            )

            # Step-over-step % change
            if i == 0:
                s["step_pct_ventiles"] = [0.0] * NUM_VENTILES
                s["step_pct_mean_kl"] = 0.0
            else:
                prev = gens_data[i - 1][stratum]
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
        if gens_data[0][stratum]["count"] == 0:
            continue
        print("=" * 80)
        print(f"SUMMARY — {stratum} (n={gens_data[0][stratum]['count']})")

        # Raw KL table
        print("-" * 80)
        print("Raw KL:")
        header = (f"{'Gen':<6}{'Reward':<8}"
                  + "".join(f"{'V' + str(d):<8}" for d in range(NUM_VENTILES))
                  + f"  {'Peak':<8}{'Mean':<8}")
        print(header)
        print("-" * len(header))
        for rec in gens_data:
            s = rec[stratum]
            peak_v = s["peak_ventile"]
            peak_label = f"V{peak_v}" if peak_v >= 0 else "N/A"
            line = (f"{rec['generation']:<6}{s['mean_reward']:<8.3f}"
                    + "".join(f"{v:<8.4f}" for v in s["mean_ventiles"])
                    + f"  {peak_label:<8}{s['mean_kl']:<8.4f}")
            print(line)

        # Cumulative % change table
        print()
        print("Cumulative % change from gen-0 baseline:")
        header = (f"{'Gen':<6}"
                  + "".join(f"{'V' + str(d):<8}" for d in range(NUM_VENTILES))
                  + f"  {'Mean':<8}")
        print(header)
        print("-" * len(header))
        for rec in gens_data:
            s = rec[stratum]
            line = (f"{rec['generation']:<6}"
                    + "".join(f"{v:<8.1f}" for v in s["cumulative_pct_ventiles"])
                    + f"  {s['cumulative_pct_mean_kl']:<8.1f}")
            print(line)

        # Gen-over-gen % change table
        print()
        print("Gen-over-gen % change:")
        print(header)
        print("-" * len(header))
        for rec in gens_data:
            s = rec[stratum]
            line = (f"{rec['generation']:<6}"
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
            "num_outer_steps": num_outer_steps,
            "num_inner_steps": num_inner_steps,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "learning_rate": learning_rate,
            "ema_decay": ema_decay,
            "timestamp": datetime.now().isoformat(),
        },
        "initial_strata_counts": total_counts,
        "generations": gens_data,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_3.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # -------------------------------------------------------------------
    # Plot: 3 rows (raw KL, cumulative %, gen-over-gen %) x strata cols
    # -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = [s for s in STRATA if gens_data[0][s]["count"] > 0]
        num_cols = len(plot_strata)
        row_configs = [
            ("mean_ventiles", "Standard KL (nats/token)", "stderr_ventiles"),
            ("cumulative_pct_ventiles", "Cumulative % change from gen 0", None),
            ("step_pct_ventiles", "Gen-over-gen % change", None),
        ]
        num_rows = len(row_configs)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows),
            squeeze=False,
        )
        gens_x = [rec["generation"] for rec in gens_data]
        cmap = plt.cm.viridis

        for row, (key, ylabel, stderr_key) in enumerate(row_configs):
            for col, stratum in enumerate(plot_strata):
                ax = axes[row][col]
                n = gens_data[0][stratum]["count"]

                for d in range(NUM_VENTILES):
                    values = [rec[stratum][key][d] for rec in gens_data]
                    color = cmap(d / (NUM_VENTILES - 1))
                    label = f"{d * 5}-{(d + 1) * 5}%"
                    ax.plot(gens_x, values, color=color, label=label, linewidth=1.5)

                    if stderr_key is not None:
                        stderrs = [rec[stratum][stderr_key][d] for rec in gens_data]
                        ax.fill_between(
                            gens_x,
                            [v - se for v, se in zip(values, stderrs)],
                            [v + se for v, se in zip(values, stderrs)],
                            color=color, alpha=0.15,
                        )

                # Add zero line for percentage-change rows
                if row > 0:
                    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

                ax.set_xlabel("Generation")
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
        description="Experiment 3: KL Shift Across Iterative SDPO Generations"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--num-outer-steps", type=int, default=5)
    parser.add_argument("--num-inner-steps", type=int, default=3)
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
        num_outer_steps=args.num_outer_steps,
        num_inner_steps=args.num_inner_steps,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
