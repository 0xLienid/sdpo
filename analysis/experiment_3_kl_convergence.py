"""
Experiment 3: KL Signal Propagation Across Iterative SDPO Training

Demonstrates the "saturation front" in SDPO training: the teacher's correction
signal is front-loaded (concentrated at early token positions), so the model
learns early tokens first.  As early positions saturate, the signal gradually
propagates to later positions.

For each problem independently, runs a generate-train loop:
  - Generate rollouts, get feedback
  - For each inner step: measure KL, then train
  - Regenerate and repeat

Key metrics tracked at every gradient step:
  - KL fraction per ventile: kl[v] / sum(kl) — where the signal is concentrated
  - Center of mass: weighted average position of the KL signal (0=start, 1=end)

The x-axis is total gradient steps, with vertical markers at regeneration points.
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
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
# Midpoint of each ventile as a fraction of total sequence length (0–1).
VENTILE_MIDPOINTS = [(d + 0.5) / NUM_VENTILES for d in range(NUM_VENTILES)]


def aggregate_values(
    values: List[float],
) -> Tuple[float, float]:
    """Compute mean and stderr for a list of scalars."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    stderr = math.sqrt(var / n) if n > 0 else 0.0
    return mean, stderr


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
        var = sum((v - means[i]) ** 2 for v in vals) / \
            (n - 1) if n > 1 else 0.0
        stderrs.append(math.sqrt(var / n) if n > 0 else 0.0)
    return means, stderrs


def compute_kl_fraction_and_com(
    ventiles: List[float],
) -> Tuple[List[float], float]:
    """Compute KL fraction per ventile and center of mass from ventile means.

    Returns:
        fraction: kl[v] / sum(kl) for each ventile (sums to 1.0).
        com: Weighted average position (0=sequence start, 1=sequence end).
    """
    total = sum(ventiles)
    if total <= 0:
        return [0.0] * NUM_VENTILES, float("nan")
    fraction = [v / total for v in ventiles]
    com = sum(VENTILE_MIDPOINTS[d] * ventiles[d]
              for d in range(NUM_VENTILES)) / total
    return fraction, com


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
    Compute per-position standard KL and teacher disagreement for one rollout.

    Standard KL = KL(student || teacher) where both evaluate the same completion,
    student is conditioned on problem only, teacher on problem + feedback.

    Teacher disagreement = fraction of tokens where the teacher assigns lower
    probability to the actual next token than the student does, i.e.
    log P_teacher(token) < log P_student(token).

    Returns dict with 'kl', 'ventiles', 'fraction', 'com', 'seq_len',
    and 'disagreement_ventiles' (list of 20 floats, each in [0, 1]).
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

    ventiles = bin_into_ventiles(kl.detach().cpu())
    fraction, com = compute_kl_fraction_and_com(ventiles)

    # --- Teacher disagreement per position ---
    # Get completion token IDs from the student-side tokenization.
    full_messages = student_msgs + [
        {"role": "assistant", "content": completion},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        student_msgs, tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(
        full_text, truncation=True, max_length=10240, padding=False,
    ).input_ids
    prompt_len_s = len(tokenizer(
        prompt_text, truncation=True, max_length=10240, padding=False,
    ).input_ids)
    completion_token_ids = torch.tensor(
        full_ids[prompt_len_s:], device=student_logits.device,
    )

    min_len = min(
        student_logits.shape[0],
        teacher_logits.shape[0],
        len(completion_token_ids),
    )

    if min_len > 0:
        s_log_probs = F.log_softmax(student_logits[:min_len], dim=-1)
        t_log_probs = F.log_softmax(teacher_logits[:min_len], dim=-1)
        token_ids = completion_token_ids[:min_len]
        idx = torch.arange(min_len, device=s_log_probs.device)

        s_token_logp = s_log_probs[idx, token_ids]
        t_token_logp = t_log_probs[idx, token_ids]

        # Disagreement: teacher assigns lower prob than student.
        disagreement = (t_token_logp - s_token_logp < -0.025).float()
        disagreement_ventiles = bin_into_ventiles(disagreement.detach().cpu())
    else:
        disagreement_ventiles = [float("nan")] * NUM_VENTILES

    return {
        "kl": kl.detach().cpu(),
        "ventiles": ventiles,
        "fraction": fraction,
        "com": com,
        "seq_len": len(kl),
        "disagreement_ventiles": disagreement_ventiles,
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


def measure_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rollout_data: List[Dict[str, Any]],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Measure KL for each rollout, returning per-rollout stats."""
    results = []
    for item in rollout_data:
        kl_result = compute_standard_kl(
            model, tokenizer,
            question=item["question"],
            completion=item["completion"],
            feedback_text=item["feedback_text"],
            student_code=item["student_code"],
            top_k=top_k,
        )
        results.append({
            "category": item["category"],
            "reward": item["reward"],
            "fraction": kl_result["fraction"],
            "com": kl_result["com"],
            "mean_kl": (
                kl_result["kl"].mean().item()
                if kl_result["seq_len"] > 0
                else float("nan")
            ),
            "disagreement_ventiles": kl_result["disagreement_ventiles"],
        })
    return results


def aggregate_measurements(
    per_rollout: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-rollout measurements into per-stratum summaries."""
    result: Dict[str, Dict[str, Any]] = {}
    for stratum in STRATA:
        if stratum == "all":
            subset = per_rollout
        else:
            subset = [r for r in per_rollout if r["category"] == stratum]

        if not subset:
            result[stratum] = {
                "count": 0,
                "kl_fraction_ventiles": [float("nan")] * NUM_VENTILES,
                "kl_fraction_stderr": [float("nan")] * NUM_VENTILES,
                "disagreement_ventiles": [float("nan")] * NUM_VENTILES,
                "disagreement_stderr": [float("nan")] * NUM_VENTILES,
                "center_of_mass": float("nan"),
                "center_of_mass_stderr": float("nan"),
                "mean_kl": float("nan"),
                "mean_reward": float("nan"),
            }
            continue

        frac_mean, frac_stderr = aggregate_ventiles(
            [r["fraction"] for r in subset]
        )
        disagree_mean, disagree_stderr = aggregate_ventiles(
            [r["disagreement_ventiles"] for r in subset]
        )
        com_mean, com_stderr = aggregate_values(
            [r["com"] for r in subset]
        )
        mean_kl = sum(r["mean_kl"] for r in subset) / len(subset)
        mean_reward = sum(r["reward"] for r in subset) / len(subset)

        result[stratum] = {
            "count": len(subset),
            "kl_fraction_ventiles": frac_mean,
            "kl_fraction_stderr": frac_stderr,
            "disagreement_ventiles": disagree_mean,
            "disagreement_stderr": disagree_stderr,
            "center_of_mass": com_mean,
            "center_of_mass_stderr": com_stderr,
            "mean_kl": mean_kl,
            "mean_reward": mean_reward,
        }
    return result


def run_experiment_3(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 15,
    num_rollouts: int = 4,
    num_outer_steps: int = 5,
    num_inner_steps: int = 5,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    learning_rate: float = 1e-6,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 3: KL Signal Propagation Across Iterative SDPO Training.

    For each problem independently:
      - Restore model to initial weights
      - For each generation (outer step):
        1. Generate fresh rollouts, get feedback
        2. For each inner step: measure KL fraction + COM, then train
      - Final measurement after last training round

    Measurements happen at every gradient step.  The x-axis is total gradient
    steps, with regeneration markers at generation boundaries.
    """

    total_steps = num_outer_steps * num_inner_steps

    print("=" * 60)
    print("Experiment 3: KL Signal Propagation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts/gen: {num_rollouts}")
    print(
        f"Generations: {num_outer_steps} | Inner steps/gen: {num_inner_steps}")
    print(f"Total gradient steps per problem: {total_steps}")
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
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
    # all_snapshots[step_idx] is a list of per-rollout measurement dicts
    # across all problems.  step_idx 0..total_steps (inclusive).
    num_measurement_points = total_steps + 1
    all_snapshots: List[List[Dict[str, Any]]] = [
        [] for _ in range(num_measurement_points)
    ]
    # Track which measurement indices are regeneration points.
    regeneration_indices = [
        g * num_inner_steps for g in range(num_outer_steps)]

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

        step_idx = 0

        for gen in range(num_outer_steps):
            # ----- Generate rollouts -----
            model.eval()
            rollouts = livecodebench_rollout(
                model, tokenizer, example,
                num_rollouts=num_rollouts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            # ----- Build rollout data with feedback -----
            gen_rollout_data: List[Dict[str, Any]] = []
            for rollout in rollouts:
                fb = get_environment_feedback(
                    prompt=rollout.prompt, completion=rollout.completion,
                    example=example,
                )
                reward = compute_reward(fb)
                category = "correct" if reward == 1.0 else "incorrect"
                gen_rollout_data.append({
                    "question": question,
                    "completion": rollout.completion,
                    "feedback_text": fb.feedback_text,
                    "student_code": extract_python_code(rollout.completion),
                    "category": category,
                    "reward": reward,
                })

            n_correct = sum(
                1 for r in gen_rollout_data if r["category"] == "correct"
            )
            print(f"  Gen {gen}: {n_correct}/{num_rollouts} correct")

            # ----- Inner loop: measure then train -----
            for inner in range(num_inner_steps):
                model.eval()
                measurements = measure_rollouts(
                    model, tokenizer, gen_rollout_data, top_k=top_k,
                )
                all_snapshots[step_idx].extend(measurements)

                avg_com = sum(m["com"]
                              for m in measurements) / len(measurements)
                avg_kl = sum(m["mean_kl"]
                             for m in measurements) / len(measurements)
                all_dis = [m["disagreement_ventiles"] for m in measurements]
                avg_dis = sum(
                    sum(d) / len(d) for d in all_dis
                ) / len(all_dis)
                print(
                    f"    Step {step_idx}: com={avg_com:.3f} "
                    f"avg_kl={avg_kl:.4f} dis={avg_dis:.1%}",
                    end="",
                )

                loss = sdpo_training_step(
                    model, teacher, optimizer, tokenizer,
                    gen_rollout_data, hparams,
                )
                print(f"  loss={loss:.6f}")
                step_idx += 1

        # ----- Final measurement after all training -----
        model.eval()
        final_rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        final_rollout_data: List[Dict[str, Any]] = []
        for rollout in final_rollouts:
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(fb)
            category = "correct" if reward == 1.0 else "incorrect"
            final_rollout_data.append({
                "question": question,
                "completion": rollout.completion,
                "feedback_text": fb.feedback_text,
                "student_code": extract_python_code(rollout.completion),
                "category": category,
                "reward": reward,
            })

        final_measurements = measure_rollouts(
            model, tokenizer, final_rollout_data, top_k=top_k,
        )
        all_snapshots[step_idx].extend(final_measurements)

        n_correct_final = sum(
            1 for r in final_rollout_data if r["category"] == "correct"
        )
        avg_com = sum(m["com"]
                      for m in final_measurements) / len(final_measurements)
        avg_kl = sum(m["mean_kl"]
                     for m in final_measurements) / len(final_measurements)
        print(
            f"    Step {step_idx} (final): {n_correct_final}/{num_rollouts} correct, "
            f"com={avg_com:.3f} avg_kl={avg_kl:.4f}"
        )
        print()

    # -------------------------------------------------------------------
    # Aggregate across all problems at each measurement point
    # -------------------------------------------------------------------
    measurements_data: List[Dict[str, Any]] = []

    for step_idx in range(num_measurement_points):
        gen = step_idx // num_inner_steps
        inner = step_idx % num_inner_steps
        is_regen = step_idx in regeneration_indices
        is_final = step_idx == total_steps

        record: Dict[str, Any] = {
            "step": step_idx,
            "generation": min(gen, num_outer_steps),
            "inner_step": inner if not is_final else 0,
            "is_regeneration": is_regen or is_final,
        }

        stratified = aggregate_measurements(all_snapshots[step_idx])
        record.update(stratified)
        measurements_data.append(record)

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    for stratum in STRATA:
        if measurements_data[0][stratum]["count"] == 0:
            continue
        print("=" * 70)
        print(f"SUMMARY — {stratum}")
        print("-" * 70)

        header = (
            f"{'Step':<6}{'Gen':<5}{'R':<3}"
            f"{'COM':<8}{'KL':<8}"
            f"{'Frac Q1':<9}{'Frac Q4':<9}"
            f"{'Dis Q1':<9}{'Dis Q2':<9}{'Dis Q3':<9}{'Dis Q4':<9}"
        )
        print(header)
        print("-" * len(header))

        for rec in measurements_data:
            s = rec[stratum]
            if s["count"] == 0:
                continue
            frac = s["kl_fraction_ventiles"]
            dis = s["disagreement_ventiles"]
            fq1 = sum(frac[0:5]) / 5
            fq4 = sum(frac[15:20]) / 5
            dq1 = sum(dis[0:5]) / 5
            dq2 = sum(dis[5:10]) / 5
            dq3 = sum(dis[10:15]) / 5
            dq4 = sum(dis[15:20]) / 5
            regen = "*" if rec["is_regeneration"] else ""
            line = (
                f"{rec['step']:<6}{rec['generation']:<5}{regen:<3}"
                f"{s['center_of_mass']:<8.3f}{s['mean_kl']:<8.4f}"
                f"{fq1:<9.3f}{fq4:<9.3f}"
                f"{dq1:<9.1%}{dq2:<9.1%}{dq3:<9.1%}{dq4:<9.1%}"
            )
            print(line)

    print("=" * 70)

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
            "total_gradient_steps": total_steps,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "learning_rate": learning_rate,
            "ema_decay": ema_decay,
            "timestamp": datetime.now().isoformat(),
        },
        "regeneration_indices": regeneration_indices,
        "measurements": measurements_data,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_3.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = [
            s for s in STRATA if measurements_data[0][s]["count"] > 0
        ]
        num_cols = len(plot_strata)
        fig, axes = plt.subplots(
            3, num_cols, figsize=(10 * num_cols, 18), squeeze=False,
        )
        steps_x = [rec["step"] for rec in measurements_data]
        cmap = plt.cm.viridis

        for col, stratum in enumerate(plot_strata):
            n = measurements_data[0][stratum]["count"]

            # --- Row 0: KL fraction per ventile ---
            ax = axes[0][col]
            for d in range(NUM_VENTILES):
                values = [
                    rec[stratum]["kl_fraction_ventiles"][d]
                    for rec in measurements_data
                ]
                stderrs = [
                    rec[stratum]["kl_fraction_stderr"][d]
                    for rec in measurements_data
                ]
                color = cmap(d / (NUM_VENTILES - 1))
                label = f"{d * 5}-{(d + 1) * 5}%"
                ax.plot(
                    steps_x, values, color=color, label=label, linewidth=1.5,
                )
                ax.fill_between(
                    steps_x,
                    [v - se for v, se in zip(values, stderrs)],
                    [v + se for v, se in zip(values, stderrs)],
                    color=color, alpha=0.1,
                )

            for ri in regeneration_indices:
                ax.axvline(x=ri, color="red", linestyle="--", alpha=0.4)
            # Final measurement regeneration marker
            ax.axvline(x=total_steps, color="red", linestyle="--", alpha=0.4)

            ax.set_xlabel("Gradient Step")
            ax.set_ylabel("KL Fraction")
            ax.set_title(f"KL fraction per ventile — {stratum} (n={n})")
            if col == num_cols - 1:
                ax.legend(
                    title="Position Ventile", bbox_to_anchor=(1.02, 1),
                    loc="upper left", fontsize="x-small", ncol=2,
                )

            # --- Row 1: Center of mass ---
            ax = axes[1][col]
            coms = [
                rec[stratum]["center_of_mass"] for rec in measurements_data
            ]
            com_stderrs = [
                rec[stratum]["center_of_mass_stderr"]
                for rec in measurements_data
            ]
            ax.plot(steps_x, coms, color="steelblue", linewidth=2)
            ax.fill_between(
                steps_x,
                [c - se for c, se in zip(coms, com_stderrs)],
                [c + se for c, se in zip(coms, com_stderrs)],
                color="steelblue", alpha=0.2,
            )

            for ri in regeneration_indices:
                ax.axvline(x=ri, color="red", linestyle="--", alpha=0.4)
            ax.axvline(x=total_steps, color="red", linestyle="--", alpha=0.4)

            ax.set_xlabel("Gradient Step")
            ax.set_ylabel("Center of Mass (0=start, 1=end)")
            ax.set_title(f"KL center of mass — {stratum} (n={n})")
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

            # --- Row 2: Teacher disagreement per ventile ---
            ax = axes[2][col]
            for d in range(NUM_VENTILES):
                values = [
                    rec[stratum]["disagreement_ventiles"][d]
                    for rec in measurements_data
                ]
                stderrs = [
                    rec[stratum]["disagreement_stderr"][d]
                    for rec in measurements_data
                ]
                color = cmap(d / (NUM_VENTILES - 1))
                label = f"{d * 5}-{(d + 1) * 5}%"
                ax.plot(
                    steps_x, values, color=color, label=label, linewidth=1.5,
                )
                ax.fill_between(
                    steps_x,
                    [v - se for v, se in zip(values, stderrs)],
                    [v + se for v, se in zip(values, stderrs)],
                    color=color, alpha=0.1,
                )

            for ri in regeneration_indices:
                ax.axvline(x=ri, color="red", linestyle="--", alpha=0.4)
            ax.axvline(x=total_steps, color="red", linestyle="--", alpha=0.4)

            ax.set_xlabel("Gradient Step")
            ax.set_ylabel("Disagreement Rate")
            ax.set_title(
                f"Teacher disagreement rate — {stratum} (n={n})"
            )
            ax.set_ylim(0, 1)
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
        description="Experiment 3: KL Signal Propagation Across SDPO Training"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--num-outer-steps", type=int, default=5)
    parser.add_argument("--num-inner-steps", type=int, default=5)
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
        num_outer_steps=args.num_outer_steps,
        num_inner_steps=args.num_inner_steps,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
