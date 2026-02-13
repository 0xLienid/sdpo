"""
Experiment 4: Structural Entropy Analysis

Tests whether the front-loaded KL profile in SDPO is explained by structural
properties of code generation (early tokens have higher entropy / more viable
alternatives) rather than prefix corruption.

Key question: Does per-position student entropy track the KL profile?
If the KL/entropy ratio is roughly constant across ventiles, the front-loading
is a natural consequence of positional entropy, not a training flaw.

Metrics per ventile:
  - Student entropy: H(student) = -sum(p * log p)
  - Teacher entropy: H(teacher)
  - KL(student || teacher): top-K filtered, matching SDPO
  - KL / entropy ratio: the key diagnostic
  - Disagreement rate: frac of tokens where P_teacher(token) < P_student(token)
  - Disagreement magnitude: mean |log P_t - log P_s| at disagreeing positions

No training loop — pure inference analysis on model-generated rollouts.
"""

import argparse
import json
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
    compute_topk_kl_per_position,
    get_completion_logits,
)
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import build_teacher_messages

NUM_VENTILES = 20
STRATA = ["all", "incorrect", "correct"]


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient, ignoring NaN pairs."""
    pairs = [
        (xi, yi) for xi, yi in zip(x, y)
        if not (math.isnan(xi) or math.isnan(yi))
    ]
    n = len(pairs)
    if n < 3:
        return float("nan")
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx = sum(xv) / n
    my = sum(yv) / n
    cov = sum((xv[i] - mx) * (yv[i] - my) for i in range(n)) / (n - 1)
    vx = sum((xv[i] - mx) ** 2 for i in range(n)) / (n - 1)
    vy = sum((yv[i] - my) ** 2 for i in range(n)) / (n - 1)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def disagree_magnitude_ventiles(
    logp_diff: torch.Tensor,
    disagreement: torch.Tensor,
) -> List[float]:
    """Mean |log P_teacher - log P_student| for disagreeing tokens per ventile."""
    n = len(logp_diff)
    if n == 0:
        return [float("nan")] * NUM_VENTILES
    result: List[float] = []
    for d in range(NUM_VENTILES):
        start = int(d * n / NUM_VENTILES)
        end = int((d + 1) * n / NUM_VENTILES)
        if start >= end:
            result.append(float("nan"))
            continue
        seg_diff = logp_diff[start:end]
        seg_dis = disagreement[start:end].bool()
        if seg_dis.any():
            # logp_diff is negative when disagreeing; magnitude = -logp_diff
            result.append((-seg_diff[seg_dis]).mean().item())
        else:
            result.append(0.0)
    return result


def analyze_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    feedback_text: str,
    student_code: str,
    top_k: int = 20,
) -> Optional[Dict[str, Any]]:
    """Compute all per-position metrics for one rollout.

    Returns None if the completion has zero tokens.
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

    min_len = min(student_logits.shape[0], teacher_logits.shape[0])
    if min_len == 0:
        return None

    s = student_logits[:min_len]
    t = teacher_logits[:min_len]

    # Full-vocab softmax for entropy and disagreement
    s_probs = F.softmax(s, dim=-1)
    s_log_probs = F.log_softmax(s, dim=-1)
    t_probs = F.softmax(t, dim=-1)
    t_log_probs = F.log_softmax(t, dim=-1)

    # Per-position entropy
    student_entropy = -(s_probs * s_log_probs).sum(dim=-1)
    teacher_entropy = -(t_probs * t_log_probs).sum(dim=-1)

    # Top-K KL (matching SDPO)
    kl = compute_topk_kl_per_position(s, t, top_k)

    # Completion token IDs for disagreement analysis
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
        full_ids[prompt_len_s:], device=s.device,
    )

    token_len = min(min_len, len(completion_token_ids))
    idx = torch.arange(token_len, device=s.device)
    token_ids = completion_token_ids[:token_len]

    s_token_logp = s_log_probs[idx, token_ids]
    t_token_logp = t_log_probs[idx, token_ids]

    logp_diff = t_token_logp - s_token_logp  # negative when teacher disagrees
    disagreement = (logp_diff < 0).float()

    # Bin into ventiles
    s_ent_v = bin_into_ventiles(student_entropy[:token_len].detach().cpu())
    t_ent_v = bin_into_ventiles(teacher_entropy[:token_len].detach().cpu())
    kl_v = bin_into_ventiles(kl[:token_len].detach().cpu())
    dis_rate_v = bin_into_ventiles(disagreement.detach().cpu())
    dis_mag_v = disagree_magnitude_ventiles(
        logp_diff.detach().cpu(), disagreement.detach().cpu(),
    )

    # KL / entropy ratio per ventile
    kl_ent_ratio_v = [
        kl_v[d] / s_ent_v[d] if s_ent_v[d] > 1e-8 else float("nan")
        for d in range(NUM_VENTILES)
    ]

    return {
        "student_entropy": s_ent_v,
        "teacher_entropy": t_ent_v,
        "kl": kl_v,
        "kl_entropy_ratio": kl_ent_ratio_v,
        "disagreement_rate": dis_rate_v,
        "disagreement_magnitude": dis_mag_v,
        "mean_kl": kl[:token_len].mean().item(),
        "mean_entropy": student_entropy[:token_len].mean().item(),
        "seq_len": token_len,
    }


def aggregate_ventile_lists(
    lists: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """Compute mean and stderr across ventile vectors, skipping NaNs."""
    if not lists:
        return [float("nan")] * NUM_VENTILES, [float("nan")] * NUM_VENTILES
    means: List[float] = []
    stderrs: List[float] = []
    for i in range(NUM_VENTILES):
        vals = [lst[i] for lst in lists if not math.isnan(lst[i])]
        if not vals:
            means.append(float("nan"))
            stderrs.append(float("nan"))
            continue
        m = sum(vals) / len(vals)
        means.append(m)
        if len(vals) > 1:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            stderrs.append(math.sqrt(var / len(vals)))
        else:
            stderrs.append(0.0)
    return means, stderrs


def run_experiment_4(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 15,
    num_rollouts: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 4: Structural Entropy Analysis.

    For each problem, generates rollouts, gets environment feedback, and
    computes per-position entropy, KL, disagreement rate, and disagreement
    magnitude.  The key output is whether the KL/entropy ratio is constant
    across ventiles (implying structural entropy explains the front-loaded KL).
    """

    print("=" * 60)
    print("Experiment 4: Structural Entropy Analysis")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts/problem: {num_rollouts}")
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

    # ---------------------------------------------------------------
    # Collect per-rollout results
    # ---------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        question = example.get("question_content", example.get("question", ""))

        print(f"Problem {prob_idx}/{len(dataset) - 1} ({title})")

        rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        for r_idx, rollout in enumerate(rollouts):
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(fb)
            category = "correct" if reward == 1.0 else "incorrect"
            student_code = extract_python_code(rollout.completion)

            result = analyze_rollout(
                model, tokenizer,
                question=question,
                completion=rollout.completion,
                feedback_text=fb.feedback_text,
                student_code=student_code,
                top_k=top_k,
            )
            if result is None:
                continue

            result["category"] = category
            result["reward"] = reward
            result["problem_idx"] = prob_idx
            result["rollout_idx"] = r_idx
            all_results.append(result)

            print(
                f"  Rollout {r_idx}: {category} "
                f"(reward={reward:.2f}) "
                f"entropy={result['mean_entropy']:.3f} "
                f"kl={result['mean_kl']:.4f}"
            )

    print(f"\nTotal rollouts analyzed: {len(all_results)}")
    n_correct = sum(1 for r in all_results if r["category"] == "correct")
    n_incorrect = sum(1 for r in all_results if r["category"] == "incorrect")
    print(f"  Correct: {n_correct} | Incorrect: {n_incorrect}\n")

    # ---------------------------------------------------------------
    # Aggregate per stratum
    # ---------------------------------------------------------------
    strata_data: Dict[str, Dict[str, Any]] = {}

    for stratum in STRATA:
        if stratum == "all":
            subset = all_results
        else:
            subset = [r for r in all_results if r["category"] == stratum]

        if not subset:
            strata_data[stratum] = {"count": 0}
            continue

        s_ent_mean, s_ent_se = aggregate_ventile_lists(
            [r["student_entropy"] for r in subset],
        )
        t_ent_mean, t_ent_se = aggregate_ventile_lists(
            [r["teacher_entropy"] for r in subset],
        )
        kl_mean, kl_se = aggregate_ventile_lists(
            [r["kl"] for r in subset],
        )
        ratio_mean, ratio_se = aggregate_ventile_lists(
            [r["kl_entropy_ratio"] for r in subset],
        )
        dis_rate_mean, dis_rate_se = aggregate_ventile_lists(
            [r["disagreement_rate"] for r in subset],
        )
        dis_mag_mean, dis_mag_se = aggregate_ventile_lists(
            [r["disagreement_magnitude"] for r in subset],
        )

        # Correlations across ventiles (on the means)
        corr_entropy_kl = pearson_correlation(s_ent_mean, kl_mean)
        corr_entropy_disrate = pearson_correlation(s_ent_mean, dis_rate_mean)
        corr_entropy_dismag = pearson_correlation(s_ent_mean, dis_mag_mean)

        strata_data[stratum] = {
            "count": len(subset),
            "student_entropy": s_ent_mean,
            "student_entropy_stderr": s_ent_se,
            "teacher_entropy": t_ent_mean,
            "teacher_entropy_stderr": t_ent_se,
            "kl": kl_mean,
            "kl_stderr": kl_se,
            "kl_entropy_ratio": ratio_mean,
            "kl_entropy_ratio_stderr": ratio_se,
            "disagreement_rate": dis_rate_mean,
            "disagreement_rate_stderr": dis_rate_se,
            "disagreement_magnitude": dis_mag_mean,
            "disagreement_magnitude_stderr": dis_mag_se,
            "corr_entropy_kl": corr_entropy_kl,
            "corr_entropy_disrate": corr_entropy_disrate,
            "corr_entropy_dismag": corr_entropy_dismag,
        }

    # ---------------------------------------------------------------
    # Print summary tables
    # ---------------------------------------------------------------
    for stratum in STRATA:
        sd = strata_data[stratum]
        if sd.get("count", 0) == 0:
            continue

        print("=" * 90)
        print(f"SUMMARY — {stratum} (n={sd['count']})")
        print("-" * 90)

        header = (
            f"{'Ventile':<10}{'S.Entropy':<11}{'T.Entropy':<11}"
            f"{'KL':<10}{'KL/Ent':<10}"
            f"{'DisRate':<10}{'DisMag':<10}"
        )
        print(header)
        print("-" * len(header))

        for d in range(NUM_VENTILES):
            pct = f"{d * 5}-{(d + 1) * 5}%"
            line = (
                f"{pct:<10}"
                f"{sd['student_entropy'][d]:<11.4f}"
                f"{sd['teacher_entropy'][d]:<11.4f}"
                f"{sd['kl'][d]:<10.4f}"
                f"{sd['kl_entropy_ratio'][d]:<10.4f}"
                f"{sd['disagreement_rate'][d]:<10.1%}"
                f"{sd['disagreement_magnitude'][d]:<10.4f}"
            )
            print(line)

        print()
        print(f"  Correlation (entropy vs KL):         r = {sd['corr_entropy_kl']:.4f}")
        print(f"  Correlation (entropy vs dis. rate):   r = {sd['corr_entropy_disrate']:.4f}")
        print(f"  Correlation (entropy vs dis. mag.):   r = {sd['corr_entropy_dismag']:.4f}")
        print()

    print("=" * 90)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
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
        "total_rollouts": len(all_results),
        "strata": strata_data,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_4.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = [s for s in STRATA if strata_data[s].get("count", 0) > 0]
        num_cols = len(plot_strata)
        fig, axes = plt.subplots(
            3, num_cols, figsize=(8 * num_cols, 15), squeeze=False,
        )

        ventile_x = [(d + 0.5) * 5 for d in range(NUM_VENTILES)]  # 2.5, 7.5, ...

        for col, stratum in enumerate(plot_strata):
            sd = strata_data[stratum]
            n = sd["count"]

            # --- Row 0: Entropy and KL profiles (normalized to sum=1) ---
            ax = axes[0][col]
            ent_vals = sd["student_entropy"]
            kl_vals = sd["kl"]
            ent_total = sum(v for v in ent_vals if not math.isnan(v))
            kl_total = sum(v for v in kl_vals if not math.isnan(v))
            if ent_total > 0 and kl_total > 0:
                ent_norm = [v / ent_total for v in ent_vals]
                kl_norm = [v / kl_total for v in kl_vals]
                ax.plot(
                    ventile_x, ent_norm, "o-", color="steelblue",
                    label="Student entropy (normalized)", linewidth=2,
                )
                ax.plot(
                    ventile_x, kl_norm, "s-", color="darkorange",
                    label="KL (normalized)", linewidth=2,
                )
            r = sd["corr_entropy_kl"]
            ax.set_xlabel("Sequence Position (%)")
            ax.set_ylabel("Fraction of Total")
            ax.set_title(
                f"Entropy vs KL profile — {stratum} (n={n}, r={r:.3f})"
            )
            ax.legend()

            # --- Row 1: KL / entropy ratio ---
            ax = axes[1][col]
            ratio_vals = sd["kl_entropy_ratio"]
            ratio_se = sd["kl_entropy_ratio_stderr"]
            ax.bar(
                ventile_x, ratio_vals, width=4, color="mediumpurple",
                yerr=ratio_se, capsize=2, alpha=0.8,
            )
            # Reference line at mean ratio
            valid_ratios = [v for v in ratio_vals if not math.isnan(v)]
            if valid_ratios:
                mean_ratio = sum(valid_ratios) / len(valid_ratios)
                ax.axhline(
                    y=mean_ratio, color="gray", linestyle="--", alpha=0.6,
                    label=f"Mean = {mean_ratio:.4f}",
                )
                ax.legend()
            ax.set_xlabel("Sequence Position (%)")
            ax.set_ylabel("KL / Student Entropy")
            ax.set_title(f"KL-to-entropy ratio — {stratum} (n={n})")

            # --- Row 2: Disagreement rate and magnitude ---
            ax = axes[2][col]
            ax.bar(
                [x - 1.5 for x in ventile_x],
                sd["disagreement_rate"], width=3,
                color="coral", alpha=0.7, label="Disagreement rate",
                yerr=sd["disagreement_rate_stderr"], capsize=1,
            )
            ax2 = ax.twinx()
            ax2.plot(
                ventile_x, sd["disagreement_magnitude"], "D-",
                color="darkgreen", linewidth=2, markersize=4,
                label="Disagreement magnitude",
            )
            ax2.set_ylabel("Mean |log P_t - log P_s|", color="darkgreen")
            ax.set_xlabel("Sequence Position (%)")
            ax.set_ylabel("Disagreement Rate", color="coral")
            ax.set_title(f"Disagreement — {stratum} (n={n})")
            ax.set_ylim(0, None)
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "experiment_4.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 4: Structural Entropy Analysis"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    run_experiment_4(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
