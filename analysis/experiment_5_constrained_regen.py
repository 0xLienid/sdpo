"""
Experiment 5: Constrained Regeneration

Tests whether constraining the teacher to generate from the student's top-k
token set at each position produces a flatter (less front-loaded) KL profile.

The teacher generates autoregressively from its own context (problem + feedback
+ student attempt), but at each ordinal position can only select from the
student's top-k tokens at that same position. This gives the teacher a
self-consistent prefix (unlike standard SDPO where it evaluates the student's
exact prefix) while staying within the student's probability mass (unlike free
regeneration which diverges too far).

Key metrics per ventile:
  - Standard KL: student vs standard teacher (evaluating student's completion)
  - Constrained KL: student vs constrained teacher (evaluating constrained completion)
  - Token overlap: fraction of positions where constrained and student tokens match
  - Reward: execution reward of constrained completion vs original

No training loop — pure inference analysis.
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
VENTILE_MIDPOINTS = [(d + 0.5) / NUM_VENTILES for d in range(NUM_VENTILES)]


def constrained_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    teacher_messages: List[Dict[str, str]],
    student_topk_indices: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[str, List[int]]:
    """Generate autoregressively, constrained to student's top-k per position.

    At each ordinal position t, the teacher can only emit one of the student's
    top-k tokens at that position (plus EOS). Generation stops at EOS or when
    all student positions are exhausted.

    Args:
        teacher_messages: Chat messages for the teacher context (no assistant turn).
        student_topk_indices: (completion_len, top_k) — allowed tokens per position.
        temperature: 0 for greedy, >0 for sampling from the masked distribution.

    Returns:
        completion_text: The decoded constrained completion.
        generated_ids: List of generated token IDs.
    """
    prompt_text = tokenizer.apply_chat_template(
        teacher_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True,
        max_length=8192, padding=False,
    ).to(model.device)

    max_positions = student_topk_indices.shape[0]
    generated_ids: List[int] = []
    past_key_values = None
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for t in range(max_positions):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            # Mask to student's top-k at position t, plus EOS
            allowed = student_topk_indices[t]
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits[allowed] = logits[allowed]
            if eos_id is not None:
                masked_logits[eos_id] = logits[eos_id]

            if temperature <= 0:
                next_token = masked_logits.argmax()
            else:
                probs = F.softmax(masked_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)[0]

            token_id = next_token.item()
            generated_ids.append(token_id)

            if token_id == eos_id:
                break

            input_ids = next_token.unsqueeze(0).unsqueeze(0)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype),
            ], dim=1)

    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion_text, generated_ids


def compute_kl_fraction_and_com(
    ventiles: List[float],
) -> Tuple[List[float], float]:
    """KL fraction per ventile and center of mass."""
    total = sum(v for v in ventiles if not math.isnan(v))
    if total <= 0:
        return [0.0] * NUM_VENTILES, float("nan")
    fraction = [v / total if not math.isnan(v) else 0.0 for v in ventiles]
    com = sum(
        VENTILE_MIDPOINTS[d] * ventiles[d]
        for d in range(NUM_VENTILES)
        if not math.isnan(ventiles[d])
    ) / total
    return fraction, com


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


def aggregate_scalars(values: List[float]) -> Tuple[float, float]:
    """Mean and stderr for a list of scalars."""
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n > 1:
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        return m, math.sqrt(var / n)
    return m, 0.0


def analyze_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    completion: str,
    feedback_text: str,
    student_code: str,
    example: Dict[str, Any],
    original_reward: float,
    top_k: int = 20,
    constrained_temperature: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Compute standard KL, constrained KL, overlap, and reward for one rollout."""

    student_msgs = [{"role": "user", "content": question}]

    # --- Student logits ---
    student_logits, _ = get_completion_logits(
        model, tokenizer, student_msgs, completion,
    )
    if student_logits.shape[0] == 0:
        return None

    # --- Student top-k indices for constraining ---
    _, topk_indices = torch.topk(student_logits, top_k, dim=-1)

    # --- Standard teacher logits (evaluating student's completion) ---
    teacher_full = build_teacher_messages(
        prompt=question, completion=completion,
        feedback=feedback_text, prior_solution=None,
        student_attempt=student_code,
    )
    teacher_msgs = [teacher_full[0]]

    standard_teacher_logits, _ = get_completion_logits(
        model, tokenizer, teacher_msgs, completion,
    )

    # --- Standard KL ---
    standard_kl = compute_topk_kl_per_position(
        student_logits, standard_teacher_logits, top_k,
    )
    standard_kl_v = bin_into_ventiles(standard_kl.detach().cpu())
    standard_frac, standard_com = compute_kl_fraction_and_com(standard_kl_v)

    # --- Constrained generation ---
    constrained_text, constrained_ids = constrained_generate(
        model, tokenizer,
        teacher_messages=teacher_msgs,
        student_topk_indices=topk_indices,
        temperature=constrained_temperature,
    )

    gen_len = len(constrained_ids)
    if gen_len == 0:
        return None

    # --- Constrained teacher logits (single forward pass on constrained completion) ---
    constrained_teacher_logits, _ = get_completion_logits(
        model, tokenizer, teacher_msgs, constrained_text,
    )

    # --- Constrained KL ---
    min_len = min(student_logits.shape[0], constrained_teacher_logits.shape[0])
    if min_len == 0:
        return None

    constrained_kl = compute_topk_kl_per_position(
        student_logits[:min_len], constrained_teacher_logits[:min_len], top_k,
    )
    constrained_kl_v = bin_into_ventiles(constrained_kl.detach().cpu())
    constrained_frac, constrained_com = compute_kl_fraction_and_com(
        constrained_kl_v,
    )

    # --- Token overlap ---
    # Get student's completion token IDs via re-tokenization (consistent with
    # how other experiments compute token-level metrics).
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
    student_token_ids = full_ids[prompt_len_s:]

    overlap_len = min(len(student_token_ids), gen_len)
    if overlap_len > 0:
        overlap = torch.tensor([
            1.0 if constrained_ids[i] == student_token_ids[i] else 0.0
            for i in range(overlap_len)
        ])
        overlap_ventiles = bin_into_ventiles(overlap)
    else:
        overlap_ventiles = [float("nan")] * NUM_VENTILES

    # --- Constrained completion reward ---
    fb_constrained = get_environment_feedback(
        prompt=question, completion=constrained_text, example=example,
    )
    constrained_reward = compute_reward(fb_constrained)

    return {
        "standard_kl": standard_kl_v,
        "constrained_kl": constrained_kl_v,
        "standard_kl_fraction": standard_frac,
        "constrained_kl_fraction": constrained_frac,
        "standard_com": standard_com,
        "constrained_com": constrained_com,
        "token_overlap": overlap_ventiles,
        "original_reward": original_reward,
        "constrained_reward": constrained_reward,
        "gen_len": gen_len,
        "student_len": len(student_token_ids),
        "mean_standard_kl": standard_kl.mean().item(),
        "mean_constrained_kl": constrained_kl[:min_len].mean().item(),
    }


def run_experiment_5(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    constrained_temperature: float = 0.0,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    """Run Experiment 5: Constrained Regeneration.

    For each problem, generates rollouts, then for each rollout:
      1. Computes student logits and extracts top-k per position
      2. Generates a constrained completion (teacher, restricted to student top-k)
      3. Computes standard KL and constrained KL profiles
      4. Measures token overlap and constrained completion reward
    """

    print("=" * 60)
    print("Experiment 5: Constrained Regeneration")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts/problem: {num_rollouts}")
    print(f"Top-K: {top_k} | Constrained temp: {constrained_temperature}")
    print("=" * 60)
    print()

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

    print("Loading dataset...")
    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems\n")

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
            original_reward = compute_reward(fb)
            category = "correct" if original_reward == 1.0 else "incorrect"
            student_code = extract_python_code(rollout.completion)

            print(f"  Rollout {r_idx} ({category}, reward={original_reward:.2f}): ", end="", flush=True)

            result = analyze_rollout(
                model, tokenizer,
                question=question,
                completion=rollout.completion,
                feedback_text=fb.feedback_text,
                student_code=student_code,
                example=example,
                original_reward=original_reward,
                top_k=top_k,
                constrained_temperature=constrained_temperature,
            )

            if result is None:
                print("skipped (empty)")
                continue

            result["category"] = category
            result["problem_idx"] = prob_idx
            result["rollout_idx"] = r_idx
            all_results.append(result)

            print(
                f"std_com={result['standard_com']:.3f} "
                f"con_com={result['constrained_com']:.3f} "
                f"overlap_q1={sum(result['token_overlap'][:5])/5:.1%} "
                f"overlap_q4={sum(result['token_overlap'][15:])/5:.1%} "
                f"con_reward={result['constrained_reward']:.2f}"
            )

    print(f"\nTotal rollouts analyzed: {len(all_results)}")
    n_correct = sum(1 for r in all_results if r["category"] == "correct")
    n_incorrect = sum(1 for r in all_results if r["category"] == "incorrect")
    print(f"  Correct: {n_correct} | Incorrect: {n_incorrect}\n")

    # -------------------------------------------------------------------
    # Aggregate per stratum
    # -------------------------------------------------------------------
    strata_data: Dict[str, Dict[str, Any]] = {}

    for stratum in STRATA:
        if stratum == "all":
            subset = all_results
        else:
            subset = [r for r in all_results if r["category"] == stratum]

        if not subset:
            strata_data[stratum] = {"count": 0}
            continue

        std_kl_mean, std_kl_se = aggregate_ventile_lists(
            [r["standard_kl"] for r in subset],
        )
        con_kl_mean, con_kl_se = aggregate_ventile_lists(
            [r["constrained_kl"] for r in subset],
        )
        std_frac_mean, std_frac_se = aggregate_ventile_lists(
            [r["standard_kl_fraction"] for r in subset],
        )
        con_frac_mean, con_frac_se = aggregate_ventile_lists(
            [r["constrained_kl_fraction"] for r in subset],
        )
        overlap_mean, overlap_se = aggregate_ventile_lists(
            [r["token_overlap"] for r in subset],
        )

        std_com_mean, std_com_se = aggregate_scalars(
            [r["standard_com"] for r in subset],
        )
        con_com_mean, con_com_se = aggregate_scalars(
            [r["constrained_com"] for r in subset],
        )
        orig_reward_mean, orig_reward_se = aggregate_scalars(
            [r["original_reward"] for r in subset],
        )
        con_reward_mean, con_reward_se = aggregate_scalars(
            [r["constrained_reward"] for r in subset],
        )
        mean_std_kl, _ = aggregate_scalars(
            [r["mean_standard_kl"] for r in subset],
        )
        mean_con_kl, _ = aggregate_scalars(
            [r["mean_constrained_kl"] for r in subset],
        )
        gen_len_mean, _ = aggregate_scalars(
            [float(r["gen_len"]) for r in subset],
        )
        student_len_mean, _ = aggregate_scalars(
            [float(r["student_len"]) for r in subset],
        )

        strata_data[stratum] = {
            "count": len(subset),
            "standard_kl": std_kl_mean,
            "standard_kl_stderr": std_kl_se,
            "constrained_kl": con_kl_mean,
            "constrained_kl_stderr": con_kl_se,
            "standard_kl_fraction": std_frac_mean,
            "standard_kl_fraction_stderr": std_frac_se,
            "constrained_kl_fraction": con_frac_mean,
            "constrained_kl_fraction_stderr": con_frac_se,
            "token_overlap": overlap_mean,
            "token_overlap_stderr": overlap_se,
            "standard_com": std_com_mean,
            "standard_com_stderr": std_com_se,
            "constrained_com": con_com_mean,
            "constrained_com_stderr": con_com_se,
            "original_reward": orig_reward_mean,
            "original_reward_stderr": orig_reward_se,
            "constrained_reward": con_reward_mean,
            "constrained_reward_stderr": con_reward_se,
            "mean_standard_kl": mean_std_kl,
            "mean_constrained_kl": mean_con_kl,
            "mean_gen_len": gen_len_mean,
            "mean_student_len": student_len_mean,
        }

    # -------------------------------------------------------------------
    # Print summary tables
    # -------------------------------------------------------------------
    for stratum in STRATA:
        sd = strata_data[stratum]
        if sd.get("count", 0) == 0:
            continue

        print("=" * 95)
        print(f"SUMMARY — {stratum} (n={sd['count']})")
        print(f"  Standard COM: {sd['standard_com']:.3f} | "
              f"Constrained COM: {sd['constrained_com']:.3f}")
        print(f"  Mean standard KL: {sd['mean_standard_kl']:.4f} | "
              f"Mean constrained KL: {sd['mean_constrained_kl']:.4f}")
        print(f"  Original reward: {sd['original_reward']:.3f} | "
              f"Constrained reward: {sd['constrained_reward']:.3f}")
        print(f"  Mean student len: {sd['mean_student_len']:.0f} | "
              f"Mean constrained len: {sd['mean_gen_len']:.0f}")
        print("-" * 95)

        header = (
            f"{'Ventile':<10}"
            f"{'Std KL':<10}{'Con KL':<10}"
            f"{'Std Frac':<10}{'Con Frac':<10}"
            f"{'Overlap':<10}"
        )
        print(header)
        print("-" * len(header))

        for d in range(NUM_VENTILES):
            pct = f"{d * 5}-{(d + 1) * 5}%"
            line = (
                f"{pct:<10}"
                f"{sd['standard_kl'][d]:<10.4f}"
                f"{sd['constrained_kl'][d]:<10.4f}"
                f"{sd['standard_kl_fraction'][d]:<10.3f}"
                f"{sd['constrained_kl_fraction'][d]:<10.3f}"
                f"{sd['token_overlap'][d]:<10.1%}"
            )
            print(line)

        print()

    print("=" * 95)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "constrained_temperature": constrained_temperature,
            "timestamp": datetime.now().isoformat(),
        },
        "total_rollouts": len(all_results),
        "strata": strata_data,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_5.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plot_strata = [
            s for s in STRATA if strata_data[s].get("count", 0) > 0
        ]
        num_cols = len(plot_strata)
        fig, axes = plt.subplots(
            3, num_cols, figsize=(8 * num_cols, 15), squeeze=False,
        )

        ventile_x = [(d + 0.5) * 5 for d in range(NUM_VENTILES)]

        for col, stratum in enumerate(plot_strata):
            sd = strata_data[stratum]
            n = sd["count"]

            # --- Row 0: Standard KL vs Constrained KL (raw) ---
            ax = axes[0][col]
            ax.plot(
                ventile_x, sd["standard_kl"], "o-", color="steelblue",
                label=f"Standard (COM={sd['standard_com']:.3f})",
                linewidth=2,
            )
            ax.fill_between(
                ventile_x,
                [v - se for v, se in zip(
                    sd["standard_kl"], sd["standard_kl_stderr"])],
                [v + se for v, se in zip(
                    sd["standard_kl"], sd["standard_kl_stderr"])],
                color="steelblue", alpha=0.15,
            )
            ax.plot(
                ventile_x, sd["constrained_kl"], "s-", color="darkorange",
                label=f"Constrained (COM={sd['constrained_com']:.3f})",
                linewidth=2,
            )
            ax.fill_between(
                ventile_x,
                [v - se for v, se in zip(
                    sd["constrained_kl"], sd["constrained_kl_stderr"])],
                [v + se for v, se in zip(
                    sd["constrained_kl"], sd["constrained_kl_stderr"])],
                color="darkorange", alpha=0.15,
            )
            ax.set_xlabel("Sequence Position (%)")
            ax.set_ylabel("KL(student || teacher)")
            ax.set_title(f"KL profiles — {stratum} (n={n})")
            ax.legend()

            # --- Row 1: KL fraction (normalized shape) ---
            ax = axes[1][col]
            ax.plot(
                ventile_x, sd["standard_kl_fraction"], "o-",
                color="steelblue", label="Standard fraction", linewidth=2,
            )
            ax.fill_between(
                ventile_x,
                [v - se for v, se in zip(
                    sd["standard_kl_fraction"],
                    sd["standard_kl_fraction_stderr"])],
                [v + se for v, se in zip(
                    sd["standard_kl_fraction"],
                    sd["standard_kl_fraction_stderr"])],
                color="steelblue", alpha=0.15,
            )
            ax.plot(
                ventile_x, sd["constrained_kl_fraction"], "s-",
                color="darkorange", label="Constrained fraction", linewidth=2,
            )
            ax.fill_between(
                ventile_x,
                [v - se for v, se in zip(
                    sd["constrained_kl_fraction"],
                    sd["constrained_kl_fraction_stderr"])],
                [v + se for v, se in zip(
                    sd["constrained_kl_fraction"],
                    sd["constrained_kl_fraction_stderr"])],
                color="darkorange", alpha=0.15,
            )
            ax.axhline(
                y=1.0 / NUM_VENTILES, color="gray", linestyle=":",
                alpha=0.5, label="Uniform",
            )
            ax.set_xlabel("Sequence Position (%)")
            ax.set_ylabel("KL Fraction")
            ax.set_title(f"KL fraction (normalized) — {stratum} (n={n})")
            ax.legend()

            # --- Row 2: Token overlap ---
            ax = axes[2][col]
            ax.bar(
                ventile_x, sd["token_overlap"], width=4,
                color="mediumpurple", alpha=0.8,
                yerr=sd["token_overlap_stderr"], capsize=2,
            )
            ax.set_xlabel("Sequence Position (%)")
            ax.set_ylabel("Token Overlap Rate")
            ax.set_title(f"Token overlap — {stratum} (n={n})")
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "experiment_5.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 5: Constrained Regeneration"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--constrained-temperature", type=float, default=0.0,
        help="Temperature for constrained generation (0=greedy)",
    )
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    run_experiment_5(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        constrained_temperature=args.constrained_temperature,
        output_dir=args.output_dir,
    )
