"""
GRPO training for the non-causal correction head (Experiment 8).

Training loop:
1. Pre-compute base model hidden states and student top-k for all rollouts
2. For each epoch:
   a. For each problem: accumulate GRPO loss over its rollouts, step optimizer
   b. Periodically evaluate with greedy decoding
3. Save results and correction head state dict

The correction head is the only trainable component. The base model and lm_head
are completely frozen — gradients flow *through* lm_head to the correction head
but lm_head's weights are never updated.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.experiment_1_reward_on_regen import compute_reward
from analysis.experiment_8_correction_head.model import CorrectionHead
from analysis.utils import get_completion_logits
from data_modules.livecodebench.code_execution import extract_python_code
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from training.sdpo import build_teacher_messages


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PrecomputedRollout:
    """Pre-computed data for one rollout, reused across training epochs."""
    question: str
    completion: str
    feedback_text: str
    student_code: str
    category: str          # "correct" or "incorrect"
    reward: float          # original student reward
    hidden_states: torch.Tensor   # (seq_len, hidden_dim) post-norm, on GPU
    topk_indices: torch.Tensor    # (seq_len, top_k) on GPU
    example: Dict[str, Any]
    problem_idx: int
    rollout_idx: int


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def get_completion_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_messages: List[Dict[str, str]],
    completion: str,
    max_seq_length: int = 10240,
) -> torch.Tensor:
    """Get post-norm hidden states at completion-token positions.

    Calls model.model (the inner transformer) directly so we get the
    last_hidden_state *after* the final RMSNorm, ready for lm_head.

    Returns: (completion_len, hidden_dim)
    """
    full_messages = user_messages + [
        {"role": "assistant", "content": completion},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True,
    )

    full_enc = tokenizer(
        full_text, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(model.device)

    prompt_len = len(tokenizer(
        prompt_text, truncation=True, max_length=max_seq_length,
        padding=False,
    ).input_ids)

    with torch.no_grad():
        base_out = model.model(
            input_ids=full_enc.input_ids,
            attention_mask=full_enc.attention_mask,
        )
    hidden = base_out.last_hidden_state  # (1, seq_len, hidden_dim)

    seq_len = full_enc.input_ids.shape[1]
    # positions that predict completion tokens
    return hidden[0, prompt_len - 1 : seq_len - 1, :]


def precompute_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
) -> Dict[int, List[PrecomputedRollout]]:
    """Generate rollouts and pre-compute hidden states + top-k indices.

    Returns a dict mapping problem_idx → list of PrecomputedRollout.
    """
    by_problem: Dict[int, List[PrecomputedRollout]] = {}

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        question = example.get("question_content", example.get("question", ""))

        print(f"  Problem {prob_idx}/{len(dataset) - 1} ({title})")

        rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        problem_data: List[PrecomputedRollout] = []
        for r_idx, rollout in enumerate(rollouts):
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(fb)
            category = "correct" if reward == 1.0 else "incorrect"
            student_code = extract_python_code(rollout.completion)

            # Student logits → top-k
            student_msgs = [{"role": "user", "content": question}]
            student_logits, _ = get_completion_logits(
                model, tokenizer, student_msgs, rollout.completion,
            )
            if student_logits.shape[0] == 0:
                print(f"    Rollout {r_idx}: skipped (empty)")
                continue

            _, topk_idx = torch.topk(student_logits, top_k, dim=-1)

            # Teacher hidden states
            teacher_full = build_teacher_messages(
                prompt=question, completion=rollout.completion,
                feedback=fb.feedback_text, prior_solution=None,
                student_attempt=student_code,
            )
            teacher_msgs = [teacher_full[0]]

            hidden = get_completion_hidden_states(
                model, tokenizer, teacher_msgs, rollout.completion,
            )

            # Align lengths
            min_len = min(hidden.shape[0], topk_idx.shape[0])
            if min_len == 0:
                print(f"    Rollout {r_idx}: skipped (len mismatch)")
                continue

            problem_data.append(PrecomputedRollout(
                question=question,
                completion=rollout.completion,
                feedback_text=fb.feedback_text,
                student_code=student_code,
                category=category,
                reward=reward,
                hidden_states=hidden[:min_len].detach(),
                topk_indices=topk_idx[:min_len].detach(),
                example=example,
                problem_idx=prob_idx,
                rollout_idx=r_idx,
            ))

            print(
                f"    Rollout {r_idx}: {category} "
                f"(reward={reward:.2f}, len={min_len})"
            )

        by_problem[prob_idx] = problem_data

    return by_problem


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss_for_rollout(
    correction_head: CorrectionHead,
    lm_head: torch.nn.Module,
    rollout: PrecomputedRollout,
    tokenizer: AutoTokenizer,
    group_size: int = 8,
    temperature: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GRPO loss for one rollout.

    1. Forward correction head → corrections
    2. corrected_logits = lm_head(hidden + corrections), gather top-k
    3. Sample group_size sequences (independent per position)
    4. Decode and execute each → rewards
    5. Normalize advantages within the group
    6. Loss = -mean(advantage * sequence_log_prob)
    """
    device = rollout.hidden_states.device
    hidden = rollout.hidden_states        # (L, D)
    topk_idx = rollout.topk_indices       # (L, K)
    seq_len, top_k = topk_idx.shape

    # Forward correction head
    corrections = correction_head(hidden.unsqueeze(0)).squeeze(0)  # (L, D)
    corrected_hidden = hidden + corrections

    # Corrected logits at top-k positions only (keeps gradient path clean)
    corrected_logits = lm_head(corrected_hidden)               # (L, V)
    topk_logits = corrected_logits.gather(1, topk_idx).float() # (L, K) fp32

    # Probabilities over the top-k tokens
    topk_log_probs = F.log_softmax(topk_logits / temperature, dim=-1)  # (L, K)
    topk_probs = F.softmax(topk_logits / temperature, dim=-1)          # (L, K)

    # Sample group_size tokens per position from top-k distribution
    # multinomial: (L, group_size) — each row samples group_size indices in [0, K)
    sampled_k = torch.multinomial(topk_probs, group_size, replacement=True)  # (L, G)
    sampled_k = sampled_k.T  # (G, L)

    # Map back to vocab IDs
    pos_range = torch.arange(seq_len, device=device)
    sampled_vocab = topk_idx[pos_range.unsqueeze(0), sampled_k]  # (G, L)

    # Per-token log probs (differentiable)
    per_token_lp = topk_log_probs[pos_range.unsqueeze(0), sampled_k]  # (G, L)
    seq_log_probs = per_token_lp.sum(dim=-1)                          # (G,)

    # Decode and evaluate each group member
    rewards: List[float] = []
    for g in range(group_size):
        token_ids = sampled_vocab[g].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        try:
            fb = get_environment_feedback(
                prompt=rollout.question, completion=text,
                example=rollout.example,
            )
            r = compute_reward(fb)
        except Exception:
            r = 0.0
        rewards.append(r)

    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

    # GRPO: group-normalised advantages
    reward_std = rewards_t.std()
    if reward_std > 1e-8:
        advantages = (rewards_t - rewards_t.mean()) / (reward_std + 1e-8)
    else:
        advantages = torch.zeros_like(rewards_t)

    loss = -(advantages.detach() * seq_log_probs).mean()

    metrics = {
        "loss": loss.item(),
        "mean_reward": rewards_t.mean().item(),
        "max_reward": rewards_t.max().item(),
        "correction_norm": corrections.detach().norm(dim=-1).mean().item(),
        "reward_std": reward_std.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_greedy(
    correction_head: CorrectionHead,
    lm_head: torch.nn.Module,
    all_rollouts: List[PrecomputedRollout],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Evaluate with greedy decoding from corrected logits."""
    correction_head.eval()
    results: List[Dict[str, Any]] = []

    for rollout in all_rollouts:
        hidden = rollout.hidden_states
        topk_idx = rollout.topk_indices

        corrections = correction_head(hidden.unsqueeze(0)).squeeze(0)
        corrected_hidden = hidden + corrections
        corrected_logits = lm_head(corrected_hidden)

        # Mask to top-k
        topk_logits = corrected_logits.gather(1, topk_idx)
        masked = torch.full_like(corrected_logits, float("-inf"))
        masked.scatter_(1, topk_idx, topk_logits)

        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            masked[:, eos_id] = corrected_logits[:, eos_id]

        greedy_tokens = masked.argmax(dim=-1)

        # Truncate at EOS
        if eos_id is not None:
            eos_pos = (greedy_tokens == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                greedy_tokens = greedy_tokens[: eos_pos[0]]

        text = tokenizer.decode(greedy_tokens.tolist(), skip_special_tokens=True)

        try:
            fb = get_environment_feedback(
                prompt=rollout.question, completion=text,
                example=rollout.example,
            )
            corrected_reward = compute_reward(fb)
        except Exception:
            corrected_reward = 0.0

        results.append({
            "problem_idx": rollout.problem_idx,
            "rollout_idx": rollout.rollout_idx,
            "category": rollout.category,
            "original_reward": rollout.reward,
            "corrected_reward": corrected_reward,
            "correction_norm": corrections.norm(dim=-1).mean().item(),
        })

    correction_head.train()

    total = len(results)
    if total == 0:
        return {"total": 0}

    mean_orig = sum(r["original_reward"] for r in results) / total
    mean_corr = sum(r["corrected_reward"] for r in results) / total
    improved = sum(1 for r in results if r["corrected_reward"] > r["original_reward"])
    degraded = sum(1 for r in results if r["corrected_reward"] < r["original_reward"])

    return {
        "total": total,
        "mean_original_reward": mean_orig,
        "mean_corrected_reward": mean_corr,
        "improved": improved,
        "degraded": degraded,
        "same": total - improved - degraded,
        "per_rollout": results,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_8(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    top_k: int = 20,
    # Correction head
    head_num_layers: int = 2,
    head_num_heads: int = 16,
    head_ff_mult: int = 4,
    head_init_scale: float = 0.01,
    # GRPO
    group_size: int = 8,
    sampling_temperature: float = 0.3,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,
    num_epochs: int = 10,
    eval_every: int = 2,
    # Output
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:

    print("=" * 60)
    print("Experiment 8: Non-Causal Correction Head (GRPO)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts/problem: {num_rollouts}")
    print(f"Head: {head_num_layers} layers, {head_num_heads} heads")
    print(f"GRPO: group={group_size}, temp={sampling_temperature}")
    print(f"Training: lr={learning_rate}, epochs={num_epochs}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Load model (frozen) and tokenizer
    # ------------------------------------------------------------------
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Create correction head
    # ------------------------------------------------------------------
    hidden_dim = model.config.hidden_size
    head = CorrectionHead(
        hidden_dim=hidden_dim,
        num_layers=head_num_layers,
        num_heads=head_num_heads,
        ff_mult=head_ff_mult,
        init_scale=head_init_scale,
    ).to(device=model.device, dtype=torch.bfloat16)

    print(f"Correction head: {head.num_parameters():,} parameters")
    print()

    # ------------------------------------------------------------------
    # Pre-compute rollout data
    # ------------------------------------------------------------------
    print("Pre-computing rollout data...")
    by_problem = precompute_rollouts(
        model, tokenizer,
        dataset=LiveCodeBenchDataset(subset_size=num_problems),
        num_rollouts=num_rollouts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
    )

    all_rollouts = [r for rlist in by_problem.values() for r in rlist]
    n_total = len(all_rollouts)
    n_correct = sum(1 for r in all_rollouts if r.category == "correct")
    n_incorrect = n_total - n_correct
    print(f"\nPre-computed {n_total} rollouts "
          f"(correct={n_correct}, incorrect={n_incorrect})\n")

    if n_total == 0:
        print("No rollouts to train on. Exiting.")
        return {}

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=0.01,
    )

    lm_head = model.lm_head  # frozen Linear, used for forward pass only

    # ------------------------------------------------------------------
    # Initial evaluation (epoch 0)
    # ------------------------------------------------------------------
    print("Initial evaluation (before training)...")
    eval_result = evaluate_greedy(head, lm_head, all_rollouts, tokenizer)
    print(
        f"  Original reward: {eval_result['mean_original_reward']:.3f} | "
        f"Corrected reward: {eval_result['mean_corrected_reward']:.3f} | "
        f"Improved: {eval_result['improved']} / {eval_result['total']}"
    )
    print()

    eval_history: List[Dict[str, Any]] = [
        {"epoch": 0, **{k: v for k, v in eval_result.items() if k != "per_rollout"}},
    ]
    step_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    problem_indices = list(by_problem.keys())
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"--- Epoch {epoch}/{num_epochs} ---")
        random.shuffle(problem_indices)

        epoch_losses, epoch_rewards, epoch_norms = [], [], []

        for prob_idx in problem_indices:
            rollouts = by_problem[prob_idx]
            if not rollouts:
                continue

            head.train()
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_metrics: Dict[str, List[float]] = {
                "loss": [], "mean_reward": [], "max_reward": [],
                "correction_norm": [], "reward_std": [],
            }

            for rollout in rollouts:
                loss, metrics = grpo_loss_for_rollout(
                    head, lm_head, rollout, tokenizer,
                    group_size=group_size,
                    temperature=sampling_temperature,
                )
                (loss / len(rollouts)).backward()
                batch_loss += loss.item()
                for k, v in metrics.items():
                    batch_metrics[k].append(v)

            torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
            optimizer.step()
            global_step += 1

            avg = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
            epoch_losses.append(avg["loss"])
            epoch_rewards.append(avg["mean_reward"])
            epoch_norms.append(avg["correction_norm"])

            step_history.append({
                "epoch": epoch, "step": global_step,
                "problem_idx": prob_idx, **avg,
            })

        # Epoch summary
        if epoch_losses:
            print(
                f"  loss={sum(epoch_losses)/len(epoch_losses):.4f}  "
                f"reward={sum(epoch_rewards)/len(epoch_rewards):.3f}  "
                f"norm={sum(epoch_norms)/len(epoch_norms):.4f}"
            )

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == num_epochs:
            eval_result = evaluate_greedy(head, lm_head, all_rollouts, tokenizer)
            print(
                f"  [eval] Corrected reward: "
                f"{eval_result['mean_corrected_reward']:.3f} | "
                f"Improved: {eval_result['improved']} / {eval_result['total']}"
            )
            eval_history.append({
                "epoch": epoch,
                **{k: v for k, v in eval_result.items() if k != "per_rollout"},
            })

        print()

    # ------------------------------------------------------------------
    # Final evaluation with full details
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_eval = evaluate_greedy(head, lm_head, all_rollouts, tokenizer)

    print(f"  Total rollouts:      {final_eval['total']}")
    print(f"  Original reward:     {final_eval['mean_original_reward']:.3f}")
    print(f"  Corrected reward:    {final_eval['mean_corrected_reward']:.3f}")
    print(f"  Improved:            {final_eval['improved']}")
    print(f"  Degraded:            {final_eval['degraded']}")
    print(f"  Same:                {final_eval['same']}")
    print()

    for r in final_eval["per_rollout"]:
        delta = r["corrected_reward"] - r["original_reward"]
        marker = "+" if delta > 0 else ("-" if delta < 0 else " ")
        print(
            f"  P{r['problem_idx']}/R{r['rollout_idx']} "
            f"({r['category']:<9s}) "
            f"orig={r['original_reward']:.2f} → "
            f"corr={r['corrected_reward']:.2f} "
            f"[{marker}] norm={r['correction_norm']:.4f}"
        )
    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Correction head weights
    head_path = os.path.join(output_dir, "experiment_8_head.pt")
    torch.save(head.state_dict(), head_path)
    print(f"Head saved to: {head_path}")

    # Results JSON
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "head_num_layers": head_num_layers,
            "head_num_heads": head_num_heads,
            "head_ff_mult": head_ff_mult,
            "head_init_scale": head_init_scale,
            "head_parameters": head.num_parameters(),
            "group_size": group_size,
            "sampling_temperature": sampling_temperature,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "num_epochs": num_epochs,
            "hidden_dim": hidden_dim,
            "timestamp": datetime.now().isoformat(),
        },
        "total_rollouts": n_total,
        "eval_history": eval_history,
        "step_history": step_history,
        "final_eval": final_eval,
    }

    json_path = os.path.join(output_dir, "experiment_8.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 8: Non-Causal Correction Head (GRPO)"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=20)
    # Head architecture
    parser.add_argument("--head-num-layers", type=int, default=2)
    parser.add_argument("--head-num-heads", type=int, default=16)
    parser.add_argument("--head-ff-mult", type=int, default=4)
    parser.add_argument("--head-init-scale", type=float, default=0.01)
    # GRPO
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=2)
    # Output
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    run_experiment_8(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        head_num_layers=args.head_num_layers,
        head_num_heads=args.head_num_heads,
        head_ff_mult=args.head_ff_mult,
        head_init_scale=args.head_init_scale,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
    )
