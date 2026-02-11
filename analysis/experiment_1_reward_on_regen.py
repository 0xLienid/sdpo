"""
Experiment 1: Reward on Regeneration

Establishes that regenerated rollouts (with prior attempt + execution feedback
in context) produce better solutions than original rollouts, validating that
the model's in-context self-correction is effective.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from data_modules.livecodebench.code_execution import extract_python_code
from training.sdpo import FeedbackResult, RolloutResult, build_teacher_regen_prompt

logger = logging.getLogger(__name__)


def generate_from_messages(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    num_rollouts: int = 1,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> List[RolloutResult]:
    """
    Generate completions from an arbitrary chat message list.

    Mirrors the generation logic in livecodebench_rollout but accepts
    pre-built messages instead of constructing them from an example dict.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding=False,
    ).to(model.device)

    prompt_ids = inputs.input_ids.clone()
    prompt_len = inputs.input_ids.shape[1]

    do_sample = temperature > 0 and num_rollouts > 1

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
            "num_return_sequences": num_rollouts,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95

        outputs = model.generate(**inputs, **generation_kwargs)

    pad_token_id = tokenizer.pad_token_id
    results = []
    for i in range(num_rollouts):
        completion_ids = outputs[i, prompt_len:]

        if pad_token_id is not None:
            pad_mask = completion_ids == pad_token_id
            if pad_mask.any():
                first_pad = pad_mask.nonzero(as_tuple=True)[0][0].item()
                completion_ids = completion_ids[:first_pad]

        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Store the user message content as the prompt (not the full chat template)
        user_content = messages[0]["content"] if messages else ""
        results.append(RolloutResult(
            prompt=user_content,
            completion=completion,
            prompt_ids=prompt_ids[0],
            completion_ids=completion_ids,
        ))

    tokenizer.padding_side = original_padding_side
    return results


def generate_from_message_batches(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message_batches: List[List[Dict[str, str]]],
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> List[RolloutResult]:
    """
    Generate one completion per message batch in a single parallel model call.

    This is used to process re-generation prompts concurrently for a problem.
    """
    if not message_batches:
        return []

    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        for messages in message_batches
    ]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    prompt_tokenized = tokenizer(
        prompt_texts,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding=True,
    ).to(model.device)

    # Keep per-sample prompt token ids (without left padding) for parity.
    prompt_ids_per_sample = []
    for text in prompt_texts:
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=False,
        ).to(model.device)
        prompt_ids_per_sample.append(tokenized.input_ids[0].clone())

    prompt_len = prompt_tokenized.input_ids.shape[1]

    # Keep behavior aligned with generate_from_messages(num_rollouts=1): greedy decode.
    do_sample = False

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
            "num_return_sequences": 1,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95

        outputs = model.generate(**prompt_tokenized, **generation_kwargs)

    pad_token_id = tokenizer.pad_token_id
    results: List[RolloutResult] = []
    for i, messages in enumerate(message_batches):
        completion_ids = outputs[i, prompt_len:]

        if pad_token_id is not None:
            pad_mask = completion_ids == pad_token_id
            if pad_mask.any():
                first_pad = pad_mask.nonzero(as_tuple=True)[0][0].item()
                completion_ids = completion_ids[:first_pad]

        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        user_content = messages[0]["content"] if messages else ""
        results.append(RolloutResult(
            prompt=user_content,
            completion=completion,
            prompt_ids=prompt_ids_per_sample[i],
            completion_ids=completion_ids,
        ))

    tokenizer.padding_side = original_padding_side
    return results


def compute_reward(feedback_result: FeedbackResult) -> float:
    """Extract fractional pass rate from a FeedbackResult."""
    metadata = feedback_result.metadata or {}
    total = metadata.get("total_count", 0)
    if total == 0:
        return 0.0
    return metadata.get("passed_count", 0) / total


def run_experiment_1(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 10,
    num_rollouts: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    output_path: str = "analysis/results/experiment_1.json",
) -> Dict[str, Any]:
    """Run Experiment 1: Reward on Regeneration."""

    print("=" * 60)
    print("Experiment 1: Reward on Regeneration")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Problems: {num_problems} | Rollouts per problem: {num_rollouts}")
    print("=" * 60)
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems")
    print()

    # Run experiment
    all_problem_results = []
    all_original_rewards = []
    all_regen_rewards = []

    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example.get("question_title", f"Problem {prob_idx}")
        print(f"Problem {prob_idx} ({title})...")

        # Generate original rollouts
        print(f"  Generating {num_rollouts} original rollouts...")
        original_rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Collect feedback and prepare regen prompts
        rollout_details = []
        problem_original_rewards = []
        problem_regen_rewards = []
        regen_inputs = []

        for r_idx, rollout in enumerate(original_rollouts):
            # Get feedback on original rollout
            orig_fb = get_environment_feedback(
                prompt=rollout.prompt,
                completion=rollout.completion,
                example=example,
            )
            orig_reward = compute_reward(orig_fb)
            problem_original_rewards.append(orig_reward)

            # Build regen prompt with prior attempt + feedback
            regen_messages = build_teacher_regen_prompt(
                prompt=rollout.prompt,
                feedback=orig_fb.feedback_text,
                student_attempt=extract_python_code(rollout.completion),
            )

            print(f"  Rollout {r_idx}: original reward={orig_reward:.3f}, queued regen prompt")
            regen_inputs.append({
                "rollout": rollout,
                "orig_fb": orig_fb,
                "orig_reward": orig_reward,
                "regen_messages": regen_messages,
            })

        # Generate all regen rollouts in parallel (single batched model call).
        print(f"  Generating {len(regen_inputs)} regen rollouts in parallel...")
        regen_rollouts = generate_from_message_batches(
            model=model,
            tokenizer=tokenizer,
            message_batches=[item["regen_messages"] for item in regen_inputs],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        for r_idx, (item, regen_rollout) in enumerate(zip(regen_inputs, regen_rollouts)):
            rollout = item["rollout"]
            orig_fb = item["orig_fb"]
            orig_reward = item["orig_reward"]

            # Get feedback on regen rollout
            regen_fb = get_environment_feedback(
                prompt=rollout.prompt,
                completion=regen_rollout.completion,
                example=example,
            )
            regen_reward = compute_reward(regen_fb)
            problem_regen_rewards.append(regen_reward)

            rollout_details.append({
                "original_completion": rollout.completion,
                "original_feedback": orig_fb.feedback_text,
                "original_reward": orig_reward,
                "original_passed": orig_fb.metadata.get("passed_count", 0),
                "original_total": orig_fb.metadata.get("total_count", 0),
                "regen_completion": regen_rollout.completion,
                "regen_feedback": regen_fb.feedback_text,
                "regen_reward": regen_reward,
                "regen_passed": regen_fb.metadata.get("passed_count", 0),
                "regen_total": regen_fb.metadata.get("total_count", 0),
            })

        orig_mean = sum(problem_original_rewards) / len(problem_original_rewards)
        regen_mean = sum(problem_regen_rewards) / len(problem_regen_rewards)
        delta = regen_mean - orig_mean

        all_original_rewards.extend(problem_original_rewards)
        all_regen_rewards.extend(problem_regen_rewards)

        all_problem_results.append({
            "problem_idx": prob_idx,
            "question_title": title,
            "original_rewards": problem_original_rewards,
            "regen_rewards": problem_regen_rewards,
            "original_mean_reward": orig_mean,
            "regen_mean_reward": regen_mean,
            "delta": delta,
            "rollout_details": rollout_details,
        })

        print(f"  -> original={orig_mean:.3f}  regen={regen_mean:.3f}  delta={delta:+.3f}")
        print()

    # Summary statistics
    total_rollouts = len(all_original_rewards)
    num_improved = sum(
        1 for o, r in zip(all_original_rewards, all_regen_rewards) if r > o
    )
    num_same = sum(
        1 for o, r in zip(all_original_rewards, all_regen_rewards) if r == o
    )
    num_degraded = sum(
        1 for o, r in zip(all_original_rewards, all_regen_rewards) if r < o
    )

    grand_orig_mean = sum(all_original_rewards) / total_rollouts
    grand_regen_mean = sum(all_regen_rewards) / total_rollouts
    grand_delta = grand_regen_mean - grand_orig_mean

    summary = {
        "original_mean_reward": grand_orig_mean,
        "regen_mean_reward": grand_regen_mean,
        "delta": grand_delta,
        "num_improved": num_improved,
        "num_same": num_same,
        "num_degraded": num_degraded,
        "total_rollouts": total_rollouts,
    }

    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Original mean reward:     {grand_orig_mean:.3f}")
    print(f"  Regen mean reward:        {grand_regen_mean:.3f}")
    print(f"  Delta:                   {grand_delta:+.3f}")
    print()
    print(f"  Rollout-level comparison (n={total_rollouts}):")
    print(f"    Improved:  {num_improved} ({100 * num_improved / total_rollouts:.1f}%)")
    print(f"    Same:      {num_same} ({100 * num_same / total_rollouts:.1f}%)")
    print(f"    Degraded:  {num_degraded} ({100 * num_degraded / total_rollouts:.1f}%)")
    print("=" * 60)

    # Save results
    results = {
        "config": {
            "model_name": model_name,
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "problems": all_problem_results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Reward on Regeneration"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=25)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--output", type=str, default="analysis/results/experiment_1.json"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_experiment_1(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )
