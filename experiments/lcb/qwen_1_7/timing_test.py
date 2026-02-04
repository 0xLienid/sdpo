"""
Timing test for a single SDPO training step.

Run with:
    uv run python -m experiments.lcb.qwen_1_7.timing_test
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from training.sdpo import SDPOHparams, compute_sdpo_loss_batched, EMATeacher
from data_modules.livecodebench import LiveCodeBenchDataset, livecodebench_rollout
from data_modules.livecodebench.feedback import (
    create_feedback_fn,
    create_batch_feedback_fn,
    get_feedback_batch_parallel,
)


def main():
    print("=" * 60)
    print("SDPO Single Step Timing Test")
    print("=" * 60)

    # Config
    model_name = "Qwen/Qwen3-1.7B"
    num_rollouts = 8
    rollout_temperature = 1.0

    # Load model
    print("\n[1] Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"    Model loaded in {time.time() - t0:.2f}s")

    # Create EMA teacher
    print("\n[2] Creating EMA teacher...")
    t0 = time.time()
    teacher = EMATeacher(model, decay=0.99)
    print(f"    Teacher created in {time.time() - t0:.2f}s")

    # Load dataset
    print("\n[3] Loading dataset...")
    t0 = time.time()
    dataset = LiveCodeBenchDataset(subset_size=10)
    example = dataset[0]
    print(f"    Dataset loaded in {time.time() - t0:.2f}s")

    # Setup hparams
    hparams = SDPOHparams(
        num_rollouts=num_rollouts,
        rollout_temperature=rollout_temperature,
        max_prompt_length=2048,
        max_response_length=2048,
        top_k_distillation=20,
    )

    # ===== ROLLOUT GENERATION =====
    print("\n[4] Generating rollouts...")
    print(f"    num_rollouts={num_rollouts}, temperature={rollout_temperature}")

    model.eval()
    # Disable gradient checkpointing for generation
    gc_was_enabled = model.is_gradient_checkpointing if hasattr(model, 'is_gradient_checkpointing') else False
    if gc_was_enabled:
        model.gradient_checkpointing_disable()

    t0 = time.time()
    rollouts = livecodebench_rollout(
        model, tokenizer, example,
        num_rollouts=num_rollouts,
        temperature=rollout_temperature,
        max_new_tokens=1024,
    )
    rollout_time = time.time() - t0

    if gc_was_enabled:
        model.gradient_checkpointing_enable()
    model.train()

    print(f"    Generated {len(rollouts)} rollouts in {rollout_time:.2f}s")
    print(f"    Avg completion length: {sum(len(r.completion) for r in rollouts) / len(rollouts):.0f} chars")

    # ===== FEEDBACK COLLECTION (SEQUENTIAL) =====
    print("\n[5] Collecting feedback (SEQUENTIAL)...")
    feedback_fn = create_feedback_fn(include_outside_feedback=False)

    t0 = time.time()
    feedback_results_seq = []
    for rollout in rollouts:
        result = feedback_fn(rollout.prompt, rollout.completion, example)
        feedback_results_seq.append(result)
    feedback_seq_time = time.time() - t0

    print(f"    Sequential feedback in {feedback_seq_time:.2f}s")
    print(f"    Successes: {sum(1 for r in feedback_results_seq if r.success)}/{len(feedback_results_seq)}")

    # ===== FEEDBACK COLLECTION (PARALLEL) =====
    print("\n[6] Collecting feedback (PARALLEL)...")

    prompts = [r.prompt for r in rollouts]
    completions = [r.completion for r in rollouts]
    examples = [example] * len(rollouts)

    t0 = time.time()
    feedback_results_par = get_feedback_batch_parallel(
        prompts, completions, examples,
        timeout_seconds=10,
        max_workers=8,
        parallel_tests_per_rollout=True,
    )
    feedback_par_time = time.time() - t0

    print(f"    Parallel feedback in {feedback_par_time:.2f}s")
    print(f"    Speedup: {feedback_seq_time / feedback_par_time:.2f}x")
    print(f"    Successes: {sum(1 for r in feedback_results_par if r.success)}/{len(feedback_results_par)}")

    # ===== LOSS COMPUTATION =====
    print("\n[7] Computing loss (batched)...")

    feedbacks = [r.feedback_text for r in feedback_results_par]
    prior_solutions = [None] * len(rollouts)

    t0 = time.time()
    loss, metrics = compute_sdpo_loss_batched(
        student_model=model,
        teacher_model=teacher.model,
        tokenizer=tokenizer,
        prompts=prompts,
        completions=completions,
        feedbacks=feedbacks,
        prior_solutions=prior_solutions,
        hparams=hparams,
    )
    loss_time = time.time() - t0

    print(f"    Loss computed in {loss_time:.2f}s")
    print(f"    Loss value: {loss.item():.4f}")
    print(f"    Completion tokens: {metrics['completion_tokens']}")

    # ===== BACKWARD PASS =====
    print("\n[8] Backward pass...")
    t0 = time.time()
    loss.backward()
    backward_time = time.time() - t0
    print(f"    Backward in {backward_time:.2f}s")

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("TIMING SUMMARY (single step)")
    print("=" * 60)
    print(f"  Rollout generation:     {rollout_time:>8.2f}s")
    print(f"  Feedback (sequential):  {feedback_seq_time:>8.2f}s")
    print(f"  Feedback (parallel):    {feedback_par_time:>8.2f}s  ({feedback_seq_time/feedback_par_time:.1f}x speedup)")
    print(f"  Loss computation:       {loss_time:>8.2f}s")
    print(f"  Backward pass:          {backward_time:>8.2f}s")
    print("-" * 60)
    total_seq = rollout_time + feedback_seq_time + loss_time + backward_time
    total_par = rollout_time + feedback_par_time + loss_time + backward_time
    print(f"  Total (sequential fb):  {total_seq:>8.2f}s")
    print(f"  Total (parallel fb):    {total_par:>8.2f}s  ({total_seq/total_par:.1f}x overall speedup)")
    print("=" * 60)


if __name__ == "__main__":
    main()
