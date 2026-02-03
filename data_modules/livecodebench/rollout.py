"""
Rollout Function for LiveCodeBench

Generates code completions for LiveCodeBench problems.
"""

import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.sdpo import RolloutResult


def livecodebench_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    num_rollouts: int = 1,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> List[RolloutResult]:
    """
    Generate multiple code completions for a LiveCodeBench problem.

    Args:
        model: The language model
        tokenizer: The tokenizer
        example: Dict containing at minimum 'question_content'
        num_rollouts: Number of completions to generate
        temperature: Sampling temperature (>0 for sampling, 0 for greedy)
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of RolloutResults with prompts and generated completions
    """
    question = example.get("question_content", example.get("question", ""))

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding=False,
    ).to(model.device)

    prompt_ids = inputs.input_ids.clone()
    prompt_len = inputs.input_ids.shape[1]

    # Generate multiple rollouts
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

    # Extract completions, stripping pad tokens
    pad_token_id = tokenizer.pad_token_id
    results = []
    for i in range(num_rollouts):
        completion_ids = outputs[i, prompt_len:]

        # Strip pad tokens from the end (generated sequences are right-padded)
        if pad_token_id is not None:
            # Find first pad token or use full length
            pad_mask = completion_ids == pad_token_id
            if pad_mask.any():
                first_pad = pad_mask.nonzero(as_tuple=True)[0][0].item()
                completion_ids = completion_ids[:first_pad]

        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

        results.append(RolloutResult(
            prompt=question,
            completion=completion,
            prompt_ids=prompt_ids[0],
            completion_ids=completion_ids,
        ))

    tokenizer.padding_side = original_padding_side
    return results


def batch_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list,
    max_new_tokens: int = 2048,
    batch_size: int = 4,
) -> list:
    """
    Generate completions for multiple examples in batches.

    Args:
        model: The language model
        tokenizer: The tokenizer
        examples: List of example dicts
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation

    Returns:
        List of RolloutResults
    """
    results = []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i:i + batch_size]

        # Build prompts
        prompts = []
        questions = []
        for example in batch_examples:
            question = example.get("question_content", example.get("question", ""))
            questions.append(question)
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(prompt)

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True,
        ).to(model.device)

        prompt_lengths = [
            (inputs.attention_mask[j] == 1).sum().item()
            for j in range(len(prompts))
        ]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        # Extract completions, stripping pad tokens
        pad_token_id = tokenizer.pad_token_id
        for j, (output, question, prompt_len) in enumerate(zip(outputs, questions, prompt_lengths)):
            completion_ids = output[prompt_len:]

            # Strip pad tokens from the end
            if pad_token_id is not None:
                pad_mask = completion_ids == pad_token_id
                if pad_mask.any():
                    first_pad = pad_mask.nonzero(as_tuple=True)[0][0].item()
                    completion_ids = completion_ids[:first_pad]

            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            results.append(RolloutResult(
                prompt=question,
                completion=completion,
                prompt_ids=inputs.input_ids[j, :prompt_len],
                completion_ids=completion_ids,
            ))

    tokenizer.padding_side = original_padding_side
    return results
