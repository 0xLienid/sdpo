"""Shared utilities for SDPO analysis experiments."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_completion_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_messages: List[Dict[str, str]],
    completion: str,
    max_seq_length: int = 10240,
) -> Tuple[torch.Tensor, int]:
    """
    Run a forward pass and return logits at each completion token position.

    Args:
        user_messages: The user turn(s) of the conversation (no assistant turn).
        completion: The assistant's response text.
        max_seq_length: Max tokens for the full sequence.

    Returns:
        logits: (completion_len, vocab_size) logits predicting each completion token.
        prompt_len: Number of prompt tokens.
    """
    full_messages = user_messages + [{"role": "assistant", "content": completion}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True,
    )

    full_encoding = tokenizer(
        full_text, return_tensors="pt", truncation=True,
        max_length=max_seq_length, padding=False,
    ).to(model.device)

    prompt_len = len(tokenizer(
        prompt_text, truncation=True, max_length=max_seq_length, padding=False,
    ).input_ids)

    with torch.no_grad():
        outputs = model(**full_encoding)

    seq_len = full_encoding.input_ids.shape[1]

    # Logits at position i predict token i+1.
    # Logits at [prompt_len-1 .. seq_len-2] predict completion tokens [0 .. completion_len-1].
    completion_logits = outputs.logits[0, prompt_len - 1: seq_len - 1, :]

    return completion_logits, prompt_len


def compute_topk_kl_per_position(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int = 20,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-position KL(student || teacher) using the student's top-K tokens.

    At each position, identifies the student's top-K tokens, gathers logits from
    both student and teacher at those indices, renormalizes via softmax, and
    computes KL divergence.  Matches the SDPO paper / codebase implementation.

    Args:
        student_logits: (student_len, vocab_size)
        teacher_logits: (teacher_len, vocab_size)

    Returns:
        kl_per_token: (min(student_len, teacher_len),)
    """
    min_len = min(student_logits.shape[0], teacher_logits.shape[0])
    if min_len == 0:
        return torch.tensor([], device=student_logits.device)

    s = student_logits[:min_len] / temperature
    t = teacher_logits[:min_len] / temperature

    _, topk_indices = torch.topk(s, top_k, dim=-1)
    s_topk = torch.gather(s, -1, topk_indices)
    t_topk = torch.gather(t, -1, topk_indices)

    s_probs = F.softmax(s_topk, dim=-1)
    s_log_probs = F.log_softmax(s_topk, dim=-1)
    t_log_probs = F.log_softmax(t_topk, dim=-1)

    kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    return kl_per_token


def bin_into_deciles(values: torch.Tensor) -> List[float]:
    """Bin a 1-D tensor into 10 equal-width bins by relative position."""
    n = len(values)
    if n == 0:
        return [float("nan")] * 10

    decile_means = []
    for d in range(10):
        start = int(d * n / 10)
        end = int((d + 1) * n / 10)
        if start < end:
            decile_means.append(values[start:end].mean().item())
        else:
            decile_means.append(float("nan"))
    return decile_means
