"""Shared utilities for SDPO analysis experiments."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_standard_completion_logits_completion_ids_and_mask(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_messages: List[str],
    assistant_messages: List[str],
    # max_seq_length: int = 10240,
) -> Tuple[torch.Tensor, torch.Tensor]:
    full_messages = [
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ] for user_message, assistant_message in zip(user_messages, assistant_messages)
    ]
    prompt_lengths = [len(tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False, add_generation_prompt=True,
        )
    ).input_ids) for user_message in user_messages]

    full_encodings = tokenizer.apply_chat_template(
        full_messages, tokenize=True, return_tensors="pt", padding=True, return_in_dict=True
    )
    completion_lengths = full_encodings["attention_mask"].sum(
        dim=-1) - torch.tensor(prompt_lengths)

    full_encodings = {k: v.to(model.device) for k, v in full_encodings.items()}
    with torch.no_grad():
        outputs = model(**full_encodings)

    max_completion_length = completion_lengths.max().item()
    logits = torch.zeros(
        (outputs.logits.shape[0], max_completion_length, outputs.logits.shape[-1]), device=model.device)
    completion_ids = torch.zeros((outputs.logits.shape[0], max_completion_length), dtype=torch.int64, device=model.device)
    assistant_mask = torch.zeros((outputs.logits.shape[0], max_completion_length), dtype=torch.bool, device=model.device)
    for i in range(len(user_messages)):
        end_idx = prompt_lengths[i] + completion_lengths[i] - 1
        logits[i, :completion_lengths[i], :] = outputs.logits[i,
                                                              prompt_lengths[i]-1:end_idx, :]
        completion_ids[i, :completion_lengths[i]] = full_encodings["input_ids"][i,
                                                                                 prompt_lengths[i]:end_idx+1].to(torch.int64)
        assistant_mask[i, :completion_lengths[i]] = full_encodings["attention_mask"][i,
                                                                                     prompt_lengths[i]-1:end_idx]

    return logits, completion_ids, assistant_mask


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


def compute_full_kl_per_position(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-position KL(student || teacher) using the full vocabulary.

    Same interface as compute_topk_kl_per_position but without top-K filtering.

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

    s_probs = F.softmax(s, dim=-1)
    s_log_probs = F.log_softmax(s, dim=-1)
    t_log_probs = F.log_softmax(t, dim=-1)

    kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    return kl_per_token


def bin_into_ventiles(values: torch.Tensor, mask: torch.Tensor) -> List[float]:
    """Bin a 3-dimensional tensor into 20 equal-width bins by relative position across dimension 1."""
    B, T = values.shape
    device = values.device
    dtype = values.dtype

    n = mask.sum(dim=1).to(torch.long)
    n_safe = n.clamp_min(1)

    pos = torch.arange(T, device=device, dtype=dtype)
    valid = pos < n_safe.unsqueeze(1)

    bins = ((pos * 20) // n_safe.unsqueeze(1)).clamp_max(19).to(torch.int64)

    sums = torch.zeros((B, 20), device=device, dtype=torch.float32)
    counts = torch.zeros((B, 20), device=device, dtype=dtype)

    sums.scatter_add_(1, bins, values * valid.to(dtype))
    counts.scatter_add_(1, bins, valid.to(dtype))

    means = sums / counts.clamp_min(1)
    means[counts == 0] = float("nan")
    return means.detach().cpu().numpy().tolist()
