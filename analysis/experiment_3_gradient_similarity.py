import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from analysis.experiment_1_reward_on_regen import compute_reward


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(tensor1, tensor2, dim=-1)


def compute_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int = 20
) -> torch.Tensor:
    student_top_k_logits, student_top_k_indices = torch.topk(
        student_logits, k, dim=-1)
    teacher_logits_at_top_k_indices = torch.gather(
        teacher_logits, -1, student_top_k_indices)

    s_probs = F.softmax(student_top_k_logits, dim=-1)
    s_log_probs = F.log_softmax(student_top_k_logits, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits_at_top_k_indices, dim=-1)
    return (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1) * mask


def compute_gradient(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    student_messages: List[Dict[str, str]],
    teacher_messages: List[Dict[str, str]],
    top_k: int = 20,
    mode: str = "first_25",  # "first_25" or "last_25"
) -> torch.Tensor:
    model.train()

    student_prompt_lengths = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": student_messages[0]["content"]}],
        tokenize=True, add_generation_prompt=True,
    )["input_ids"])
    teacher_prompt_lengths = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": teacher_messages[0]["content"]}],
        tokenize=True, add_generation_prompt=True,
    )["input_ids"])

    student_full = tokenizer.apply_chat_template(
        student_messages, tokenize=True, add_generation_prompt=False, padding=True, return_tensors="pt", return_in_dict=True)
    teacher_full = tokenizer.apply_chat_template(
        teacher_messages, tokenize=True, add_generation_prompt=False, padding=True, return_tensors="pt", return_in_dict=True)
    completion_lengths = student_full["attention_mask"].sum(
        dim=-1) - torch.tensor(student_prompt_lengths)

    student_full = {k: v.to(model.device) for k, v in student_full.items()}
    outputs = model(**student_full)

    teacher_full = {k: v.to(model.device) for k, v in teacher_full.items()}
    with torch.no_grad():
        teacher_outputs = model(**teacher_full)

    end_idx = student_prompt_lengths + completion_lengths - 1
    teacher_end_idx = teacher_prompt_lengths + completion_lengths - 1
    student_logits = outputs.logits[:, student_prompt_lengths-1:end_idx, :]
    teacher_logits = teacher_outputs.logits[:,
                                            teacher_prompt_lengths-1:teacher_end_idx, :]

    mask = torch.zeros(
        (student_logits.shape[0], student_logits.shape[1]),
        dtype=torch.float32,
        device=student_logits.device,
    )

    if mode == "first_25":
        mask_end_idx = (completion_lengths // 4)
        mask[:, :mask_end_idx] = 1
    elif mode == "last_25":
        mask_start_idx = completion_lengths - (completion_lengths // 4)
        mask[:, mask_start_idx:completion_lengths] = 1
    else:
        raise ValueError(f"Invalid mode: {mode}")

    token_losses = compute_loss(
        student_logits, teacher_logits, mask, top_k)
    loss = token_losses.sum() / mask.sum().clamp(min=1.0)
    loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().clone().flatten())
    flat_grads = torch.cat(grads)

    model.zero_grad(set_to_none=True)
    model.eval()

    return flat_grads.cpu()


def run_experiment_3(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 15,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 8192,
    top_k: int = 20,
    output_dir: str = "analysis/results",
) -> Dict[str, Any]:
    def mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return sum(values) / len(values)

    def pairwise_cosine_similarities(grads: List[torch.Tensor]) -> List[float]:
        similarities: List[float] = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                similarities.append(cosine_similarity(
                    grads[i], grads[j]).item())
        return similarities

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

    print("Loading dataset...")
    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems\n")

    results = []
    for prob_idx in range(len(dataset)):
        example = dataset[prob_idx]
        title = example["question_title"]
        question = example["question_content"]
        print(f"Problem {prob_idx} ({title})...")

        print(f"Generating {num_rollouts} rollouts...")
        rollouts = livecodebench_rollout(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        rollout_records = []
        for r_idx, rollout in enumerate(rollouts):
            fb = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(fb)

            student_messages = [
                {"role": "user", "content": f"Answer the following question, please keep your reasoning concise, and put your code in a ```python{{code}}``` block:\n\n{question}"},
                {"role": "assistant", "content": rollout.completion},
            ]

            teacher_messages = [
                {"role": "user",
                    "content": f"## Question\n{question}\n\n## Previous Attempt\n{rollout.completion}\n\n## Feedback (from environment) for the previous attempt\n{fb.feedback_text}\nCorrectly solve the original question."},
                {"role": "assistant", "content": rollout.completion},
            ]

            first_25_grads = compute_gradient(
                model, tokenizer, student_messages, teacher_messages, mode="first_25")
            last_25_grads = compute_gradient(
                model, tokenizer, student_messages, teacher_messages, mode="last_25")

            rollout_records.append({
                "rollout_idx": r_idx,
                "correct": reward == 1.0,
                "first_25_grads": first_25_grads,
                "last_25_grads": last_25_grads,
            })

        correct_records = [r for r in rollout_records if r["correct"]]
        incorrect_records = [r for r in rollout_records if not r["correct"]]

        all_similarities = {
            "first_25": pairwise_cosine_similarities(
                [r["first_25_grads"] for r in rollout_records]
            ),
            "last_25": pairwise_cosine_similarities(
                [r["last_25_grads"] for r in rollout_records]
            ),
        }
        correct_similarities = {
            "first_25": pairwise_cosine_similarities(
                [r["first_25_grads"] for r in correct_records]
            ),
            "last_25": pairwise_cosine_similarities(
                [r["last_25_grads"] for r in correct_records]
            ),
        }
        incorrect_similarities = {
            "first_25": pairwise_cosine_similarities(
                [r["first_25_grads"] for r in incorrect_records]
            ),
            "last_25": pairwise_cosine_similarities(
                [r["last_25_grads"] for r in incorrect_records]
            ),
        }

        results.append({
            "problem_idx": prob_idx,
            "all_similarities": all_similarities,
            "correct_similarities": correct_similarities,
            "incorrect_similarities": incorrect_similarities,
        })

        print(f"Problem {prob_idx} summary")
        print(f"  Num correct: {len(correct_records)}/{len(rollout_records)}")
        print(f"  all_similarities: {all_similarities}")
        print(f"  correct_similarities: {correct_similarities}")
        print(f"  incorrect_similarities: {incorrect_similarities}")
        print()

    summary: Dict[str, Any] = {}
    print("=" * 70)
    for stratum_name in ["all_similarities", "correct_similarities", "incorrect_similarities"]:
        first_25_problem_means = [
            mean_or_nan(r[stratum_name]["first_25"])
            for r in results
            if r[stratum_name]["first_25"]
        ]
        last_25_problem_means = [
            mean_or_nan(r[stratum_name]["last_25"])
            for r in results
            if r[stratum_name]["last_25"]
        ]

        mean_first_25_across_problems = mean_or_nan(first_25_problem_means)
        mean_last_25_across_problems = mean_or_nan(last_25_problem_means)

        summary[stratum_name] = {
            "num_problems_with_pairs_first_25": len(first_25_problem_means),
            "num_problems_with_pairs_last_25": len(last_25_problem_means),
            "mean_first_25_across_problems": mean_first_25_across_problems,
            "mean_last_25_across_problems": mean_last_25_across_problems,
        }

        print(f"SUMMARY — {stratum_name} (across problems)")
        print(
            f"  Mean first 25 cosine similarity: {mean_first_25_across_problems:.4f}")
        print(
            f"  Mean last 25 cosine similarity: {mean_last_25_across_problems:.4f}")
        print()
        print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3: Cosine similarity of gradients"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-problems", type=int, default=15)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    run_experiment_3(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
