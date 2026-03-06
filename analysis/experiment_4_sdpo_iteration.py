import os
import json
import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
from data_modules.livecodebench.dataset import LiveCodeBenchDataset
from data_modules.livecodebench.feedback import get_environment_feedback
from data_modules.livecodebench.rollout import livecodebench_rollout
from analysis.experiment_1_reward_on_regen import compute_reward
from analysis.experiment_2_5_teacher_prompt_ablation import get_metrics


class EMATeacher:
    def __init__(
        self,
        student_model: AutoModelForCausalLM,
        decay: float = 0.99,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.model = copy.deepcopy(student_model)

        if device is not None:
            self.model = self.model.to(device)

        self.model.requires_grad_(False)
        self.model.eval()

        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def to(self, device: torch.device) -> "EMATeacher":
        self.model = self.model.to(device)
        return self

    @torch.no_grad()
    def update(self, student_model: AutoModelForCausalLM):
        student_params = dict(student_model.named_parameters())
        for name, teacher_param in self.model.named_parameters():
            if name in student_params:
                student_param = student_params[name]
                teacher_param.mul_(self.decay).add_(
                    student_param.data, alpha=1 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def to_jsonable(value: Any) -> Any:
    """Recursively convert tensors/containers to JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


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


def run_training_loop(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Any,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 8192,
    top_k: int = 20,
    num_iterations: int = 5
) -> List[Dict[str, Any]]:
    # initial_rollouts = livecodebench_rollout(
    #     model, tokenizer, example,
    #     num_rollouts=num_rollouts,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    # )
    # initial_feedbacks = [get_environment_feedback(
    #     prompt=rollout.prompt, completion=rollout.completion,
    #     example=example,
    # ) for rollout in initial_rollouts]
    # rollout_reward = sum([compute_reward(feedback)
    #                      for feedback in initial_feedbacks])

    # if rollout_reward > 0.0:
    #     print("Initial rollouts were successful, skipping training loop")
    #     return []

    print(
        f"Running training loop for problem {example['question_title']} for {num_iterations} iterations")
    original_state_dict = {
        name: tensor.detach().clone().cpu()
        for name, tensor in model.state_dict().items()
    }
    model.train()

    results = []

    for rollout_idx in range(num_rollouts):
        teacher = EMATeacher(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for iteration in range(num_iterations):
            print(f"Running iteration {iteration + 1} of {num_iterations}")

            rollout_records = []

            rollout = livecodebench_rollout(
                model, tokenizer, example,
                num_rollouts=1,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )[0]

            feedback = get_environment_feedback(
                prompt=rollout.prompt, completion=rollout.completion,
                example=example,
            )
            reward = compute_reward(feedback)

            student_messages = [
                {"role": "user",
                    "content": f"Answer the following question, please keep your reasoning concise, and put your code in a ```python{{code}}``` block:\n\n{example['question_content']}"},
                {"role": "assistant", "content": rollout.completion}
            ]
            teacher_messages = [
                {"role": "user",
                    "content": f"## Question\n{example['question_content']}\n\n## Previous Attempt\n{rollout.completion}\n\n## Feedback (from environment) for the previous attempt\n{feedback.feedback_text}\nCorrectly solve the original question."},
                {"role": "assistant", "content": rollout.completion}
            ]

            student_prompt_lengths = len(tokenizer.apply_chat_template(
                [student_messages[0]], tokenize=True, add_generation_prompt=True,
            )["input_ids"])
            teacher_prompt_lengths = len(tokenizer.apply_chat_template(
                [teacher_messages[0]], tokenize=True, add_generation_prompt=True,
            )["input_ids"])

            student_full = tokenizer.apply_chat_template(
                student_messages, tokenize=True, add_generation_prompt=False, padding=True, return_tensors="pt", return_in_dict=True)
            teacher_full = tokenizer.apply_chat_template(
                teacher_messages, tokenize=True, add_generation_prompt=False, padding=True, return_tensors="pt", return_in_dict=True)
            completion_length = student_full["input_ids"].shape[1] - \
                student_prompt_lengths

            student_full = {k: v.to(model.device)
                            for k, v in student_full.items()}
            outputs = model(**student_full)

            teacher_full = {k: v.to(model.device)
                            for k, v in teacher_full.items()}
            with torch.no_grad():
                teacher_outputs = model(**teacher_full)
            del teacher_full

            student_logits = outputs.logits[:, student_prompt_lengths -
                                            1:student_prompt_lengths + completion_length - 1, :]
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_lengths -
                                                    1:teacher_prompt_lengths + completion_length - 1, :]
            completion_ids = student_full["input_ids"][:,
                                                       student_prompt_lengths:student_prompt_lengths + completion_length]
            mask = torch.ones(1, completion_length, device=model.device)

            optimizer.zero_grad()
            token_losses = compute_loss(
                student_logits, teacher_logits, mask, top_k)
            loss = token_losses.sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

            teacher.update(model)

            metrics = get_metrics(
                student_logits, teacher_logits, completion_ids, mask, top_k)
            results.append({
                "rollout_idx": rollout_idx,
                "iteration": iteration,
                "correct": reward == 1.0,
                "metrics": metrics,
            })

        model.load_state_dict(original_state_dict)

    model.eval()
    return results


def run_experiment_4(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_problems: int = 4,
    num_rollouts: int = 4,
    temperature: float = 1.0,
    max_new_tokens: int = 8192,
    top_k: int = 20,
    num_iterations: int = 5,
    output_dir: str = "analysis/results/test",
) -> Dict[str, Any]:
    """Run Experiment 4: SDPO Iteration."""
    print("Running Experiment 4: SDPO Iteration")

    def aggregate_ventile_lists(lists: List[List[float]]) -> List[float]:
        if not lists:
            return [float("nan")] * 20
        means = []
        for i in range(20):
            vals = [lst[i] for lst in lists if not math.isnan(lst[i])]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        return means

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

    dataset = LiveCodeBenchDataset(subset_size=num_problems)
    print(f"Loaded {len(dataset)} problems\n")

    results = []
    for prob_idx in range(3, len(dataset)):
        example = dataset[prob_idx]
        title = example["question_title"]
        question = example["question_content"]
        print(f"Problem {prob_idx} ({title})...")

        metrics = run_training_loop(
            model, tokenizer, example,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            num_iterations=num_iterations,
        )

        if metrics != []:
            results.append({
                "problem_idx": prob_idx,
                "title": title,
                "question": question,
                "metrics": metrics,
            })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_problems", type=int, default=15)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default="analysis/results/test")
    args = parser.parse_args()

    results = run_experiment_4(
        model_name=args.model_name,
        num_problems=args.num_problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "experiment_4.json"), "w") as f:
        json.dump(to_jsonable(results), f, indent=2)
    print(f"Results saved to: {args.output_dir}")
