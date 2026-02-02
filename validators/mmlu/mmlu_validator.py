import torch
import random
from typing import Dict, List
from datasets import Dataset, get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from validators.validator import Validator


class MMLUValidator(Validator):
    def __init__(self):
        super().__init__("mmlu")

        self.dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
        self.test_dataset = load_dataset(
            "cais/mmlu", "all", split="test").select(range(16))

        self.CHOICE_LABELS = ["A", "B", "C", "D"]

    def create_few_shot_prompt(self, subset: str, n_shots: int):
        subset_dataset = self.dev_dataset.filter(
            lambda x: x["subject"] == subset)

        shots = min(n_shots, len(subset_dataset))
        selected_indices = random.sample(range(len(subset_dataset)), shots)
        selected_examples = [subset_dataset[idx] for idx in selected_indices]
        return "\n\n".join([self.format_example(example, include_answer=True) for example in selected_examples])

    def format_example(self, example: Dict, include_answer: bool = False) -> str:
        prompt = f"Question: {example['question']}\n"
        for label, choice in zip(self.CHOICE_LABELS, example["choices"]):
            prompt += f"{label}. {choice}\n"

        prompt += "\nAnswer:"
        if include_answer:
            answer_label = self.CHOICE_LABELS[example["answer"]]
            prompt += f"{answer_label}"

        return prompt

    def build_prompt(self, question: Dict, subset: str, n_shots: int = 5) -> str:
        few_shot_prompt = self.create_few_shot_prompt(subset, n_shots)
        question_prompt = self.format_example(question, include_answer=False)
        return f"{few_shot_prompt}\n\n{question_prompt}"

    def validate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 1,  # unused for logit-based eval, kept for interface consistency
        max_seq_length: int = 2048,
    ) -> float:
        print("Validating MMLU...")

        model.eval()

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        choice_token_ids = [tokenizer.encode(
            label, add_special_tokens=False)[0] for label in self.CHOICE_LABELS]

        correct = 0
        total_questions = len(self.test_dataset)

        for i in range(0, total_questions, batch_size):
            batch_end = min(i + batch_size, total_questions)
            batch_indices = range(i, batch_end)
            batch_data = self.test_dataset.select(batch_indices)

            batch_prompts = []
            batch_answers = []
            for example in batch_data:
                example_subset = example["subject"]
                raw_prompt = self.build_prompt(example, example_subset)
                batch_prompts.append(raw_prompt)
                batch_answers.append(example["answer"])

            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            batch_inputs = {k: v.to(model.device)
                            for k, v in batch_inputs.items()}

            with torch.no_grad():
                outputs = model(**batch_inputs)
                logits = outputs.logits

            batch_rows = torch.arange(logits.size(0), device=logits.device)
            final_token_logits = logits[batch_rows, -1, :]
            choice_logits = final_token_logits[:, choice_token_ids]
            predictions = torch.argmax(choice_logits, dim=-1)

            for pred_idx, answer_idx in zip(predictions.tolist(), batch_answers):
                if pred_idx == answer_idx:
                    correct += 1

        accuracy = correct / total_questions
        return accuracy

