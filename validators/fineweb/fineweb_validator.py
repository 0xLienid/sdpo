import random
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from validators.validator import Validator


class FinewebValidator(Validator):
    def __init__(self):
        super().__init__("fineweb")

        self.test_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    def validate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 0,  # unused for perplexity, kept for interface consistency
        max_seq_length: int = 1024,
        num_samples: int = 16,
    ) -> float:
        print("Validating FineWeb...")

        model.eval()

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        dataset_iter = iter(self.test_dataset)

        total_nll = 0.0
        total_token_count = 0

        processed_samples = 0
        while processed_samples < num_samples:
            batch = []
            for _ in range(batch_size):
                if processed_samples >= num_samples:
                    break

                try:
                    sample = next(dataset_iter)
                except StopIteration:
                    break

                batch.append(sample["text"])
                processed_samples += 1

            if len(batch) == 0:
                break

            batch_inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                padding_side="left",
            ).to(model.device)

            labels = batch_inputs["input_ids"].clone()
            if "attention_mask" in batch_inputs:
                labels[batch_inputs["attention_mask"] == 0] = -100

            with torch.no_grad():
                outputs = model(**batch_inputs)
                logits = outputs.logits.float()

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss_sum = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                valid_tokens = (shift_labels != -100).sum().item()
                if valid_tokens > 0:
                    total_nll += loss_sum.item()
                    total_token_count += valid_tokens

        if total_token_count == 0:
            return float("inf")

        average_nll = total_nll / total_token_count
        perplexity = math.exp(average_nll)
        return perplexity
