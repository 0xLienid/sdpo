import torch
import dataclasses
import validators.ifeval.instructions_registry as instructions_registry
from typing import Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from validators.validator import Validator


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


class IFEvalValidator(Validator):
    def __init__(self):
        super().__init__("ifeval")

        self.test_dataset = load_dataset("google/IFEval", split="train").select(range(16))

    def is_correct(self, input_example: InputExample, completion: str) -> bool:
        instructions_list = input_example.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instructions_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            kwargs = {k: v for k,
                      v in input_example.kwargs[index].items() if v}
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=input_example.prompt)

            if completion.strip() and instruction.check_following(completion):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        return all(is_following_list)

    def validate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 256,
        max_seq_length: int = 2048,
    ) -> float:
        print("Validating IFEval...")

        model.eval()

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        correct = 0
        total_examples = len(self.test_dataset)

        for i in range(0, total_examples, batch_size):
            batch_end = min(i + batch_size, total_examples)
            batch_indices = range(i, batch_end)
            batch_data = self.test_dataset.select(batch_indices)

            batch_prompts = []
            for j, question in enumerate(batch_data):
                chat_formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question["prompt"]}],
                    add_generation_prompt=True,
                    tokenize=False
                )
                batch_prompts.append(chat_formatted_prompt)

            batch_inputs = tokenizer(batch_prompts, return_tensors="pt",
                                     padding=True, truncation=True, max_length=max_seq_length, padding_side="left").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            generated_ids = outputs[:, batch_inputs.input_ids.shape[-1]:]
            completions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            for j, completion in enumerate(completions):
                question_data = batch_data[j]
                input_example = InputExample(
                    key=question_data["key"],
                    instruction_id_list=question_data["instruction_id_list"],
                    prompt=question_data["prompt"],
                    kwargs=question_data["kwargs"]
                )
                is_correct = self.is_correct(input_example, completion)

                if is_correct:
                    correct += 1

        accuracy = correct / total_examples
        return accuracy
