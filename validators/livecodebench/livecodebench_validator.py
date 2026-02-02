import os
import re
import json
import zlib
import pickle
import base64
import subprocess
import tempfile
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from huggingface_hub import hf_hub_download
from validators.validator import Validator
from decimal import Decimal

# Canonical LiveCodeBench base imports to make common libraries available
BASE_IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\n"
    "sys.setrecursionlimit(50000)\n"
)


class LiveCodeBenchValidator(Validator):
    def __init__(self):
        super().__init__("livecodebench")

        self.CODE_BLOCK_RE = re.compile(
            r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
        self.files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]

        self.download_files()
        self.test_dataset = self.load_dataset().select(range(8))

    def download_files(self):
        for file in self.files:
            hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=file,
                repo_type="dataset",
                local_dir="data/livecodebench"
            )

    def load_dataset(self):
        data = []
        for file in self.files:
            with open(f"data/livecodebench/{file}", "r") as f:
                for line in f:
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    def extract_python_code(self, text: str) -> str:
        match = self.CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def get_stripped_lines(self, val: str) -> List[str]:
        val = (val or "").strip()
        return [line.strip() for line in val.split("\n")]

    def convert_line_to_decimals(self, line: str):
        try:
            return True, [Decimal(elem) for elem in line.split()]
        except Exception:
            return False, []

    def run_test_cases(self, code: str, test_cases: List[Dict[str, str]], timeout_seconds: int = 10) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(f"{BASE_IMPORTS}\n{code}")
            tmp_path = tmp_file.name

        try:
            saw_stdin_case = False
            for case in test_cases:
                if case.get("testtype") != "stdin":
                    continue
                saw_stdin_case = True

                test_input = case.get("input", "")
                expected_output = str(case.get("output", ""))

                result = subprocess.run(
                    ["python", tmp_path],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )

                stdout = result.stdout

                # Canonical behavior: compare line-wise after stripping; if not exact,
                # attempt numeric equality by converting whitespace-separated tokens to Decimals.
                pred_lines = self.get_stripped_lines(stdout)
                gt_lines = self.get_stripped_lines(expected_output)

                if len(pred_lines) != len(gt_lines):
                    return False

                for pred_line, gt_line in zip(pred_lines, gt_lines):
                    if pred_line == gt_line:
                        continue

                    ok_pred, dec_pred = self.convert_line_to_decimals(
                        pred_line)
                    ok_gt, dec_gt = self.convert_line_to_decimals(gt_line)
                    if not ok_pred or not ok_gt:
                        return False
                    if dec_pred != dec_gt:
                        return False

            # If there were no stdin cases at all, this should be considered a failure
            # rather than an implicit pass, to avoid overstating performance.
            return True if saw_stdin_case else False
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def validate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_new_tokens: int = 4096,
        max_seq_length: int = 4096,
        timeout_seconds: int = 10,
    ) -> float:
        print("Validating LiveCodeBench...")

        model.eval()

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        correct = 0
        total_questions = len(self.test_dataset)

        for i in range(0, total_questions, batch_size):
            print(f"Processing batch {i} of {total_questions}")

            batch_end = min(i + batch_size, total_questions)
            batch_indices = range(i, batch_end)
            batch_data = self.test_dataset.select(batch_indices)

            batch_prompts = []
            for example in batch_data:
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": example["question_content"]}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                batch_prompts.append(chat_prompt)

            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                padding_side="left",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    tokenizer=tokenizer,
                )

            generated_ids = outputs[:, batch_inputs.input_ids.shape[-1]:]
            completions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            for j, completion in enumerate(completions):
                raw_test_cases = batch_data[j]["private_test_cases"]
                raw_test_cases = pickle.loads(zlib.decompress(base64.b64decode(raw_test_cases.encode("utf-8"))))
                try:
                    test_cases = json.loads(raw_test_cases)
                except Exception:
                    test_cases = []

                code = self.extract_python_code(completion)
                if self.run_test_cases(code, test_cases):
                    correct += 1

        model.train()
        tokenizer.padding_side = original_padding_side
        
        return correct / total_questions if total_questions > 0 else 0.0
