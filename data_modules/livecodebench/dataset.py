"""
LiveCodeBench Dataset Management

Handles loading and managing the LiveCodeBench dataset for SDPO training.
"""

import os
import json
import zlib
import pickle
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datasets import Dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader


@dataclass
class LiveCodeBenchExample:
    """A single LiveCodeBench example."""
    question_title: str
    question_content: str
    public_test_cases: List[Dict[str, str]]
    private_test_cases: List[Dict[str, str]]
    difficulty: str
    platform: str
    contest_id: str
    question_id: str


class LiveCodeBenchDataset:
    """
    LiveCodeBench dataset for SDPO training.

    Provides access to coding problems with public/private test cases.
    """

    FILES = [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ]

    def __init__(
        self,
        data_dir: str = "data/livecodebench",
        subset_size: Optional[int] = None,
    ):
        """
        Initialize the LiveCodeBench dataset.

        Training uses public test cases for feedback, validation uses private test cases.
        No train/test split of questions - all questions are used for both.

        Args:
            data_dir: Directory to store/load data
            subset_size: Optional limit on number of examples
        """
        self.data_dir = data_dir
        self.subset_size = subset_size

        self._download_files()
        self._dataset = self._load_dataset()

        if subset_size is not None:
            self._dataset = self._dataset.select(range(min(subset_size, len(self._dataset))))

    def _download_files(self):
        """Download dataset files from HuggingFace Hub."""
        os.makedirs(self.data_dir, exist_ok=True)

        for file in self.FILES:
            if not os.path.exists(os.path.join(self.data_dir, file)):
                hf_hub_download(
                    repo_id="livecodebench/code_generation_lite",
                    filename=file,
                    repo_type="dataset",
                    local_dir=self.data_dir,
                )

    def _load_dataset(self) -> Dataset:
        """Load all JSONL files into a HuggingFace Dataset."""
        data = []
        for file in self.FILES:
            filepath = os.path.join(self.data_dir, file)
            with open(filepath, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        return self._dataset[idx]

    def get_example(self, idx: int) -> LiveCodeBenchExample:
        """Get a typed example."""
        raw = self._dataset[idx]
        return self._parse_example(raw)

    def _parse_example(self, raw: Dict[str, Any]) -> LiveCodeBenchExample:
        """Parse a raw example into a typed dataclass."""
        # Decode public test cases
        public_tests = json.loads(raw.get("public_test_cases", "[]"))

        # Decode private test cases (compressed)
        private_tests_raw = raw.get("private_test_cases", "")
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(private_tests_raw.encode("utf-8"))))
            private_tests = json.loads(decoded)
        except Exception:
            private_tests = []

        return LiveCodeBenchExample(
            question_title=raw.get("question_title", ""),
            question_content=raw.get("question_content", ""),
            public_test_cases=public_tests,
            private_test_cases=private_tests,
            difficulty=raw.get("difficulty", ""),
            platform=raw.get("platform", ""),
            contest_id=raw.get("contest_id", ""),
            question_id=raw.get("question_id", str(idx)),
        )

    def get_public_test_cases(self, idx: int) -> List[Dict[str, str]]:
        """Get public test cases for an example."""
        raw = self._dataset[idx]
        return json.loads(raw.get("public_test_cases", "[]"))

    def get_private_test_cases(self, idx: int) -> List[Dict[str, str]]:
        """Get private test cases for an example (decoded)."""
        raw = self._dataset[idx]
        private_tests_raw = raw.get("private_test_cases", "")
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(private_tests_raw.encode("utf-8"))))
            return json.loads(decoded)
        except Exception:
            return []

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        """Get a PyTorch DataLoader for the dataset."""
        return DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Collate function for DataLoader."""
        if len(batch) == 1:
            return batch[0]

        # Stack into dict of lists
        result = {}
        for key in batch[0].keys():
            result[key] = [item[key] for item in batch]
        return result
