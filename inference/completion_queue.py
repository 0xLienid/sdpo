"""
Completion Queue for async SDPO training.

Buffers completions from inference server for training consumption.
"""

import threading
import queue
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CompletionItem:
    """A single completion item ready for training."""
    prompt: str
    completion: str
    example: Dict[str, Any]


@dataclass
class CompletionBatch:
    """A batch of completions for training."""
    prompts: List[str]
    completions: List[str]
    examples: List[Dict[str, Any]]

    @classmethod
    def from_items(cls, items: List[CompletionItem]) -> "CompletionBatch":
        """Create a batch from a list of items."""
        return cls(
            prompts=[item.prompt for item in items],
            completions=[item.completion for item in items],
            examples=[item.example for item in items],
        )

    def __len__(self) -> int:
        return len(self.prompts)


class CompletionQueue:
    """
    Thread-safe queue for completions awaiting training.

    Manages the buffer between inference and training pipelines.
    """

    def __init__(
        self,
        max_size: int = 1000,
        batch_size: int = 8,
    ):
        """
        Initialize the completion queue.

        Args:
            max_size: Maximum number of items in the queue
            batch_size: Number of items per training batch
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self._queue: queue.Queue[CompletionItem] = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._shutdown = False

    def put(self, item: CompletionItem, timeout: Optional[float] = None) -> bool:
        """
        Add a completion item to the queue.

        Args:
            item: CompletionItem to add
            timeout: Optional timeout in seconds

        Returns:
            True if item was added, False if queue is full or shutdown
        """
        if self._shutdown:
            return False

        try:
            self._queue.put(item, block=True, timeout=timeout)
            return True
        except queue.Full:
            return False

    def put_batch(
        self,
        prompts: List[str],
        completions_list: List[List[str]],
        examples: List[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> int:
        """
        Add multiple completions from a batch of examples.

        Args:
            prompts: List of prompts (one per example)
            completions_list: List of completion lists (multiple per example)
            examples: List of example dicts
            timeout: Optional timeout per item

        Returns:
            Number of items successfully added
        """
        added = 0

        for prompt, completions, example in zip(prompts, completions_list, examples):
            for completion in completions:
                item = CompletionItem(
                    prompt=prompt,
                    completion=completion,
                    example=example,
                )
                if self.put(item, timeout=timeout):
                    added += 1
                else:
                    break

        return added

    def get_batch(self, timeout: Optional[float] = None) -> Optional[CompletionBatch]:
        """
        Get a batch of completions for training.

        Blocks until batch_size items are available or timeout.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            CompletionBatch if available, None if timeout or shutdown
        """
        items = []

        while len(items) < self.batch_size:
            if self._shutdown and self._queue.empty():
                break

            try:
                item = self._queue.get(block=True, timeout=timeout)
                items.append(item)
            except queue.Empty:
                break

        if not items:
            return None

        return CompletionBatch.from_items(items)

    def get_available(self) -> Optional[CompletionBatch]:
        """
        Get all currently available items as a batch.

        Non-blocking, returns whatever is in the queue.

        Returns:
            CompletionBatch if items available, None otherwise
        """
        items = []

        while True:
            try:
                item = self._queue.get_nowait()
                items.append(item)
            except queue.Empty:
                break

        if not items:
            return None

        return CompletionBatch.from_items(items)

    def size(self) -> int:
        """Return current queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return self._queue.empty()

    def shutdown(self) -> None:
        """Signal shutdown - stop accepting new items."""
        self._shutdown = True

    def is_shutdown(self) -> bool:
        """Return True if queue is shutdown."""
        return self._shutdown
