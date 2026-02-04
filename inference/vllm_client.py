"""
vLLM Inference Client for async SDPO training.

Manages a vLLM server process and provides async inference via HTTP API.
"""

import os
import json
import time
import signal
import logging
import subprocess
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for inference."""
    prompt: str
    example: Dict[str, Any]
    num_rollouts: int = 1
    temperature: float = 1.0
    max_tokens: int = 2048


@dataclass
class InferenceResult:
    """Result from inference."""
    prompt: str
    completions: List[str]
    example: Dict[str, Any]


class VLLMInferenceClient:
    """
    Client for vLLM inference server.

    Manages the vLLM server lifecycle and provides inference via HTTP API.
    Uses server restart for weight updates (simple but adds latency).
    """

    def __init__(
        self,
        model_name_or_path: str,
        gpu_ids: List[int],
        port: int = 8000,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the vLLM inference client.

        Args:
            model_name_or_path: HuggingFace model name or local path
            gpu_ids: List of GPU IDs to use for inference
            port: Port to run vLLM server on
            dtype: Model dtype (bfloat16, float16, etc.)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory fraction for vLLM
        """
        self.model_name_or_path = model_name_or_path
        self.gpu_ids = gpu_ids
        self.port = port
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self.server_process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"
        self._current_weights_path: Optional[str] = None

    def start_server(self, weights_path: Optional[str] = None) -> None:
        """
        Start the vLLM server.

        Args:
            weights_path: Path to model weights. If None, uses model_name_or_path.
        """
        if self.server_process is not None:
            self.stop_server()

        model_path = weights_path or self.model_name_or_path
        self._current_weights_path = model_path

        # Build CUDA_VISIBLE_DEVICES
        cuda_devices = ",".join(str(g) for g in self.gpu_ids)

        # Build vLLM command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--port", str(self.port),
            "--tensor-parallel-size", str(len(self.gpu_ids)),
            "--trust-remote-code",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        logger.info(f"Starting vLLM server on GPUs {cuda_devices} with model {model_path}")

        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        self._wait_for_server()
        logger.info(f"vLLM server ready at {self.base_url}")

    def _wait_for_server(self, timeout: int = 300, poll_interval: float = 2.0) -> None:
        """Wait for the vLLM server to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.Timeout:
                pass

            # Check if process died
            if self.server_process is not None and self.server_process.poll() is not None:
                stdout, _ = self.server_process.communicate()
                raise RuntimeError(
                    f"vLLM server died during startup. Output:\n{stdout.decode() if stdout else 'No output'}"
                )

            time.sleep(poll_interval)

        raise TimeoutError(f"vLLM server did not start within {timeout} seconds")

    def stop_server(self) -> None:
        """Stop the vLLM server."""
        if self.server_process is None:
            return

        logger.info("Stopping vLLM server...")

        # Send SIGTERM
        self.server_process.terminate()

        try:
            self.server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            self.server_process.kill()
            self.server_process.wait()

        self.server_process = None
        logger.info("vLLM server stopped")

    def update_weights(self, weights_path: str) -> None:
        """
        Update the server with new model weights.

        This restarts the server with the new weights path.
        Simple but adds latency (~30-60s for restart).

        Args:
            weights_path: Path to the new model weights
        """
        if weights_path == self._current_weights_path:
            logger.info("Weights path unchanged, skipping update")
            return

        logger.info(f"Updating vLLM server weights to {weights_path}")
        self.stop_server()
        self.start_server(weights_path)

    def generate(
        self,
        prompts: List[str],
        num_completions: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompts to complete
            num_completions: Number of completions per prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop: Optional stop sequences

        Returns:
            List of lists of completions, one list per prompt
        """
        if self.server_process is None:
            raise RuntimeError("vLLM server not started. Call start_server() first.")

        results = []

        for prompt in prompts:
            payload = {
                "model": self._current_weights_path or self.model_name_or_path,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": num_completions,
            }

            if stop:
                payload["stop"] = stop

            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=300,  # 5 minute timeout for long generations
            )

            if response.status_code != 200:
                logger.error(f"vLLM API error: {response.text}")
                raise RuntimeError(f"vLLM API error: {response.status_code}")

            data = response.json()
            completions = [choice["text"] for choice in data["choices"]]
            results.append(completions)

        return results

    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        num_completions: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 0.95,
    ) -> List[List[str]]:
        """
        Generate chat completions for a batch of message lists.

        Args:
            messages_list: List of message lists (each is a conversation)
            num_completions: Number of completions per conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter

        Returns:
            List of lists of completions
        """
        if self.server_process is None:
            raise RuntimeError("vLLM server not started. Call start_server() first.")

        results = []

        for messages in messages_list:
            payload = {
                "model": self._current_weights_path or self.model_name_or_path,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": num_completions,
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )

            if response.status_code != 200:
                logger.error(f"vLLM API error: {response.text}")
                raise RuntimeError(f"vLLM API error: {response.status_code}")

            data = response.json()
            completions = [choice["message"]["content"] for choice in data["choices"]]
            results.append(completions)

        return results

    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
        return False


class VLLMInferencePool:
    """
    Pool of vLLM inference requests with batching.

    Collects requests and batches them for efficient inference.
    """

    def __init__(
        self,
        client: VLLMInferenceClient,
        tokenizer,
        batch_size: int = 8,
    ):
        """
        Initialize the inference pool.

        Args:
            client: vLLM inference client
            tokenizer: Tokenizer for formatting prompts
            batch_size: Batch size for inference requests
        """
        self.client = client
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def generate_rollouts(
        self,
        examples: List[Dict[str, Any]],
        num_rollouts: int = 8,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> List[InferenceResult]:
        """
        Generate rollouts for a batch of examples.

        Args:
            examples: List of example dicts
            num_rollouts: Number of rollouts per example
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            List of InferenceResults
        """
        # Build prompts using chat template
        prompts = []
        for example in examples:
            question = example.get("question_content", example.get("question", ""))
            messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(prompt)

        # Generate completions
        completions_list = self.client.generate(
            prompts=prompts,
            num_completions=num_rollouts,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Build results
        results = []
        for example, prompt, completions in zip(examples, prompts, completions_list):
            question = example.get("question_content", example.get("question", ""))
            results.append(InferenceResult(
                prompt=question,
                completions=completions,
                example=example,
            ))

        return results
