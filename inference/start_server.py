#!/usr/bin/env python
"""
Start vLLM inference server for async SDPO training.

Run this first, then run the training script in a separate terminal.

Usage:
    # Start server on GPUs 0-3
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m inference.start_server \
        --model Qwen/Qwen3-1.7B --port 8000

    # Or with a local checkpoint
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m inference.start_server \
        --model outputs/checkpoint-100 --port 8000
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path to local checkpoint",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run server on (default: 8000)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (default: number of visible GPUs)",
    )

    args = parser.parse_args()

    # Build vLLM command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--port", str(args.port),
        "--trust-remote-code",
    ]

    if args.tensor_parallel_size:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"Server will be available at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped")


if __name__ == "__main__":
    main()
