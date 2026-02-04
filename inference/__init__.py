"""
Inference module for vLLM server.

Usage:
    # Start server
    CUDA_VISIBLE_DEVICES=0,1 uv run python -m inference.start_server \
        --model Qwen/Qwen3-1.7B --port 8000
"""
