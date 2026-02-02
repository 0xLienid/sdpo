#!/bin/bash
# Run all Qwen3-1.7B LiveCodeBench experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Running all Qwen3-1.7B SDPO experiments"
echo "========================================"

# Check for OpenAI API key if running full_feedback experiment
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. The full_feedback experiment will not have outside feedback."
fi

# Run experiments
uv run python -m experiments.lcb.qwen_1_7.run_experiment --all

echo "========================================"
echo "All experiments completed!"
echo "========================================"
