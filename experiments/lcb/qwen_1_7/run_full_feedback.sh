#!/bin/bash
# Run Experiment 3: Full feedback (env + outside + prior)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Experiment 3: Full Feedback"
echo "========================================"
echo "Teacher context: test results + GPT-5-mini critique + prior correct solution"
echo "========================================"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set!"
    echo "Set it with: export OPENAI_API_KEY=your-key"
    echo "Continuing without outside feedback..."
fi

uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment full_feedback "$@"
