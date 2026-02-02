#!/bin/bash
# Run Experiment 2: Environment feedback + prior correct solutions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Experiment 2: Env Feedback + Prior Solutions"
echo "========================================"
echo "Teacher context: test results + prior correct solution (if available)"
echo "========================================"

uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment env_feedback_with_prior "$@"
