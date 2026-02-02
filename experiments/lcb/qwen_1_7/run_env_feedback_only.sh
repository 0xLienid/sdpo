#!/bin/bash
# Run Experiment 1: Environment feedback only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Experiment 1: Environment Feedback Only"
echo "========================================"
echo "Teacher context: test results only"
echo "========================================"

uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment env_feedback_only "$@"
