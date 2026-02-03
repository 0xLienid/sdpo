#!/bin/bash
# Run env_feedback_with_attempt experiment
# Standard SDPO but teacher sees student's attempt + feedback in context

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting env_feedback_with_attempt experiment..."
uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment env_feedback_with_attempt

echo "Experiment complete!"
