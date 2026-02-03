#!/bin/bash
# Run distill_on_regen_with_attempt experiment
# Distill-on-regen with student's attempt + feedback visible to teacher when regenerating

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting distill_on_regen_with_attempt experiment..."
uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment distill_on_regen_with_attempt

echo "Experiment complete!"
