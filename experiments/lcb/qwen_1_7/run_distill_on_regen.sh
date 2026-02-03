#!/bin/bash
# Run distill-on-regen experiment
# Teacher regenerates with feedback, student distills toward teacher's distribution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting distill_on_regen experiment..."
uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment distill_on_regen

echo "Experiment complete!"
