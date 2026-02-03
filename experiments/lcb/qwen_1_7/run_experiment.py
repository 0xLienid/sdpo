"""
Run SDPO experiments for Qwen3-1.7B on LiveCodeBench.

Usage:
    uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment env_feedback_only
    uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment env_feedback_with_prior
    uv run python -m experiments.lcb.qwen_1_7.run_experiment --experiment full_feedback
    uv run python -m experiments.lcb.qwen_1_7.run_experiment --all
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from training.sdpo import sdpo_train, SDPOHparams, ValidatorRunConfig
from data_modules.livecodebench import (
    LiveCodeBenchDataset,
    livecodebench_rollout,
)
from data_modules.livecodebench.feedback import create_feedback_fn
from data_modules.livecodebench.validation import create_verification_fn
from validators.livecodebench.livecodebench_validator import LiveCodeBenchValidator
from validators.mmlu.mmlu_validator import MMLUValidator
from validators.fineweb.fineweb_validator import FinewebValidator
from validators.ifeval.ifeval_validator import IFEvalValidator

from experiments.lcb.qwen_1_7.config import (
    ExperimentConfig,
    ALL_EXPERIMENTS,
    get_experiment_by_name,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: ExperimentConfig):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {config.model_name}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded on device: {model.device}")
    return model, tokenizer


def setup_validators(config: ExperimentConfig):
    """Initialize validators with their configurations."""
    logger.info("Setting up validators...")

    validators = []
    val_configs = config.validator_configs

    def get_run_config(name: str) -> ValidatorRunConfig:
        """Convert experiment ValidatorConfig to training ValidatorRunConfig."""
        vc = val_configs.get(name)
        if vc is None:
            return ValidatorRunConfig()
        return ValidatorRunConfig(
            batch_size=vc.batch_size,
            max_new_tokens=vc.max_new_tokens,
            max_seq_length=vc.max_seq_length,
        )

    # LiveCodeBench - main metric for improvement
    try:
        lcb_validator = LiveCodeBenchValidator()
        run_config = get_run_config("livecodebench")
        validators.append((lcb_validator, run_config))
        logger.info(f"  - LiveCodeBench validator ready (batch_size={run_config.batch_size}, max_new_tokens={run_config.max_new_tokens})")
    except Exception as e:
        logger.warning(f"  - Failed to load LiveCodeBench validator: {e}")

    # FineWeb - perplexity for catastrophic forgetting
    try:
        fineweb_validator = FinewebValidator()
        run_config = get_run_config("fineweb")
        validators.append((fineweb_validator, run_config))
        logger.info(f"  - FineWeb validator ready (batch_size={run_config.batch_size})")
    except Exception as e:
        logger.warning(f"  - Failed to load FineWeb validator: {e}")

    # MMLU - knowledge retention
    try:
        mmlu_validator = MMLUValidator()
        run_config = get_run_config("mmlu")
        validators.append((mmlu_validator, run_config))
        logger.info(f"  - MMLU validator ready (batch_size={run_config.batch_size})")
    except Exception as e:
        logger.warning(f"  - Failed to load MMLU validator: {e}")

    # IFEval - instruction following
    try:
        ifeval_validator = IFEvalValidator()
        run_config = get_run_config("ifeval")
        validators.append((ifeval_validator, run_config))
        logger.info(f"  - IFEval validator ready (batch_size={run_config.batch_size}, max_new_tokens={run_config.max_new_tokens})")
    except Exception as e:
        logger.warning(f"  - Failed to load IFEval validator: {e}")

    return validators


def run_experiment(
    config: ExperimentConfig,
    openai_api_key: Optional[str] = None,
    wandb_project: Optional[str] = None,
):
    """Run a single SDPO experiment."""
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "name": config.name,
            "description": config.description,
            "model_name": config.model_name,
            "include_prior_solutions": config.include_prior_solutions,
            "include_outside_feedback": config.include_outside_feedback,
            "distill_on_regen": config.distill_on_regen,
            "regen_temperature": config.regen_temperature if config.distill_on_regen else None,
            "include_student_attempt": config.include_student_attempt,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load dataset
    logger.info("Loading LiveCodeBench dataset...")
    dataset = LiveCodeBenchDataset(
        subset_size=config.dataset_subset_size,
    )
    dataloader = dataset.get_dataloader(batch_size=config.batch_size, shuffle=True)
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    # Setup validators
    validators = setup_validators(config)

    # Setup feedback function
    feedback_fn = create_feedback_fn(
        include_outside_feedback=config.include_outside_feedback,
        openai_api_key=openai_api_key,
        openai_model=config.openai_model,
    )

    # Setup verification function (for prior solutions)
    # Always uses public tests - private tests are only for the validator
    verification_fn = None
    if config.include_prior_solutions:
        verification_fn = create_verification_fn()

    # Setup hyperparameters
    hparams = SDPOHparams(
        # Optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        # Training
        num_epochs=config.num_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Sequence lengths
        max_prompt_length=config.max_prompt_length,
        max_response_length=config.max_response_length,
        # SDPO loss
        teacher_ema_rate=config.teacher_ema_rate,
        top_k_distillation=config.top_k_distillation,
        temperature=config.temperature,
        importance_sampling_clip=config.importance_sampling_clip,
        clip_advantages=config.clip_advantages,
        # Distill-on-regen
        distill_on_regen=config.distill_on_regen,
        regen_temperature=config.regen_temperature,
        # Student attempt in teacher context
        include_student_attempt=config.include_student_attempt,
        # Rollouts
        num_rollouts=config.num_rollouts,
        rollout_temperature=config.rollout_temperature,
        # Logging
        log_interval=config.log_interval,
        validation_interval=config.validation_interval,
        save_interval=config.save_interval,
    )

    # Run training
    logger.info("Starting SDPO training...")
    results = sdpo_train(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        hparams=hparams,
        rollout_fn=livecodebench_rollout,
        get_feedback_fn=feedback_fn,
        validators=validators,
        include_prior_solutions=config.include_prior_solutions,
        verify_solution_fn=verification_fn,
        output_dir=config.output_dir,
        wandb_project=wandb_project,
        wandb_run_name=config.name,
    )

    # Save results
    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, "w") as f:
        # Convert any non-serializable objects
        serializable_results = {
            "metrics_history": results["metrics_history"],
            "validation_history": results["validation_history"],
            "final_scores": results["final_scores"],
            "completed_at": datetime.now().isoformat(),
        }
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Experiment {config.name} completed!")
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run SDPO experiments for Qwen3-1.7B")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[exp.name for exp in ALL_EXPERIMENTS],
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments sequentially",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key for outside feedback (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (or set WANDB_PROJECT env var)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Override dataset subset size (for debugging)",
    )

    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.error("Either --experiment or --all must be specified")

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "sdpo-lcb-qwen")

    if args.all:
        logger.info("Running all experiments...")
        for config in ALL_EXPERIMENTS:
            if args.subset_size:
                config.dataset_subset_size = args.subset_size
            run_experiment(config, openai_api_key, wandb_project)
    else:
        config = get_experiment_by_name(args.experiment)
        if config is None:
            parser.error(f"Unknown experiment: {args.experiment}")
        if args.subset_size:
            config.dataset_subset_size = args.subset_size
        run_experiment(config, openai_api_key, wandb_project)


if __name__ == "__main__":
    main()
