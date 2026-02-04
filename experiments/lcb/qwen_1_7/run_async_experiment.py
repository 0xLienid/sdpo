"""
Run async SDPO experiments for Qwen3-1.7B on LiveCodeBench.

Two-process architecture:
1. Start vLLM server (separate terminal):
   CUDA_VISIBLE_DEVICES=0,1 uv run python -m inference.start_server \
       --model Qwen/Qwen3-1.7B --port 8000

2. Run training (this script):
   CUDA_VISIBLE_DEVICES=2,3 uv run python -m experiments.lcb.qwen_1_7.run_async_experiment \
       --experiment env_feedback_only --vllm-url http://localhost:8000

For multi-GPU training, use accelerate:
   CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --num_processes 4 \
       -m experiments.lcb.qwen_1_7.run_async_experiment \
       --experiment env_feedback_only --vllm-url http://localhost:8000
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

from training.async_sdpo import async_sdpo_train, AsyncSDPOConfig
from training.sdpo import SDPOHparams, ValidatorRunConfig
from data_modules.livecodebench import LiveCodeBenchDataset
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
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_validators(config: ExperimentConfig):
    """Initialize validators."""
    logger.info("Setting up validators...")

    validators = []
    val_configs = config.validator_configs

    def get_run_config(name: str) -> ValidatorRunConfig:
        vc = val_configs.get(name)
        if vc is None:
            return ValidatorRunConfig()
        return ValidatorRunConfig(
            batch_size=vc.batch_size,
            max_new_tokens=vc.max_new_tokens,
            max_seq_length=vc.max_seq_length,
        )

    try:
        lcb_validator = LiveCodeBenchValidator()
        validators.append((lcb_validator, get_run_config("livecodebench")))
        logger.info("  - LiveCodeBench validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load LiveCodeBench validator: {e}")

    try:
        fineweb_validator = FinewebValidator()
        validators.append((fineweb_validator, get_run_config("fineweb")))
        logger.info("  - FineWeb validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load FineWeb validator: {e}")

    try:
        mmlu_validator = MMLUValidator()
        validators.append((mmlu_validator, get_run_config("mmlu")))
        logger.info("  - MMLU validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load MMLU validator: {e}")

    try:
        ifeval_validator = IFEvalValidator()
        validators.append((ifeval_validator, get_run_config("ifeval")))
        logger.info("  - IFEval validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load IFEval validator: {e}")

    return validators


def run_async_experiment(
    config: ExperimentConfig,
    vllm_url: str,
    openai_api_key: Optional[str] = None,
    wandb_project: Optional[str] = None,
    training_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
):
    """Run async SDPO experiment."""
    training_batch_size = training_batch_size or config.training_batch_size
    gradient_accumulation_steps = gradient_accumulation_steps or config.async_gradient_accumulation_steps

    logger.info("=" * 60)
    logger.info(f"Starting ASYNC experiment: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"vLLM server: {vllm_url}")
    logger.info("=" * 60)

    output_dir = config.output_dir + "_async"
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "name": config.name,
            "description": config.description,
            "model_name": config.model_name,
            "vllm_url": vllm_url,
            "training_batch_size": training_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_rollouts": config.num_rollouts,
            "distill_on_regen": config.distill_on_regen,
            "include_student_attempt": config.include_student_attempt,
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load dataset
    logger.info("Loading LiveCodeBench dataset...")
    dataset = LiveCodeBenchDataset(subset_size=config.dataset_subset_size)
    dataloader = dataset.get_dataloader(batch_size=config.inference_batch_size, shuffle=True)
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    # Setup validators
    validators = setup_validators(config)

    # Setup feedback function
    feedback_fn = create_feedback_fn(
        include_outside_feedback=config.include_outside_feedback,
        openai_api_key=openai_api_key,
        openai_model=config.openai_model,
    )

    # Setup verification function
    verification_fn = None
    if config.include_prior_solutions:
        verification_fn = create_verification_fn()

    # Setup hyperparameters
    hparams = SDPOHparams(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        num_epochs=config.num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=config.max_prompt_length,
        max_response_length=config.max_response_length,
        teacher_ema_rate=config.teacher_ema_rate,
        top_k_distillation=config.top_k_distillation,
        temperature=config.temperature,
        importance_sampling_clip=config.importance_sampling_clip,
        clip_advantages=config.clip_advantages,
        distill_on_regen=config.distill_on_regen,
        regen_temperature=config.regen_temperature,
        include_student_attempt=config.include_student_attempt,
        num_rollouts=config.num_rollouts,
        rollout_temperature=config.rollout_temperature,
        log_interval=config.log_interval,
        validation_interval=config.validation_interval,
        save_interval=config.save_interval,
    )

    # Setup async config
    async_config = AsyncSDPOConfig(
        vllm_url=vllm_url,
        inference_batch_size=config.inference_batch_size,
        training_batch_size=training_batch_size,
    )

    # Run training
    logger.info("Starting async SDPO training...")
    results = async_sdpo_train(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        hparams=hparams,
        async_config=async_config,
        get_feedback_fn=feedback_fn,
        validators=validators,
        include_prior_solutions=config.include_prior_solutions,
        verify_solution_fn=verification_fn,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_run_name=f"{config.name}_async",
    )

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "metrics_history": results["metrics_history"],
            "validation_history": results["validation_history"],
            "final_scores": results["final_scores"],
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Experiment {config.name} completed!")
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run async SDPO experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[exp.name for exp in ALL_EXPERIMENTS],
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key for outside feedback",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Override dataset subset size",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=None,
        help="Completions per training step (default: from config)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: from config)",
    )

    args = parser.parse_args()

    config = get_experiment_by_name(args.experiment)
    if config is None:
        parser.error(f"Unknown experiment: {args.experiment}")

    if args.subset_size:
        config.dataset_subset_size = args.subset_size

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT")

    run_async_experiment(
        config=config,
        vllm_url=args.vllm_url,
        openai_api_key=openai_api_key,
        wandb_project=wandb_project,
        training_batch_size=args.training_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
