"""
Run async SDPO experiments for Qwen3-1.7B on LiveCodeBench.

Uses vLLM for inference on dedicated GPUs while DDP training runs on the rest.

Usage (8 GPUs total: 4 inference, 4 training):
    accelerate launch --num_processes 4 \
        -m experiments.lcb.qwen_1_7.run_async_experiment \
        --experiment env_feedback_only \
        --num-gpus 8 \
        --inference-batch-size 4 \
        --training-batch-size 16 \
        --gradient-accumulation-steps 4

GPU allocation:
    - --num-gpus: Total GPUs available
    - --inference-ratio: Fraction for inference (default 0.5)
    - accelerate --num_processes: Should equal num_training_gpus

    Example with 8 GPUs, inference_ratio=0.5:
        - GPUs 0-3: vLLM inference server (managed by rank 0)
        - GPUs 4-7: DDP training (4 processes)
        - Use: accelerate launch --num_processes 4

Batch size configuration:
    - inference-batch-size: Questions per vLLM generation call
    - num_rollouts (from config): Completions per question
    - training-batch-size: Completions per forward/backward pass
    - gradient-accumulation-steps: Steps before optimizer.step()

    Completions per inference = inference_batch_size * num_rollouts
    Effective training batch = training_batch_size * gradient_accumulation_steps
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Optional, List

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


def allocate_gpus(num_gpus: int, inference_ratio: float = 0.5) -> tuple[List[int], List[int]]:
    """
    Allocate GPUs between inference and training.

    Args:
        num_gpus: Total number of GPUs available
        inference_ratio: Fraction of GPUs for inference (default 0.5 = half)

    Returns:
        Tuple of (inference_gpu_ids, training_gpu_ids)
    """
    num_inference = max(1, int(num_gpus * inference_ratio))
    num_training = num_gpus - num_inference

    if num_training < 1:
        raise ValueError(f"Not enough GPUs for training. Total: {num_gpus}, inference: {num_inference}")

    inference_gpus = list(range(num_inference))
    training_gpus = list(range(num_inference, num_gpus))

    return inference_gpus, training_gpus


def setup_model_and_tokenizer(config: ExperimentConfig, training_gpus: List[int]):
    """Load model and tokenizer for training."""
    logger.info(f"Loading model: {config.model_name}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    # Note: Don't set device_map here since accelerate will handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Enable gradient checkpointing for memory efficiency
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
    """Initialize validators with their configurations."""
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

    # LiveCodeBench - main metric
    try:
        lcb_validator = LiveCodeBenchValidator()
        run_config = get_run_config("livecodebench")
        validators.append((lcb_validator, run_config))
        logger.info(f"  - LiveCodeBench validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load LiveCodeBench validator: {e}")

    # FineWeb - perplexity
    try:
        fineweb_validator = FinewebValidator()
        run_config = get_run_config("fineweb")
        validators.append((fineweb_validator, run_config))
        logger.info(f"  - FineWeb validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load FineWeb validator: {e}")

    # MMLU - knowledge retention
    try:
        mmlu_validator = MMLUValidator()
        run_config = get_run_config("mmlu")
        validators.append((mmlu_validator, run_config))
        logger.info(f"  - MMLU validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load MMLU validator: {e}")

    # IFEval - instruction following
    try:
        ifeval_validator = IFEvalValidator()
        run_config = get_run_config("ifeval")
        validators.append((ifeval_validator, run_config))
        logger.info(f"  - IFEval validator ready")
    except Exception as e:
        logger.warning(f"  - Failed to load IFEval validator: {e}")

    return validators


def run_async_experiment(
    config: ExperimentConfig,
    num_gpus: int,
    inference_ratio: float = 0.5,
    openai_api_key: Optional[str] = None,
    wandb_project: Optional[str] = None,
    # These default to None and fall back to config values
    weight_sync_interval: Optional[int] = None,
    inference_batch_size: Optional[int] = None,
    training_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    vllm_port: Optional[int] = None,
):
    """Run a single async SDPO experiment."""
    # Use config values as defaults, CLI args override
    weight_sync_interval = weight_sync_interval or config.weight_sync_interval
    inference_batch_size = inference_batch_size or config.inference_batch_size
    training_batch_size = training_batch_size or config.training_batch_size
    gradient_accumulation_steps = gradient_accumulation_steps or config.async_gradient_accumulation_steps
    vllm_port = vllm_port or config.vllm_port

    logger.info("=" * 60)
    logger.info(f"Starting ASYNC experiment: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info("=" * 60)

    # Allocate GPUs
    inference_gpus, training_gpus = allocate_gpus(num_gpus, inference_ratio)
    logger.info(f"GPU allocation:")
    logger.info(f"  Inference GPUs: {inference_gpus}")
    logger.info(f"  Training GPUs: {training_gpus}")

    # Create output directory
    output_dir = config.output_dir + "_async"
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
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
            "num_rollouts": config.num_rollouts,
            # Batch sizes
            "inference_batch_size": inference_batch_size,
            "training_batch_size": training_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_training_batch": training_batch_size * gradient_accumulation_steps,
            # Async settings
            "async": True,
            "inference_gpus": inference_gpus,
            "training_gpus": training_gpus,
            "weight_sync_interval": weight_sync_interval,
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config, training_gpus)

    # Load dataset
    logger.info("Loading LiveCodeBench dataset...")
    dataset = LiveCodeBenchDataset(
        subset_size=config.dataset_subset_size,
    )
    # Use inference_batch_size for the dataloader (questions per inference call)
    dataloader = dataset.get_dataloader(batch_size=inference_batch_size, shuffle=True)
    logger.info(f"Dataset loaded: {len(dataset)} examples")
    logger.info(f"Batch sizes: inference={inference_batch_size} questions, "
                f"training={training_batch_size} completions, GA={gradient_accumulation_steps}")
    logger.info(f"Completions per inference: {inference_batch_size * config.num_rollouts}")
    logger.info(f"Effective training batch: {training_batch_size * gradient_accumulation_steps}")

    # Setup validators
    validators = setup_validators(config)

    # Setup feedback function
    feedback_fn = create_feedback_fn(
        include_outside_feedback=config.include_outside_feedback,
        openai_api_key=openai_api_key,
        openai_model=config.openai_model,
    )

    # Setup verification function (for prior solutions)
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
        gradient_accumulation_steps=gradient_accumulation_steps,  # Use CLI arg, not config
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

    # Setup async config
    async_config = AsyncSDPOConfig(
        inference_gpu_ids=inference_gpus,
        training_gpu_ids=training_gpus,
        inference_batch_size=inference_batch_size,
        vllm_port=vllm_port,
        vllm_dtype=config.dtype,
        vllm_max_model_len=config.max_prompt_length + config.max_response_length,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        queue_max_size=config.completion_queue_size,
        training_batch_size=training_batch_size,
        weight_sync_interval=weight_sync_interval,
        weight_sync_dir=os.path.join(output_dir, "weight_sync"),
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
    parser = argparse.ArgumentParser(description="Run async SDPO experiments for Qwen3-1.7B")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[exp.name for exp in ALL_EXPERIMENTS],
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Total number of GPUs to use (half for inference, half for training)",
    )
    parser.add_argument(
        "--inference-ratio",
        type=float,
        default=0.5,
        help="Fraction of GPUs to use for inference (default: 0.5)",
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
    parser.add_argument(
        "--weight-sync-interval",
        type=int,
        default=None,
        help="Sync weights to inference server every N gradient steps (default: from config)",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Number of questions per inference batch (default: from config)",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=None,
        help="Number of completions per training forward/backward pass (default: from config)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: from config). Effective batch = training_batch_size * this",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="Port for vLLM server (default: from config)",
    )

    args = parser.parse_args()

    config = get_experiment_by_name(args.experiment)
    if config is None:
        parser.error(f"Unknown experiment: {args.experiment}")

    if args.subset_size:
        config.dataset_subset_size = args.subset_size

    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "sdpo-lcb-qwen-async")

    run_async_experiment(
        config=config,
        num_gpus=args.num_gpus,
        inference_ratio=args.inference_ratio,
        openai_api_key=openai_api_key,
        wandb_project=wandb_project,
        weight_sync_interval=args.weight_sync_interval,
        inference_batch_size=args.inference_batch_size,
        training_batch_size=args.training_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        vllm_port=args.vllm_port,
    )


if __name__ == "__main__":
    main()
