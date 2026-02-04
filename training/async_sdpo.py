"""
Async SDPO Training with vLLM Inference Server and Multi-GPU DDP.

Separates inference (vLLM on dedicated GPUs) from training (remaining GPUs with DDP).
This allows inference and training to run in parallel, maximizing GPU utilization.

Architecture:
- Inference GPUs: Run vLLM server for fast batched rollout generation (managed by rank 0)
- Training GPUs: Run DDP training with accelerate (all ranks)
- Completion Queue: Buffers completions between inference and training (rank 0 only)
- Weight Sync: Periodically update vLLM weights to stay on-policy
- Batch Distribution: Rank 0 broadcasts batches to all training ranks

Usage:
    accelerate launch --num_processes <num_training_gpus> \
        -m experiments.lcb.qwen_1_7.run_async_experiment \
        --experiment env_feedback_only --num-gpus 8
"""

import os
import json
import copy
import time
import logging
import threading
import tempfile
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

from training.sdpo import (
    SDPOHparams,
    RolloutResult,
    FeedbackResult,
    ValidatorRunConfig,
    EMATeacher,
    compute_sdpo_loss_batched,
    compute_distill_on_regen_loss_batched,
)
from inference.vllm_client import VLLMInferenceClient, VLLMInferencePool
from inference.completion_queue import CompletionQueue, CompletionBatch, CompletionItem
from validators.validator import Validator

logger = logging.getLogger(__name__)


@dataclass
class AsyncSDPOConfig:
    """
    Runtime configuration for async SDPO training.

    This is populated from ExperimentConfig (in experiments/lcb/qwen_1_7/config.py)
    with CLI overrides. See ExperimentConfig for documentation of settings.

    Batch size relationships:
    - inference_batch_size * num_rollouts = completions per inference call
    - training_batch_size * gradient_accumulation_steps = effective training batch
    """
    # GPU allocation (determined at runtime based on available GPUs)
    inference_gpu_ids: List[int]
    training_gpu_ids: List[int]

    # Inference settings
    inference_batch_size: int = 4

    # vLLM settings
    vllm_port: int = 8000
    vllm_dtype: str = "bfloat16"
    vllm_max_model_len: int = 8192
    vllm_gpu_memory_utilization: float = 0.9

    # Queue settings
    queue_max_size: int = 1000
    training_batch_size: int = 16

    # Weight sync settings
    weight_sync_interval: int = 5
    weight_sync_dir: Optional[str] = None


def run_inference_loop(
    client: VLLMInferenceClient,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    completion_queue: CompletionQueue,
    num_rollouts: int,
    temperature: float,
    max_tokens: int,
    num_epochs: int,
    weight_sync_event: threading.Event,
    weight_sync_path_container: List[Optional[str]],
    stop_event: threading.Event,
):
    """
    Inference loop that runs in a separate thread (rank 0 only).

    Generates rollouts and adds them to the completion queue.
    """
    pool = VLLMInferencePool(client, tokenizer)

    for epoch in range(num_epochs):
        logger.info(f"[Inference] Starting epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            if stop_event.is_set():
                logger.info("[Inference] Stop signal received")
                return

            # Check for weight sync
            if weight_sync_event.is_set():
                new_weights_path = weight_sync_path_container[0]
                if new_weights_path:
                    logger.info(f"[Inference] Syncing weights from {new_weights_path}")
                    client.update_weights(new_weights_path)
                    weight_sync_path_container[0] = None
                weight_sync_event.clear()

            # Extract examples from batch
            if isinstance(batch, dict):
                if isinstance(batch.get('question_content', batch.get('question')), list):
                    batch_size = len(batch.get('question_content', batch.get('question', [])))
                    examples = [{k: v[i] if isinstance(v, list) else v for k, v in batch.items()}
                                for i in range(batch_size)]
                else:
                    examples = [batch]
            else:
                examples = [batch]

            # Generate rollouts
            try:
                results = pool.generate_rollouts(
                    examples=examples,
                    num_rollouts=num_rollouts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Add completions to queue
                for result in results:
                    for completion in result.completions:
                        item = CompletionItem(
                            prompt=result.prompt,
                            completion=completion,
                            example=result.example,
                        )
                        # Block until space available
                        while not completion_queue.put(item, timeout=1.0):
                            if stop_event.is_set():
                                return

                logger.debug(f"[Inference] Batch {batch_idx}: added {len(results) * num_rollouts} completions")

            except Exception as e:
                logger.error(f"[Inference] Error generating rollouts: {e}")
                continue

    logger.info("[Inference] Completed all epochs")
    completion_queue.shutdown()


def broadcast_batch(accelerator: Accelerator, batch: Optional[CompletionBatch]) -> Optional[CompletionBatch]:
    """
    Broadcast a CompletionBatch from rank 0 to all ranks.

    Args:
        accelerator: The accelerator instance
        batch: CompletionBatch on rank 0, None on other ranks

    Returns:
        CompletionBatch on all ranks, or None if rank 0 had None
    """
    if accelerator.num_processes == 1:
        return batch

    # Serialize batch on rank 0
    if accelerator.is_main_process:
        if batch is None:
            data = None
        else:
            data = {
                'prompts': batch.prompts,
                'completions': batch.completions,
                'examples': batch.examples,
            }
        serialized = pickle.dumps(data)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.long, device=accelerator.device)
    else:
        size_tensor = torch.tensor([0], dtype=torch.long, device=accelerator.device)

    # Broadcast size
    dist.broadcast(size_tensor, src=0)
    size = size_tensor.item()

    if size == 0:
        return None

    # Broadcast data
    if accelerator.is_main_process:
        data_tensor = torch.tensor(list(serialized), dtype=torch.uint8, device=accelerator.device)
    else:
        data_tensor = torch.zeros(size, dtype=torch.uint8, device=accelerator.device)

    dist.broadcast(data_tensor, src=0)

    # Deserialize on all ranks
    serialized = bytes(data_tensor.cpu().tolist())
    data = pickle.loads(serialized)

    if data is None:
        return None

    return CompletionBatch(
        prompts=data['prompts'],
        completions=data['completions'],
        examples=data['examples'],
    )


def broadcast_signal(accelerator: Accelerator, signal: bool) -> bool:
    """Broadcast a boolean signal from rank 0 to all ranks."""
    if accelerator.num_processes == 1:
        return signal

    signal_tensor = torch.tensor([1 if signal else 0], dtype=torch.long, device=accelerator.device)
    dist.broadcast(signal_tensor, src=0)
    return signal_tensor.item() == 1


def async_sdpo_train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    hparams: SDPOHparams,
    async_config: AsyncSDPOConfig,
    get_feedback_fn: Callable[[str, str, Any], FeedbackResult],
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    include_prior_solutions: bool = False,
    verify_solution_fn: Optional[Callable[[str, str, Any], bool]] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    output_dir: str = "outputs",
    prior_solutions_store: Optional[Dict[str, str]] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async SDPO Training with vLLM inference server and multi-GPU DDP.

    Must be launched with accelerate:
        accelerate launch --num_processes <num_training_gpus> -m ...

    Rank 0 manages the vLLM inference server and distributes batches to all ranks.
    All ranks participate in DDP training.

    Args:
        model: The language model to train
        tokenizer: The tokenizer
        dataloader: DataLoader yielding training examples (used by inference thread)
        hparams: Training hyperparameters
        async_config: Async training configuration
        get_feedback_fn: Function that gets environment feedback
        validators: List of (Validator, ValidatorRunConfig) tuples
        include_prior_solutions: Whether to include prior correct solutions
        verify_solution_fn: Function to verify if a solution is correct
        optimizer: Optional optimizer
        scheduler: Optional LR scheduler
        output_dir: Directory to save checkpoints
        prior_solutions_store: Optional dict of prior solutions
        wandb_project: W&B project name
        wandb_run_name: W&B run name

    Returns:
        Dictionary with training metrics and history
    """
    os.makedirs(output_dir, exist_ok=True)

    if include_prior_solutions and verify_solution_fn is None:
        raise ValueError("verify_solution_fn is required when include_prior_solutions=True")

    # Setup weight sync directory (rank 0 only, but all ranks need the path)
    weight_sync_dir = async_config.weight_sync_dir or tempfile.mkdtemp(prefix="sdpo_weights_")
    os.makedirs(weight_sync_dir, exist_ok=True)

    # Initialize accelerator for DDP training
    # Note: Don't manually set CUDA_VISIBLE_DEVICES - let accelerate handle it
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=hparams.gradient_accumulation_steps,
        adjust_scheduler=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb" if wandb_project else None,
    )

    logger.info(f"Rank {accelerator.process_index}/{accelerator.num_processes} initialized")

    # Initialize wandb (rank 0 only)
    if accelerator.is_main_process and wandb_project:
        accelerator.init_trackers(
            project_name=wandb_project,
            config={
                "learning_rate": hparams.learning_rate,
                "num_epochs": hparams.num_epochs,
                "gradient_accumulation_steps": hparams.gradient_accumulation_steps,
                "teacher_ema_rate": hparams.teacher_ema_rate,
                "top_k_distillation": hparams.top_k_distillation,
                "num_rollouts": hparams.num_rollouts,
                "include_prior_solutions": include_prior_solutions,
                "distill_on_regen": hparams.distill_on_regen,
                "inference_gpus": async_config.inference_gpu_ids,
                "training_gpus": async_config.training_gpu_ids,
                "num_training_processes": accelerator.num_processes,
                "async": True,
            },
            init_kwargs={"wandb": {"name": wandb_run_name}} if wandb_run_name else {},
        )

    # Initialize optimizer
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )

    # Initialize prior solutions store
    if prior_solutions_store is None:
        prior_solutions_store = {}

    # Prepare model with accelerator (handles DDP wrapping)
    model, optimizer = accelerator.prepare(model, optimizer)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # Initialize EMA teacher (each rank has its own copy, synced after updates)
    ema_decay = 1.0 - hparams.teacher_ema_rate
    unwrapped_for_teacher = accelerator.unwrap_model(model)
    teacher = EMATeacher(unwrapped_for_teacher, decay=ema_decay)

    model.train()

    # === Rank 0 only: Setup vLLM and inference ===
    vllm_client = None
    inference_thread = None
    completion_queue = None
    weight_sync_event = None
    weight_sync_path_container = None
    stop_event = None

    if accelerator.is_main_process:
        # Setup completion queue
        completion_queue = CompletionQueue(
            max_size=async_config.queue_max_size,
            batch_size=async_config.training_batch_size,
        )

        # Setup weight sync mechanisms
        weight_sync_event = threading.Event()
        weight_sync_path_container = [None]
        stop_event = threading.Event()

        # Initialize vLLM client
        vllm_client = VLLMInferenceClient(
            model_name_or_path=unwrapped_for_teacher.config._name_or_path,
            gpu_ids=async_config.inference_gpu_ids,
            port=async_config.vllm_port,
            dtype=async_config.vllm_dtype,
            max_model_len=async_config.vllm_max_model_len,
            gpu_memory_utilization=async_config.vllm_gpu_memory_utilization,
        )

        # Save initial weights for vLLM
        initial_weights_path = os.path.join(weight_sync_dir, "initial")
        unwrapped_for_teacher.save_pretrained(initial_weights_path)
        tokenizer.save_pretrained(initial_weights_path)

        logger.info(f"Starting vLLM server with initial weights from {initial_weights_path}")
        vllm_client.start_server(initial_weights_path)

        # Start inference thread
        inference_thread = threading.Thread(
            target=run_inference_loop,
            args=(
                vllm_client,
                tokenizer,
                dataloader,
                completion_queue,
                hparams.num_rollouts,
                hparams.rollout_temperature,
                hparams.max_response_length,
                hparams.num_epochs,
                weight_sync_event,
                weight_sync_path_container,
                stop_event,
            ),
            daemon=True,
        )
        inference_thread.start()
        logger.info("Started inference thread")

    # Sync all ranks before training starts
    accelerator.wait_for_everyone()

    # Training state
    global_step = 0
    total_loss = 0.0
    metrics_history = defaultdict(list)
    validation_history = defaultdict(list)

    # Training loop
    try:
        while True:
            # === Rank 0: Get batch from queue ===
            batch = None
            done = False

            if accelerator.is_main_process:
                batch = completion_queue.get_batch(timeout=5.0)
                if batch is None and completion_queue.is_shutdown() and completion_queue.empty():
                    done = True

            # Broadcast done signal to all ranks
            done = broadcast_signal(accelerator, done)
            if done:
                logger.info("Training complete")
                break

            # Broadcast batch to all ranks
            batch = broadcast_batch(accelerator, batch)
            if batch is None:
                continue

            # === All ranks: Process batch ===
            with accelerator.accumulate(model):
                # Get feedback for all completions (each rank does this independently)
                feedbacks = []
                prior_solutions_list = []
                student_attempts_list = []

                for i in range(len(batch)):
                    prompt = batch.prompts[i]
                    completion = batch.completions[i]
                    example = batch.examples[i]

                    # Get feedback
                    feedback_result = get_feedback_fn(prompt, completion, example)
                    feedbacks.append(feedback_result.feedback_text)

                    # Store correct solutions (rank 0 only to avoid conflicts)
                    if accelerator.is_main_process:
                        example_id = example.get('id', example.get('question_title', str(hash(str(example)))))
                        if include_prior_solutions and feedback_result.success:
                            if verify_solution_fn(prompt, completion, example):
                                prior_solutions_store[example_id] = completion

                    prior_solution = prior_solutions_store.get(
                        example.get('id', example.get('question_title', str(hash(str(example)))))
                    ) if include_prior_solutions else None
                    student_attempt = completion if hparams.include_student_attempt else None

                    prior_solutions_list.append(prior_solution)
                    student_attempts_list.append(student_attempt)

                # Compute loss
                if hparams.distill_on_regen:
                    loss, metrics = compute_distill_on_regen_loss_batched(
                        student_model=model,
                        teacher_model=teacher.model,
                        tokenizer=tokenizer,
                        prompts=batch.prompts,
                        student_completions=batch.completions,
                        feedbacks=feedbacks,
                        hparams=hparams,
                        student_attempts=student_attempts_list,
                    )
                else:
                    loss, metrics = compute_sdpo_loss_batched(
                        student_model=model,
                        teacher_model=teacher.model,
                        tokenizer=tokenizer,
                        prompts=batch.prompts,
                        completions=batch.completions,
                        feedbacks=feedbacks,
                        prior_solutions=prior_solutions_list,
                        hparams=hparams,
                        student_attempts=student_attempts_list,
                    )

                # Backward (accelerator handles gradient sync across ranks)
                accelerator.backward(loss)

                # Gradient step
                if accelerator.sync_gradients:
                    if hparams.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                # Update EMA teacher and sync across processes
                if accelerator.sync_gradients:
                    teacher.update(accelerator.unwrap_model(model))
                    teacher.sync_across_processes(accelerator)

            # Logging and weight sync
            if accelerator.sync_gradients:
                global_step += 1

                # Average loss across all processes
                loss_tensor = torch.tensor(loss.item(), device=accelerator.device)
                avg_loss_tensor = accelerator.reduce(loss_tensor, reduction="mean")
                batch_loss = avg_loss_tensor.item()
                total_loss += batch_loss

                # Log to wandb (rank 0 only)
                if accelerator.is_main_process:
                    log_dict = {
                        "train/loss": batch_loss,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/kl_per_token": metrics.get('kl_per_token', 0),
                        "train/completion_tokens": metrics.get('completion_tokens', 0),
                        "train/queue_size": completion_queue.size() if completion_queue else 0,
                        "train/global_step": global_step,
                    }
                    if hparams.distill_on_regen:
                        log_dict["train/teacher_completion_len"] = metrics.get('teacher_completion_len', 0)
                    accelerator.log(log_dict, step=global_step)

                # Console logging (rank 0 only)
                if accelerator.is_main_process and (global_step % hparams.log_interval == 0 or global_step == 1):
                    logged_avg_loss = total_loss / min(global_step, hparams.log_interval)
                    queue_size = completion_queue.size() if completion_queue else 0
                    logger.info(
                        f"Step {global_step} | Loss: {logged_avg_loss:.4f} | "
                        f"Queue: {queue_size}"
                    )
                    metrics_history['loss'].append(logged_avg_loss)
                    metrics_history['step'].append(global_step)
                    total_loss = 0.0

                # Weight sync to vLLM (rank 0 only)
                if accelerator.is_main_process and global_step % async_config.weight_sync_interval == 0:
                    sync_path = os.path.join(weight_sync_dir, f"step_{global_step}")
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(sync_path)
                    tokenizer.save_pretrained(sync_path)

                    weight_sync_path_container[0] = sync_path
                    weight_sync_event.set()
                    logger.info(f"Signaled weight sync at step {global_step}")

                # Validation (rank 0 only, other ranks wait)
                if global_step % hparams.validation_interval == 0:
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.eval()

                        for validator, val_config in validators:
                            logger.info(f"Running validator: {validator.name}")
                            try:
                                score = validator.validate(
                                    model=unwrapped_model,
                                    tokenizer=tokenizer,
                                    batch_size=val_config.batch_size,
                                    max_new_tokens=val_config.max_new_tokens,
                                    max_seq_length=val_config.max_seq_length,
                                )
                                logger.info(f"Validator {validator.name}: {score:.4f}")
                                validation_history[validator.name].append({
                                    'step': global_step,
                                    'score': score,
                                })
                                accelerator.log({f"val/{validator.name}": score}, step=global_step)
                            except Exception as e:
                                logger.error(f"Validator {validator.name} failed: {e}")

                        model.train()

                    # All ranks wait for validation to complete
                    accelerator.wait_for_everyone()

                # Save checkpoint (rank 0 only)
                if global_step % hparams.save_interval == 0 and accelerator.is_main_process:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                    if include_prior_solutions:
                        with open(os.path.join(checkpoint_dir, "prior_solutions.json"), "w") as f:
                            json.dump(prior_solutions_store, f)

                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Cleanup (rank 0 only)
        if accelerator.is_main_process:
            if stop_event:
                stop_event.set()
            if inference_thread:
                inference_thread.join(timeout=10)
            if vllm_client:
                vllm_client.stop_server()

    # Final validation and checkpoint (rank 0 only)
    final_scores = {}
    if accelerator.is_main_process:
        logger.info("Running final validation")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()

        for validator, val_config in validators:
            try:
                score = validator.validate(
                    model=unwrapped_model,
                    tokenizer=tokenizer,
                    batch_size=val_config.batch_size,
                    max_new_tokens=val_config.max_new_tokens,
                    max_seq_length=val_config.max_seq_length,
                )
                final_scores[validator.name] = score
                logger.info(f"Final {validator.name}: {score:.4f}")
                accelerator.log({f"val/{validator.name}": score}, step=global_step)
            except Exception as e:
                logger.error(f"Final validation for {validator.name} failed: {e}")

        # Save final checkpoint
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        if wandb_project:
            accelerator.end_training()

    # Ensure all processes finish together
    accelerator.wait_for_everyone()

    return {
        'metrics_history': dict(metrics_history),
        'validation_history': dict(validation_history),
        'final_scores': final_scores,
        'prior_solutions': prior_solutions_store if include_prior_solutions else None,
    }
