"""
Async SDPO Training - Client for external vLLM inference server.

Simple architecture:
1. Start vLLM server separately: python -m inference.start_server --model ... --port 8000
2. Run this training script: accelerate launch -m ... --vllm-url http://localhost:8000

The training script fetches completions from the vLLM server via HTTP.
Weight updates require restarting the vLLM server with new checkpoint.
"""

import os
import json
import logging
import requests
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any, Tuple

import torch
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
from validators.validator import Validator

logger = logging.getLogger(__name__)


@dataclass
class AsyncSDPOConfig:
    """Configuration for async SDPO training with external vLLM server."""
    # vLLM server URL (must be started separately)
    vllm_url: str = "http://localhost:8000"

    # Batch sizes
    inference_batch_size: int = 4  # Questions per vLLM call
    training_batch_size: int = 16  # Completions per training step

    # Weight sync (manual - save checkpoint, restart server)
    checkpoint_for_vllm_interval: int = 0  # Save checkpoint every N steps (0 = disabled)


class VLLMClient:
    """Simple HTTP client for vLLM server."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_name = None

    def health_check(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_model_name(self) -> str:
        """Get the model name from the server."""
        if self._model_name is None:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            data = response.json()
            self._model_name = data["data"][0]["id"]
        return self._model_name

    def generate(
        self,
        prompts: List[str],
        num_completions: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 0.95,
    ) -> List[List[str]]:
        """Generate completions for prompts."""
        results = []
        model_name = self.get_model_name()

        for prompt in prompts:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": num_completions,
            }

            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(f"vLLM API error: {response.text}")
                raise RuntimeError(f"vLLM API error: {response.status_code}")

            data = response.json()
            completions = [choice["text"] for choice in data["choices"]]
            results.append(completions)

        return results


def generate_rollouts_from_vllm(
    client: VLLMClient,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, Any]],
    num_rollouts: int,
    temperature: float,
    max_tokens: int,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Generate rollouts using external vLLM server.

    Returns list of (prompt, completion, example) tuples.
    """
    # Build prompts
    prompts = []
    for example in examples:
        question = example.get("question_content", example.get("question", ""))
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)

    # Generate completions
    completions_list = client.generate(
        prompts=prompts,
        num_completions=num_rollouts,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Flatten results
    results = []
    for example, completions in zip(examples, completions_list):
        question = example.get("question_content", example.get("question", ""))
        for completion in completions:
            results.append((question, completion, example))

    return results


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
    Async SDPO Training with external vLLM server.

    Requires vLLM server to be running separately:
        CUDA_VISIBLE_DEVICES=0,1 python -m inference.start_server --model ... --port 8000

    Then run training:
        CUDA_VISIBLE_DEVICES=2,3 accelerate launch ... --vllm-url http://localhost:8000
    """
    os.makedirs(output_dir, exist_ok=True)

    if include_prior_solutions and verify_solution_fn is None:
        raise ValueError("verify_solution_fn is required when include_prior_solutions=True")

    # Initialize vLLM client
    vllm_client = VLLMClient(async_config.vllm_url)
    if not vllm_client.health_check():
        raise RuntimeError(
            f"vLLM server not available at {async_config.vllm_url}. "
            "Start it first with: python -m inference.start_server --model ... --port 8000"
        )
    logger.info(f"Connected to vLLM server at {async_config.vllm_url}")

    # Initialize accelerator
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=hparams.gradient_accumulation_steps,
        adjust_scheduler=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb" if wandb_project else None,
    )

    # Initialize wandb
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
                "vllm_url": async_config.vllm_url,
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

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # Initialize EMA teacher
    ema_decay = 1.0 - hparams.teacher_ema_rate
    unwrapped_for_teacher = accelerator.unwrap_model(model)
    teacher = EMATeacher(unwrapped_for_teacher, decay=ema_decay)

    model.train()

    # Training state
    global_step = 0
    total_loss = 0.0
    metrics_history = defaultdict(list)
    validation_history = defaultdict(list)

    if accelerator.is_main_process:
        logger.info(f"Starting async SDPO training")
        logger.info(f"vLLM server: {async_config.vllm_url}")
        logger.info(f"Inference batch size: {async_config.inference_batch_size}")
        logger.info(f"Training batch size: {async_config.training_batch_size}")
        logger.info(f"Num rollouts: {hparams.num_rollouts}")

    for epoch in range(hparams.num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch + 1}/{hparams.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
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

            # Generate rollouts from vLLM server
            try:
                rollouts = generate_rollouts_from_vllm(
                    client=vllm_client,
                    tokenizer=tokenizer,
                    examples=examples,
                    num_rollouts=hparams.num_rollouts,
                    temperature=hparams.rollout_temperature,
                    max_tokens=hparams.max_response_length,
                )
            except Exception as e:
                logger.error(f"Failed to generate rollouts: {e}")
                continue

            # Process rollouts in training batches
            for i in range(0, len(rollouts), async_config.training_batch_size):
                batch_rollouts = rollouts[i:i + async_config.training_batch_size]

                with accelerator.accumulate(model):
                    prompts = [r[0] for r in batch_rollouts]
                    completions = [r[1] for r in batch_rollouts]
                    batch_examples = [r[2] for r in batch_rollouts]

                    # Get feedback
                    feedbacks = []
                    prior_solutions_list = []
                    student_attempts_list = []

                    for prompt, completion, example in batch_rollouts:
                        feedback_result = get_feedback_fn(prompt, completion, example)
                        feedbacks.append(feedback_result.feedback_text)

                        example_id = example.get('id', example.get('question_title', str(hash(str(example)))))
                        if include_prior_solutions and feedback_result.success:
                            if verify_solution_fn(prompt, completion, example):
                                prior_solutions_store[example_id] = completion

                        prior_solution = prior_solutions_store.get(example_id) if include_prior_solutions else None
                        student_attempt = completion if hparams.include_student_attempt else None

                        prior_solutions_list.append(prior_solution)
                        student_attempts_list.append(student_attempt)

                    # Compute loss
                    if hparams.distill_on_regen:
                        loss, metrics = compute_distill_on_regen_loss_batched(
                            student_model=model,
                            teacher_model=teacher.model,
                            tokenizer=tokenizer,
                            prompts=prompts,
                            student_completions=completions,
                            feedbacks=feedbacks,
                            hparams=hparams,
                            student_attempts=student_attempts_list,
                        )
                    else:
                        loss, metrics = compute_sdpo_loss_batched(
                            student_model=model,
                            teacher_model=teacher.model,
                            tokenizer=tokenizer,
                            prompts=prompts,
                            completions=completions,
                            feedbacks=feedbacks,
                            prior_solutions=prior_solutions_list,
                            hparams=hparams,
                            student_attempts=student_attempts_list,
                        )

                    # Backward
                    accelerator.backward(loss)

                    # Gradient step
                    if accelerator.sync_gradients:
                        if hparams.max_grad_norm > 0:
                            accelerator.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                    # Update EMA teacher
                    if accelerator.sync_gradients:
                        teacher.update(accelerator.unwrap_model(model))
                        teacher.sync_across_processes(accelerator)

                # Logging
                if accelerator.sync_gradients:
                    global_step += 1
                    batch_loss = loss.item()
                    total_loss += batch_loss

                    if accelerator.is_main_process:
                        log_dict = {
                            "train/loss": batch_loss,
                            "train/lr": optimizer.param_groups[0]['lr'],
                            "train/kl_per_token": metrics.get('kl_per_token', 0),
                            "train/completion_tokens": metrics.get('completion_tokens', 0),
                            "train/global_step": global_step,
                        }
                        if hparams.distill_on_regen:
                            log_dict["train/teacher_completion_len"] = metrics.get('teacher_completion_len', 0)
                        accelerator.log(log_dict, step=global_step)

                    if accelerator.is_main_process and (global_step % hparams.log_interval == 0 or global_step == 1):
                        logged_avg_loss = total_loss / min(global_step, hparams.log_interval)
                        logger.info(f"Step {global_step} | Loss: {logged_avg_loss:.4f}")
                        metrics_history['loss'].append(logged_avg_loss)
                        metrics_history['step'].append(global_step)
                        total_loss = 0.0

                    # Validation
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

                        accelerator.wait_for_everyone()

                    # Save checkpoint
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

                        # Notify about weight update
                        if async_config.checkpoint_for_vllm_interval > 0:
                            if global_step % async_config.checkpoint_for_vllm_interval == 0:
                                logger.info(
                                    f">>> To update vLLM weights, restart server with: "
                                    f"python -m inference.start_server --model {checkpoint_dir}"
                                )

    # Final checkpoint
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        if wandb_project:
            accelerator.end_training()

    accelerator.wait_for_everyone()

    return {
        'metrics_history': dict(metrics_history),
        'validation_history': dict(validation_history),
        'final_scores': {},
        'prior_solutions': prior_solutions_store if include_prior_solutions else None,
    }
