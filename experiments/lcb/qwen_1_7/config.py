"""
Shared configuration for Qwen3-1.7B LiveCodeBench experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidatorConfig:
    """Configuration for a single validator."""
    batch_size: int = 4
    max_new_tokens: int = 1024
    max_seq_length: int = 2048


@dataclass
class ExperimentConfig:
    """Configuration for a single SDPO experiment."""

    # Experiment identification
    name: str
    description: str

    # Model configuration
    model_name: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"

    # Dataset configuration
    dataset_subset_size: Optional[int] = None  # None = full dataset

    # Optimizer
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Training
    num_epochs: int = 2
    batch_size: int = 1  # Question batch size
    gradient_accumulation_steps: int = 32

    # Sequence lengths
    max_prompt_length: int = 2048
    max_response_length: int = 8192

    # SDPO loss (from paper Table 12)
    teacher_ema_rate: float = 0.01  # EMA update rate for teacher
    top_k_distillation: int = 20  # Top-K logits for distillation
    temperature: float = 1.0
    importance_sampling_clip: float = 2.0
    clip_advantages: Optional[float] = None

    # Rollouts
    num_rollouts: int = 8  # Number of rollouts per question
    rollout_temperature: float = 1.0

    # Logging and checkpointing
    log_interval: int = 1
    validation_interval: int = 10  # All validators run at this interval
    save_interval: int = 500

    # Feedback configuration
    include_prior_solutions: bool = False
    include_outside_feedback: bool = False
    include_student_attempt: bool = False  # Include student's attempt in teacher context

    # Distill-on-regen: generate with teacher, then distill student towards teacher
    distill_on_regen: bool = False
    regen_temperature: float = 0.7

    # Outside feedback configuration (if enabled)
    openai_model: str = "gpt-5-mini"

    # Per-validator configuration (batch_size, max_new_tokens, max_seq_length)
    validator_configs: dict = field(default_factory=lambda: {
        "fineweb": ValidatorConfig(batch_size=16, max_new_tokens=0, max_seq_length=1024),
        "mmlu": ValidatorConfig(batch_size=16, max_new_tokens=8, max_seq_length=1024),
        "ifeval": ValidatorConfig(batch_size=16, max_new_tokens=1024, max_seq_length=1024),
        "livecodebench": ValidatorConfig(batch_size=4, max_new_tokens=4096, max_seq_length=1024),
    })

    # Output
    output_dir: str = "outputs"


# Experiment 1: Environment feedback only
ENV_FEEDBACK_ONLY = ExperimentConfig(
    name="env_feedback_only",
    description="SDPO with environment feedback (test results) only",
    include_prior_solutions=False,
    include_outside_feedback=False,
    output_dir="outputs/qwen_1_7/env_feedback_only",
)

# Experiment 2: Environment feedback + prior correct solutions
ENV_FEEDBACK_WITH_PRIOR = ExperimentConfig(
    name="env_feedback_with_prior",
    description="SDPO with environment feedback + prior correct solutions in teacher context",
    include_prior_solutions=True,
    include_outside_feedback=False,
    output_dir="outputs/qwen_1_7/env_feedback_with_prior",
)

# Experiment 3: Environment feedback + outside feedback + prior correct solutions
FULL_FEEDBACK = ExperimentConfig(
    name="full_feedback",
    description="SDPO with environment feedback + GPT-5-mini critique + prior correct solutions",
    include_prior_solutions=True,
    include_outside_feedback=True,
    output_dir="outputs/qwen_1_7/full_feedback",
)

# Experiment 4: Distill-on-regen - teacher generates new completion, then distill
# This addresses the issue where standard SDPO has the teacher making token-wise
# corrections on the student's tokens, but each correction only sees prior student
# tokens (not the teacher's corrections). By having the teacher regenerate, its
# distribution is internally consistent - each position is conditioned on what the
# teacher actually generated.
DISTILL_ON_REGEN = ExperimentConfig(
    name="distill_on_regen",
    description="Distill-on-regen: teacher generates with feedback, student distills toward teacher's distribution",
    include_prior_solutions=False,
    include_outside_feedback=False,
    distill_on_regen=True,
    regen_temperature=0.7,
    output_dir="outputs/qwen_1_7/distill_on_regen",
)

# Experiment 5: Environment feedback + student attempt in teacher context
# The teacher sees the student's attempt alongside the feedback, giving it more
# context to understand what went wrong and how to correct it.
ENV_FEEDBACK_WITH_ATTEMPT = ExperimentConfig(
    name="env_feedback_with_attempt",
    description="SDPO with environment feedback + student attempt in teacher context",
    include_prior_solutions=False,
    include_outside_feedback=False,
    include_student_attempt=True,
    distill_on_regen=False,
    output_dir="outputs/qwen_1_7/env_feedback_with_attempt",
)

# Experiment 6: Distill-on-regen + student attempt in teacher context
# Combines distill-on-regen with the student's attempt visible to the teacher
# when regenerating, so the teacher knows what it's trying to improve.
DISTILL_ON_REGEN_WITH_ATTEMPT = ExperimentConfig(
    name="distill_on_regen_with_attempt",
    description="Distill-on-regen with student attempt + feedback in teacher context",
    include_prior_solutions=False,
    include_outside_feedback=False,
    include_student_attempt=True,
    distill_on_regen=True,
    regen_temperature=0.7,
    output_dir="outputs/qwen_1_7/distill_on_regen_with_attempt",
)

# All experiments
ALL_EXPERIMENTS = [
    ENV_FEEDBACK_ONLY,
    ENV_FEEDBACK_WITH_PRIOR,
    FULL_FEEDBACK,
    DISTILL_ON_REGEN,
    ENV_FEEDBACK_WITH_ATTEMPT,
    DISTILL_ON_REGEN_WITH_ATTEMPT,
]


def get_experiment_by_name(name: str) -> Optional[ExperimentConfig]:
    """Get an experiment configuration by name."""
    for exp in ALL_EXPERIMENTS:
        if exp.name == name:
            return exp
    return None
