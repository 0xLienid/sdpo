"""
Feedback Functions for LiveCodeBench

Provides environment feedback and outside (LLM) feedback for SDPO training.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from training.sdpo import FeedbackResult
from data_modules.livecodebench.code_execution import (
    extract_python_code,
    run_test_cases,
    run_test_cases_parallel,
    TestCaseResult,
)

load_dotenv()


@dataclass
class DetailedFeedback:
    """Detailed feedback with test results and optional LLM critique."""
    environment_feedback: str
    test_results: List[TestCaseResult]
    all_passed: bool
    outside_feedback: Optional[str] = None


def format_test_results(test_results: List[TestCaseResult], max_results: int = 5) -> str:
    """
    Format test results into human-readable feedback.

    Args:
        test_results: List of TestCaseResult objects
        max_results: Maximum number of results to include

    Returns:
        Formatted string describing test results
    """
    if not test_results:
        return "No test cases were executed."

    lines = []
    passed_count = sum(1 for r in test_results if r.passed)
    total_count = len(test_results)

    lines.append(f"Test Results: {passed_count}/{total_count} passed")
    lines.append("")

    for i, result in enumerate(test_results[:max_results]):
        lines.append(f"Test Case {i + 1}:")
        lines.append(f"  Input: {result.input[:100]}{'...' if len(result.input) > 100 else ''}")
        lines.append(f"  Expected: {result.expected_output[:100]}{'...' if len(result.expected_output) > 100 else ''}")
        lines.append(f"  Actual: {result.actual_output[:100]}{'...' if len(result.actual_output) > 100 else ''}")

        if result.passed:
            lines.append("  Status: PASSED")
        else:
            lines.append("  Status: FAILED")
            if result.timed_out:
                lines.append("  Error: Execution timed out")
            elif result.error_message:
                # Truncate long error messages
                error = result.error_message[:500]
                if len(result.error_message) > 500:
                    error += "..."
                lines.append(f"  Error: {error}")

        lines.append("")

    if len(test_results) > max_results:
        remaining = len(test_results) - max_results
        lines.append(f"... and {remaining} more test cases")

    return "\n".join(lines)


def get_environment_feedback(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    timeout_seconds: int = 10,
) -> FeedbackResult:
    """
    Get environment feedback by running public test cases.

    Args:
        prompt: The original question/prompt
        completion: The generated code completion
        example: The example dict containing test cases
        timeout_seconds: Timeout for each test case

    Returns:
        FeedbackResult with test execution results
    """
    # Extract code from completion
    code = extract_python_code(completion)

    # Get public test cases
    public_tests_raw = example.get("public_test_cases", "[]")
    try:
        if isinstance(public_tests_raw, str):
            public_tests = json.loads(public_tests_raw)
        else:
            public_tests = public_tests_raw
    except json.JSONDecodeError:
        public_tests = []

    if not public_tests:
        return FeedbackResult(
            feedback_text="No public test cases available for this problem.",
            success=False,
            metadata={"test_results": [], "all_passed": False},
        )

    # Run test cases
    all_passed, test_results = run_test_cases(
        code=code,
        test_cases=public_tests,
        timeout_seconds=timeout_seconds,
        stop_on_first_failure=False,
    )

    # Format feedback
    feedback_text = format_test_results(test_results)

    return FeedbackResult(
        feedback_text=feedback_text,
        success=all_passed,
        metadata={
            "test_results": [
                {
                    "input": r.input,
                    "expected": r.expected_output,
                    "actual": r.actual_output,
                    "passed": r.passed,
                    "error": r.error_message,
                }
                for r in test_results
            ],
            "all_passed": all_passed,
            "passed_count": sum(1 for r in test_results if r.passed),
            "total_count": len(test_results),
        },
    )


def get_outside_feedback(
    prompt: str,
    completion: str,
    environment_feedback: str,
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> str:
    """
    Get feedback from an external LLM (GPT-5-mini).

    Args:
        prompt: The original question/prompt
        completion: The generated code completion
        environment_feedback: Results from test case execution
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        model: Model to use for critique

    Returns:
        LLM critique/feedback as a string
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No OpenAI API key provided. Cannot get outside feedback."

    client = OpenAI(api_key=api_key)

    critique_prompt = f"""You are an expert code reviewer. Analyze the following code submission for a programming problem.

## Problem
{prompt}

## Submitted Code
```python
{extract_python_code(completion)}
```

## Test Results
{environment_feedback}

Please provide a concise critique of the code. Focus on:
1. Why the code might be failing (if it failed any tests)
2. Logical errors or edge cases not handled
3. Suggestions for fixing the issues

Be specific and actionable. Keep your response under 300 words."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert programming assistant providing code review feedback."},
                {"role": "user", "content": critique_prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting outside feedback: {str(e)}"


def get_environment_and_outside_feedback(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    timeout_seconds: int = 10,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-5-mini",
) -> FeedbackResult:
    """
    Get combined environment and outside (LLM) feedback.

    Args:
        prompt: The original question/prompt
        completion: The generated code completion
        example: The example dict containing test cases
        timeout_seconds: Timeout for each test case
        openai_api_key: OpenAI API key
        openai_model: Model to use for critique

    Returns:
        FeedbackResult with combined feedback
    """
    # First get environment feedback
    env_result = get_environment_feedback(prompt, completion, example, timeout_seconds)

    # Then get outside feedback
    outside_feedback = get_outside_feedback(
        prompt=prompt,
        completion=completion,
        environment_feedback=env_result.feedback_text,
        api_key=openai_api_key,
        model=openai_model,
    )

    # Combine feedback
    combined_feedback = f"""## Environment Feedback (Test Results)
{env_result.feedback_text}

## Expert Code Review
{outside_feedback}"""

    return FeedbackResult(
        feedback_text=combined_feedback,
        success=env_result.success,
        metadata={
            **env_result.metadata,
            "outside_feedback": outside_feedback,
        },
    )


def create_feedback_fn(
    include_outside_feedback: bool = False,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-5-mini",
    timeout_seconds: int = 10,
):
    """
    Factory function to create a feedback function with configured parameters.

    Args:
        include_outside_feedback: Whether to include LLM critique
        openai_api_key: OpenAI API key (for outside feedback)
        openai_model: Model to use for critique
        timeout_seconds: Timeout for test execution

    Returns:
        A feedback function compatible with SDPO training
    """
    if include_outside_feedback:
        def feedback_fn(prompt: str, completion: str, example: Dict[str, Any]) -> FeedbackResult:
            return get_environment_and_outside_feedback(
                prompt=prompt,
                completion=completion,
                example=example,
                timeout_seconds=timeout_seconds,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
        return feedback_fn
    else:
        def feedback_fn(prompt: str, completion: str, example: Dict[str, Any]) -> FeedbackResult:
            return get_environment_feedback(
                prompt=prompt,
                completion=completion,
                example=example,
                timeout_seconds=timeout_seconds,
            )
        return feedback_fn


def _get_feedback_worker(args: Tuple[int, str, str, Dict[str, Any], int, bool]) -> Tuple[int, FeedbackResult]:
    """Worker function for parallel feedback collection. Returns (index, result)."""
    idx, prompt, completion, example, timeout_seconds, use_parallel_tests = args

    code = extract_python_code(completion)

    public_tests_raw = example.get("public_test_cases", "[]")
    try:
        if isinstance(public_tests_raw, str):
            public_tests = json.loads(public_tests_raw)
        else:
            public_tests = public_tests_raw
    except json.JSONDecodeError:
        public_tests = []

    if not public_tests:
        return idx, FeedbackResult(
            feedback_text="No public test cases available for this problem.",
            success=False,
            metadata={"test_results": [], "all_passed": False},
        )

    # Use parallel test execution within this rollout if requested
    if use_parallel_tests:
        all_passed, test_results = run_test_cases_parallel(
            code=code,
            test_cases=public_tests,
            timeout_seconds=timeout_seconds,
        )
    else:
        all_passed, test_results = run_test_cases(
            code=code,
            test_cases=public_tests,
            timeout_seconds=timeout_seconds,
            stop_on_first_failure=False,
        )

    feedback_text = format_test_results(test_results)

    return idx, FeedbackResult(
        feedback_text=feedback_text,
        success=all_passed,
        metadata={
            "test_results": [
                {
                    "input": r.input,
                    "expected": r.expected_output,
                    "actual": r.actual_output,
                    "passed": r.passed,
                    "error": r.error_message,
                }
                for r in test_results
            ],
            "all_passed": all_passed,
            "passed_count": sum(1 for r in test_results if r.passed),
            "total_count": len(test_results),
        },
    )


def get_feedback_batch_parallel(
    prompts: List[str],
    completions: List[str],
    examples: List[Dict[str, Any]],
    timeout_seconds: int = 10,
    max_workers: Optional[int] = None,
    parallel_tests_per_rollout: bool = True,
) -> List[FeedbackResult]:
    """
    Get feedback for multiple rollouts in parallel.

    Parallelizes across rollouts. Each rollout can also parallelize its test cases
    if parallel_tests_per_rollout is True.

    Args:
        prompts: List of prompts
        completions: List of completions
        examples: List of example dicts (can be same example repeated)
        timeout_seconds: Timeout per test case
        max_workers: Max parallel workers for rollouts (defaults to min(batch_size, 8))
        parallel_tests_per_rollout: Whether to also parallelize test cases within each rollout

    Returns:
        List of FeedbackResults in same order as inputs
    """
    batch_size = len(prompts)
    if batch_size == 0:
        return []

    if max_workers is None:
        max_workers = min(batch_size, 8)

    # Prepare worker args
    worker_args = [
        (i, prompts[i], completions[i], examples[i], timeout_seconds, parallel_tests_per_rollout)
        for i in range(batch_size)
    ]

    results = [None] * batch_size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_feedback_worker, args): args[0] for args in worker_args}

        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                # If worker fails, return error feedback
                idx = futures[future]
                results[idx] = FeedbackResult(
                    feedback_text=f"Error collecting feedback: {str(e)}",
                    success=False,
                    metadata={"error": str(e)},
                )

    return results


def create_batch_feedback_fn(
    timeout_seconds: int = 10,
    max_workers: Optional[int] = None,
    parallel_tests_per_rollout: bool = True,
):
    """
    Factory function to create a batch feedback function for parallel execution.

    Args:
        timeout_seconds: Timeout for test execution
        max_workers: Max parallel workers
        parallel_tests_per_rollout: Whether to parallelize tests within each rollout

    Returns:
        A batch feedback function that takes lists and returns list of FeedbackResults
    """
    def batch_feedback_fn(
        prompts: List[str],
        completions: List[str],
        examples: List[Dict[str, Any]],
    ) -> List[FeedbackResult]:
        return get_feedback_batch_parallel(
            prompts=prompts,
            completions=completions,
            examples=examples,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            parallel_tests_per_rollout=parallel_tests_per_rollout,
        )
    return batch_feedback_fn
