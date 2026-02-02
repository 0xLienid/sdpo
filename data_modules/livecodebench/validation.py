"""
Validation Functions for LiveCodeBench

Verify correctness of solutions against private test cases.
"""

import json
import zlib
import pickle
import base64
from typing import Dict, Any

from data_modules.livecodebench.code_execution import (
    extract_python_code,
    run_test_cases,
)


def verify_solution(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    use_private_tests: bool = True,
    timeout_seconds: int = 10,
) -> bool:
    """
    Verify if a solution is correct.

    For SDPO training, this is used to determine if a solution should be
    stored as a "prior correct solution" for future teacher context.

    Args:
        prompt: The original question/prompt (unused, for interface compatibility)
        completion: The generated code completion
        example: The example dict containing test cases
        use_private_tests: Whether to use private (hidden) test cases
        timeout_seconds: Timeout for each test case

    Returns:
        True if the solution passes all test cases
    """
    # Extract code from completion
    code = extract_python_code(completion)

    # Get test cases
    if use_private_tests:
        # Decode private test cases
        private_tests_raw = example.get("private_test_cases", "")
        try:
            decoded = pickle.loads(
                zlib.decompress(
                    base64.b64decode(private_tests_raw.encode("utf-8"))
                )
            )
            test_cases = json.loads(decoded)
        except Exception:
            # Fall back to public tests if private tests can't be decoded
            test_cases = json.loads(example.get("public_test_cases", "[]"))
    else:
        test_cases = json.loads(example.get("public_test_cases", "[]"))

    if not test_cases:
        return False

    # Run test cases
    all_passed, _ = run_test_cases(
        code=code,
        test_cases=test_cases,
        timeout_seconds=timeout_seconds,
        stop_on_first_failure=True,  # Optimization: stop early on failure
    )

    return all_passed


def verify_solution_public_only(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    timeout_seconds: int = 10,
) -> bool:
    """
    Verify solution using only public test cases.

    This is useful during training when we want to avoid "cheating" by
    using private test cases for verification.

    Args:
        prompt: The original question/prompt
        completion: The generated code completion
        example: The example dict
        timeout_seconds: Timeout for each test case

    Returns:
        True if solution passes all public test cases
    """
    return verify_solution(
        prompt=prompt,
        completion=completion,
        example=example,
        use_private_tests=False,
        timeout_seconds=timeout_seconds,
    )


def create_verification_fn(timeout_seconds: int = 10):
    """
    Factory function to create a verification function for SDPO training.

    This always uses PUBLIC test cases only. Private test cases are reserved
    exclusively for the LiveCodeBench validator evaluation.

    Args:
        timeout_seconds: Timeout for test execution

    Returns:
        A verification function compatible with SDPO training
    """
    def verification_fn(prompt: str, completion: str, example: Dict[str, Any]) -> bool:
        return verify_solution(
            prompt=prompt,
            completion=completion,
            example=example,
            use_private_tests=False,  # Never use private tests in training
            timeout_seconds=timeout_seconds,
        )
    return verification_fn
