"""
Code Execution Utilities for LiveCodeBench

Shared utilities for running Python code and capturing output/errors.
"""

import os
import re
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from decimal import Decimal
from concurrent.futures import ProcessPoolExecutor, as_completed


# Canonical LiveCodeBench base imports
BASE_IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\n"
    "import copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\n"
    "import operator\nimport io\nimport sys\nimport json\n"
    "sys.setrecursionlimit(50000)\n"
)

# Regex for extracting Python code from markdown
CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    error_message: Optional[str] = None


@dataclass
class TestCaseResult:
    """Result of running a single test case."""
    input: str
    expected_output: str
    actual_output: str
    passed: bool
    error_message: Optional[str] = None
    timed_out: bool = False


def extract_python_code(text: str) -> str:
    """
    Extract Python code from markdown code blocks.

    Args:
        text: Text potentially containing markdown code blocks

    Returns:
        Extracted code or the original text if no code block found
    """
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_code(
    code: str,
    stdin_input: str = "",
    timeout_seconds: int = 10,
    include_base_imports: bool = True,
) -> ExecutionResult:
    """
    Run Python code in a subprocess.

    Args:
        code: Python code to execute
        stdin_input: Input to provide via stdin
        timeout_seconds: Timeout in seconds
        include_base_imports: Whether to prepend BASE_IMPORTS

    Returns:
        ExecutionResult with stdout, stderr, and status
    """
    full_code = f"{BASE_IMPORTS}\n{code}" if include_base_imports else code

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(full_code)
        tmp_path = tmp_file.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            timed_out=False,
            error_message=result.stderr if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=-1,
            timed_out=True,
            error_message=f"Execution timed out after {timeout_seconds} seconds",
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=str(e),
            return_code=-1,
            timed_out=False,
            error_message=str(e),
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_stripped_lines(val: str) -> List[str]:
    """Strip and split text into lines."""
    val = (val or "").strip()
    return [line.strip() for line in val.split("\n")]


def convert_line_to_decimals(line: str) -> Tuple[bool, List[Decimal]]:
    """Try to convert a line of whitespace-separated values to Decimals."""
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def compare_outputs(actual: str, expected: str) -> bool:
    """
    Compare actual output to expected output.

    Uses exact string comparison first, then falls back to numeric comparison
    with Decimal precision.

    Args:
        actual: Actual output from code execution
        expected: Expected output

    Returns:
        True if outputs match
    """
    actual_lines = get_stripped_lines(actual)
    expected_lines = get_stripped_lines(expected)

    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line == expected_line:
            continue

        # Try numeric comparison
        ok_actual, dec_actual = convert_line_to_decimals(actual_line)
        ok_expected, dec_expected = convert_line_to_decimals(expected_line)

        if not ok_actual or not ok_expected:
            return False
        if dec_actual != dec_expected:
            return False

    return True


def run_test_case(
    code: str,
    test_case: Dict[str, str],
    timeout_seconds: int = 10,
) -> TestCaseResult:
    """
    Run a single test case against code.

    Args:
        code: Python code to test
        test_case: Dict with 'input', 'output', and optionally 'testtype'
        timeout_seconds: Timeout in seconds

    Returns:
        TestCaseResult with pass/fail status and details
    """
    test_type = test_case.get("testtype", "stdin")
    if test_type != "stdin":
        return TestCaseResult(
            input="",
            expected_output="",
            actual_output="",
            passed=False,
            error_message=f"Unsupported test type: {test_type}",
        )

    test_input = test_case.get("input", "")
    expected_output = str(test_case.get("output", ""))

    execution_result = run_code(code, stdin_input=test_input, timeout_seconds=timeout_seconds)

    if execution_result.timed_out:
        return TestCaseResult(
            input=test_input,
            expected_output=expected_output,
            actual_output="",
            passed=False,
            error_message=execution_result.error_message,
            timed_out=True,
        )

    if execution_result.return_code != 0:
        return TestCaseResult(
            input=test_input,
            expected_output=expected_output,
            actual_output=execution_result.stdout,
            passed=False,
            error_message=execution_result.stderr,
        )

    passed = compare_outputs(execution_result.stdout, expected_output)

    return TestCaseResult(
        input=test_input,
        expected_output=expected_output,
        actual_output=execution_result.stdout,
        passed=passed,
        error_message=None if passed else "Output mismatch",
    )


def run_test_cases(
    code: str,
    test_cases: List[Dict[str, str]],
    timeout_seconds: int = 10,
    stop_on_first_failure: bool = False,
) -> Tuple[bool, List[TestCaseResult]]:
    """
    Run multiple test cases against code.

    Args:
        code: Python code to test
        test_cases: List of test case dicts
        timeout_seconds: Timeout per test case
        stop_on_first_failure: Whether to stop after first failing test

    Returns:
        Tuple of (all_passed, list of TestCaseResults)
    """
    results = []
    all_passed = True
    saw_stdin_case = False

    for test_case in test_cases:
        if test_case.get("testtype") != "stdin":
            continue

        saw_stdin_case = True
        result = run_test_case(code, test_case, timeout_seconds)
        results.append(result)

        if not result.passed:
            all_passed = False
            if stop_on_first_failure:
                break

    # If no stdin cases, consider it a failure
    if not saw_stdin_case:
        return False, []

    return all_passed, results


def _run_single_test_case_worker(args: Tuple[str, Dict[str, str], int]) -> Tuple[int, TestCaseResult]:
    """Worker function for parallel test case execution. Returns (index, result)."""
    idx, code, test_case, timeout_seconds = args[0], args[1], args[2], args[3]
    result = run_test_case(code, test_case, timeout_seconds)
    return idx, result


def run_test_cases_parallel(
    code: str,
    test_cases: List[Dict[str, str]],
    timeout_seconds: int = 10,
    max_workers: Optional[int] = None,
) -> Tuple[bool, List[TestCaseResult]]:
    """
    Run multiple test cases against code in parallel.

    Args:
        code: Python code to test
        test_cases: List of test case dicts
        timeout_seconds: Timeout per test case
        max_workers: Max parallel workers (defaults to min(num_cases, 8))

    Returns:
        Tuple of (all_passed, list of TestCaseResults)
    """
    # Filter to stdin cases only
    stdin_cases = [(i, tc) for i, tc in enumerate(test_cases) if tc.get("testtype") == "stdin"]

    if not stdin_cases:
        return False, []

    if max_workers is None:
        max_workers = min(len(stdin_cases), 8)

    # Prepare args for workers
    worker_args = [(i, code, tc, timeout_seconds) for i, tc in stdin_cases]

    results = [None] * len(stdin_cases)
    all_passed = True

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_test_case_worker, args): args[0] for args in worker_args}

        for future in as_completed(futures):
            try:
                idx, result = future.result()
                # Find position in results list
                pos = next(j for j, (i, _) in enumerate(stdin_cases) if i == idx)
                results[pos] = result
                if not result.passed:
                    all_passed = False
            except Exception as e:
                # If worker fails, mark as failed
                all_passed = False

    # Filter out any None results (shouldn't happen but be safe)
    results = [r for r in results if r is not None]

    return all_passed, results
