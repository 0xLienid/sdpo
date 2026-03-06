import argparse
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _extract_ventile_vector(value: object, expected_len: int = 20) -> Optional[List[float]]:
    """
    Normalize ventile values to a flat list of floats.
    Handles both `[v0, ...]` and `[[v0, ...]]` shapes.
    """
    if not isinstance(value, list) or not value:
        return None

    candidate = value
    if isinstance(value[0], list):
        candidate = value[0]

    if not isinstance(candidate, list) or len(candidate) != expected_len:
        return None

    out: List[float] = []
    for item in candidate:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _nanmean(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _slope(values: List[float]) -> float:
    """
    Simple endpoint slope: (last - first) / 20.
    """
    if len(values) < 2:
        return float("nan")
    return (values[-1] - values[0]) / 20.0


def aggregate(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    filtered = []
    for problem in problems:
        metrics = problem.get("metrics", [])
        iter0s = [m for m in metrics if m.get("iteration") == 0]
        
        skip = False
        for iter0 in iter0s:
            if iter0["correct"]:
                skip = True
                break

        if skip:
            continue

        filtered.append(problem)

    kl_by_iteration = defaultdict(list)
    disagreement_by_iteration = defaultdict(list)
    step_count = 0

    for problem in filtered:
        for step in problem.get("metrics", []):
            step_count += 1
            iteration = step.get("iteration")

            kl_ventiles = step["metrics"]["kl_ventiles"][0]
            disagreement_ventiles = step["metrics"]["disagreement_ventiles"][0]

            kl_by_iteration[iteration].append(kl_ventiles)
            disagreement_by_iteration[iteration].append(disagreement_ventiles)

    kl_average_by_iteration = defaultdict(list)
    disagreement_average_by_iteration = defaultdict(list)
    for iteration in kl_by_iteration.keys():
        average_kl = []
        kl_vectors = kl_by_iteration[iteration]
        for i in range(20):
            value = 0.0
            for kl_vector in kl_vectors:
                value += kl_vector[i] / len(kl_vectors)
            average_kl.append(value)
        kl_average_by_iteration[iteration] = average_kl

        average_disagreement = []
        disagreement_vectors = disagreement_by_iteration[iteration]
        for i in range(20):
            value = 0.0
            for disagreement_vector in disagreement_vectors:
                value += disagreement_vector[i] / len(disagreement_vectors)
            average_disagreement.append(value)
        disagreement_average_by_iteration[iteration] = average_disagreement

    return {
        "total_steps": step_count,
        "total_problems": len(problems),
        "problems_after_filter": len(filtered),
        "iteration_average_kl_ventiles": kl_average_by_iteration,
        "iteration_average_disagreement_ventiles": disagreement_average_by_iteration
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter out problems where iteration 0 is correct, then average "
            "kl_ventiles and disagreement_ventiles across problems per iteration."
        )
    )
    parser.add_argument(
        "--input",
        default="analysis/results/test/experiment_4.json",
        help="Path to experiment_4.json",
    )
    args = parser.parse_args()

    results = aggregate(args.input)

    print(f"Total steps: {results['total_steps']}")
    print(f"Total problems: {results['total_problems']}")
    print(f"Problems after filtering iteration 0 correct: {results['problems_after_filter']}\n")

    print("Average KL ventiles by iteration:")
    for iteration in sorted(results["iteration_average_kl_ventiles"].keys()):
        values = results["iteration_average_kl_ventiles"][iteration]
        print(f"iteration_{iteration}_average_kl_ventiles = {values}")

    print("\nAverage disagreement ventiles by iteration:")
    for iteration in sorted(results["iteration_average_disagreement_ventiles"].keys()):
        values = results["iteration_average_disagreement_ventiles"][iteration]
        print(f"iteration_{iteration}_average_disagreement_ventiles = {values}")


if __name__ == "__main__":
    main()
