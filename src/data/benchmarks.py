"""
Evaluation benchmarks for Meta-MoLE-TTT:
  1. BIG-Bench Hard (BBH): Complex algorithmic and deduction tasks.
  2. ARC-AGI: Abstraction and Reasoning Corpus (demonstration grids -> target grid).
  3. GSM-Symbolic: Grade-school math reasoning under distribution shift.
Provides answer extraction, evaluation loops, and benchmark datasets.
"""

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkItem:
    """A benchmark evaluation item with support demonstrations and a query question."""

    task_name: str
    few_shot_demos: List[Dict[str, str]]  # list of {"input": ..., "output": ...}
    query_input: str
    ground_truth: str


# Sample items for held-out evaluation
BBH_SAMPLE_ITEMS: List[BenchmarkItem] = [
    BenchmarkItem(
        task_name="bbh_boolean_expressions",
        few_shot_demos=[
            {
                "input": "Evaluate: True and not False",
                "output": "True",
            },
            {
                "input": "Evaluate: False or (True and False)",
                "output": "False",
            },
        ],
        query_input="Evaluate: not (True and (not False or False))",
        ground_truth="False",
    ),
    BenchmarkItem(
        task_name="bbh_causal_judgment",
        few_shot_demos=[
            {
                "input": "Frank drops a vase. It shatters on the floor. Did Frank cause the vase to shatter?",
                "output": "Yes",
            }
        ],
        query_input="Alice waters the plants every day. A storm comes and floods the garden. Did Alice cause the garden to flood?",
        ground_truth="No",
    ),
    BenchmarkItem(
        task_name="bbh_logical_deduction",
        few_shot_demos=[
            {
                "input": "There are three objects: a box, a ball, and a book. The box is heavier than the ball. The ball is heavier than the book. Which object is lightest?",
                "output": "The book",
            }
        ],
        query_input="There are three animals: a rabbit, a wolf, and a bear. The bear is larger than the wolf. The wolf is larger than the rabbit. Which animal is largest?",
        ground_truth="The bear",
    ),
]

ARC_AGI_SAMPLE_ITEMS: List[BenchmarkItem] = [
    BenchmarkItem(
        task_name="arc_color_inversion",
        few_shot_demos=[
            {
                "input": "[[0, 1], [1, 0]]",
                "output": "[[1, 0], [0, 1]]",
            },
            {
                "input": "[[0, 0], [1, 1]]",
                "output": "[[1, 1], [0, 0]]",
            },
        ],
        query_input="[[1, 0], [1, 0]]",
        ground_truth="[[0, 1], [0, 1]]",
    ),
    BenchmarkItem(
        task_name="arc_diagonal_fill",
        few_shot_demos=[
            {
                "input": "[[2, 0], [0, 2]]",
                "output": "[[2, 2], [2, 2]]",
            }
        ],
        query_input="[[3, 0], [0, 3]]",
        ground_truth="[[3, 3], [3, 3]]",
    ),
]

GSM_SYMBOLIC_SAMPLE_ITEMS: List[BenchmarkItem] = [
    BenchmarkItem(
        task_name="gsm_symbolic_algebra",
        few_shot_demos=[
            {
                "input": "Maya bought 3 pencils for $2 each and a notebook for $5. How much did she spend in total?",
                "output": "Total = (3 * 2) + 5 = 6 + 5 = 11. #### 11",
            }
        ],
        query_input="Leo bought 5 apples for $3 each and a melon for $8. How much did he spend in total?",
        ground_truth="23",
    ),
    BenchmarkItem(
        task_name="gsm_symbolic_rate",
        few_shot_demos=[
            {
                "input": "A car travels at 60 mph for 3 hours. How far does it go?",
                "output": "Distance = 60 * 3 = 180. #### 180",
            }
        ],
        query_input="A cyclist travels at 15 mph for 4 hours. How far does she travel?",
        ground_truth="60",
    ),
]


def extract_answer(generated_text: str, benchmark_name: str) -> str:
    """Extracts target answer string from model output according to benchmark format."""
    text = generated_text.strip()

    if "gsm" in benchmark_name.lower():
        # Look for #### 123 pattern
        match = re.search(r"####\s*(-?\d+)", text)
        if match:
            return match.group(1).strip()
        # Fallback to last number in text
        numbers = re.findall(r"-?\d+", text)
        return numbers[-1] if numbers else text

    if "bbh" in benchmark_name.lower():
        # Match boolean or options
        for opt in ["True", "False", "Yes", "No"]:
            if re.search(rf"\b{opt}\b", text, re.IGNORECASE):
                return opt
        # Fallback to first line
        return text.split("\n")[-1].strip()

    if "arc" in benchmark_name.lower():
        # Match grid pattern [[...]]
        grid_match = re.search(r"\[\[.*?\]\]", text)
        if grid_match:
            return grid_match.group(0).replace(" ", "")
        return text.replace(" ", "")

    return text


def evaluate_model_on_benchmark(
    eval_fn: Callable[[BenchmarkItem], str],
    benchmark_name: str,
    items: Optional[List[BenchmarkItem]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a model generation function on a specified benchmark.
    Returns accuracy and detailed prediction logs.
    """
    if items is None:
        if "bbh" in benchmark_name.lower():
            items = BBH_SAMPLE_ITEMS
        elif "arc" in benchmark_name.lower():
            items = ARC_AGI_SAMPLE_ITEMS
        elif "gsm" in benchmark_name.lower():
            items = GSM_SYMBOLIC_SAMPLE_ITEMS
        else:
            items = BBH_SAMPLE_ITEMS

    correct = 0
    total = len(items)
    records = []

    for item in items:
        prediction_text = eval_fn(item)
        extracted = extract_answer(prediction_text, benchmark_name)

        # Normalize comparison
        gt = item.ground_truth.strip().lower().replace(" ", "")
        pred = extracted.strip().lower().replace(" ", "")
        is_match = (gt == pred) or (gt in pred) or (pred in gt and len(pred) > 0)

        if is_match:
            correct += 1

        records.append({
            "task": item.task_name,
            "query": item.query_input,
            "prediction_raw": prediction_text,
            "extracted": extracted,
            "ground_truth": item.ground_truth,
            "is_correct": is_match,
        })

    accuracy = (correct / total) if total > 0 else 0.0
    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "records": records,
    }
