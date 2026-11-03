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
    few_shot_demos: Optional[List[Dict[str, str]]] = None  # list of {"input": ..., "output": ...}
    query_input: str = ""
    ground_truth: str = ""
    context_premise: Optional[str] = None  # Specific premise to compile into weights for Paradigm A


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

PARADIGM_A_SAMPLE_ITEMS: List[BenchmarkItem] = [
    BenchmarkItem(
        task_name="paradigm_a_inventory_math",
        context_premise="Store inventory and prices: Apples cost $3 each. Melons cost $8 each. Pineapples cost $6 each. Peaches cost $4 each.",
        query_input="Leo bought 5 apples and 1 melon. How much did he spend in total? Output the final total as #### <number>",
        ground_truth="23",
    ),
    BenchmarkItem(
        task_name="paradigm_a_entity_location",
        context_premise="Employee directory: Alice is stationed in Berlin. Bob is stationed in Tokyo. Charlie is stationed in New York. David is stationed in Sydney.",
        query_input="Which city is Charlie stationed in?",
        ground_truth="New York",
    ),
    BenchmarkItem(
        task_name="paradigm_a_secret_code",
        context_premise="Mission code words: Project Apollo has code 7492. Project Mercury has code 3180. Project Gemini has code 5821.",
        query_input="What is the code for Project Mercury? Output only the number.",
        ground_truth="3180",
    ),
    BenchmarkItem(
        task_name="paradigm_a_reasoning_rule",
        context_premise="Rules of the royal guard: All knights must wear green armor. All squires must wear blue armor. All archers must wear black armor.",
        query_input="What color armor do squires wear?",
        ground_truth="blue",
    ),
]



def extract_answer(generated_text: str, benchmark_name: str) -> str:
    """Extracts target answer string from model output according to benchmark format."""
    text = generated_text.strip()

    if "gsm" in benchmark_name.lower():
        # Look for #### 123 pattern
        match = re.search(r"####\s*(-?\d+[\d,]*)", text)
        if match:
            return match.group(1).replace(",", "").strip()
        # Look for "The answer is 123" or "= 123"
        ans_match = re.search(r"(?:answer is|=)\s*(-?\d+[\d,]*)", text, re.IGNORECASE)
        if ans_match:
            return ans_match.group(1).replace(",", "").strip()
        # Fallback to last number in text
        numbers = re.findall(r"-?\d+", text.replace(",", ""))
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


def load_benchmark_suite(
    benchmark: str = "gsm8k",
    num_samples: int = 100,
) -> List[BenchmarkItem]:
    """
    Loads benchmark instances (e.g. 100 questions) for large-scale evaluation.
    Supports GSM8K and multi-task reasoning with real datasets and robust fallbacks.
    """
    items: List[BenchmarkItem] = []

    gsm_demos = [
        {
            "input": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "output": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. #### 6",
        },
        {
            "input": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "output": "There are originally 3 cars. 2 more arrive. 3 + 2 = 5 cars. #### 5",
        },
    ]

    if "gsm" in benchmark.lower():
        try:
            from datasets import load_dataset
            print(f"[Benchmark] Fetching {num_samples} items from official openai/gsm8k test split...")
            raw = load_dataset("openai/gsm8k", "main", split="test")
            for i, row in enumerate(raw):
                if len(items) >= num_samples:
                    break
                q = row["question"]
                a = row["answer"]
                m = re.search(r"####\s*(-?\d+[\d,]*)", a)
                gt = m.group(1).replace(",", "").strip() if m else a.split()[-1]
                items.append(
                    BenchmarkItem(
                        task_name=f"gsm8k_{i+1}",
                        few_shot_demos=gsm_demos,
                        query_input=q,
                        ground_truth=gt,
                    )
                )
            print(f"[Benchmark] Successfully loaded {len(items)} GSM8K test questions.")
        except Exception as e:
            print(f"[Benchmark] Warning: Could not load GSM8K from Hugging Face ({e}). Generating parameterized problems.")
            items = []

        if not items:
            for i in range(num_samples):
                a = (i * 7 + 13) % 40 + 5
                b = (i * 3 + 17) % 25 + 3
                c = (i * 5 + 11) % 15 + 2
                ans = (a * b) + c
                q = f"A store sells packs of pens with {a} pens in each pack. John bought {b} packs and found {c} loose pens at home. How many pens does John have in total?"
                items.append(
                    BenchmarkItem(
                        task_name=f"gsm_param_{i+1}",
                        few_shot_demos=gsm_demos,
                        query_input=q,
                        ground_truth=str(ans),
                    )
                )

    return items[:num_samples]

