"""
Stage 1 DataModule for Meta-MoLE-TTT.
Loads and tokenizes multi-task instruction/reasoning mixtures for Stage 1 ERM training.
Supports HuggingFace datasets with automatic local/synthetic fallback for offline/cluster use.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


# Representative multi-task sample templates for fallback/offline/unit-testing
SAMPLE_MULTI_TASK_PROMPTS = [
    {
        "instruction": "Solve the algebraic equation: 3x + 12 = 33. Find the value of x step by step.",
        "response": "Step 1: Subtract 12 from both sides:\n3x = 33 - 12\n3x = 21\nStep 2: Divide both sides by 3:\nx = 21 / 3\nx = 7.\nThe value of x is 7.",
        "task_type": "symbolic_math",
    },
    {
        "instruction": "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring non-alphanumeric characters and casing.",
        "response": "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n```",
        "task_type": "coding",
    },
    {
        "instruction": "Premise: All mammals have backbones. A dolphin is a mammal.\nConclusion: Does a dolphin have a backbone? Explain using deductive reasoning.",
        "response": "Yes, a dolphin has a backbone.\nExplanation:\n1. Major premise: All mammals have backbones.\n2. Minor premise: A dolphin is classified as a mammal.\n3. By deductive inference (modus ponens), the dolphin must possess a backbone.",
        "task_type": "logical_deduction",
    },
    {
        "instruction": "Read the text and answer: 'Photosynthesis occurs in the chloroplasts of plant cells where sunlight is converted into chemical energy in glucose.' Where does photosynthesis take place?",
        "response": "Photosynthesis takes place in the chloroplasts of plant cells.",
        "task_type": "reading_comprehension",
    },
    {
        "instruction": "Given the table: | Item | Qty | Price |\n| Apples | 4 | $2 |\n| Oranges | 3 | $3 |\nWhat is the total cost of all items?",
        "response": "Total cost = (4 * $2) + (3 * $3) = $8 + $9 = $17.",
        "task_type": "table_qa",
    },
]


class InstructionDataset(Dataset):
    """
    Dataset that formats instruction-response pairs and tokenizes them for causal LM training.
    Prompt tokens are masked with -100 so cross-entropy loss is computed strictly on response tokens.
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        # Format prompt and completion
        prompt_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt_text + f"{response}<|im_end|>"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Mask prompt tokens
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def pad_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Dynamically pads tensors in batch to the longest sequence."""
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        cur_len = len(item["input_ids"])
        pad_len = max_len - cur_len

        # Pad right
        padded_ids = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(padded_mask)
        batch_labels.append(padded_labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
    }


def create_stage1_dataloader(
    tokenizer: PreTrainedTokenizer,
    dataset_name_or_path: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
    num_samples: int = 5000,
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a DataLoader for Stage 1 ERM training.
    Attempts to load from Hugging Face if dataset_name_or_path is provided,
    otherwise generates a synthetic stratified multi-task dataset.
    """
    samples: List[Dict[str, str]] = []

    if dataset_name_or_path and dataset_name_or_path != "synthetic":
        try:
            from datasets import load_dataset
            raw_data = load_dataset(dataset_name_or_path, split="train")
            for ex in raw_data:
                inst = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
                resp = ex.get("response") or ex.get("completion") or ex.get("answer") or ""
                if inst and resp:
                    samples.append({"instruction": str(inst), "response": str(resp)})
                if len(samples) >= num_samples:
                    break
        except Exception as e:
            print(f"[Stage1 Data] Warning: Could not load {dataset_name_or_path} ({e}). Using synthetic multi-task mixture.")
            samples = []

    if not samples:
        # Generate synthetic stratified mixture repeating multi-task templates
        base_samples = SAMPLE_MULTI_TASK_PROMPTS
        multiplier = (num_samples // len(base_samples)) + 1
        samples = (base_samples * multiplier)[:num_samples]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataset = InstructionDataset(samples=samples, tokenizer=tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: pad_collate_fn(b, pad_token_id=pad_id),
    )
