"""
Episodic task collator for Stage 2 FOMAML meta-learning.
Packs episodes into Support Set (S_m) for inner-loop TTT adaptation
and Query Set (Q_m) for outer-loop meta-updates.
Supports 30% Self-Supervised Zero-Shot Fallback mode where S_m is the query prompt prefix.
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from .stage1_datamodule import SAMPLE_MULTI_TASK_PROMPTS, pad_collate_fn


@dataclass
class EpisodicTaskBatch:
    """Represents a single meta-learning episode with support and query sets."""

    support_batch: Dict[str, torch.Tensor]
    query_batch: Dict[str, torch.Tensor]
    is_self_supervised: bool
    task_name: str


class EpisodicTaskDataset(Dataset):
    """
    Dataset partitioned into distinct task clusters for episodic meta-training.
    Each episode samples M (3 to 5) examples from a task cluster.
    """

    def __init__(
        self,
        task_clusters: Dict[str, List[Dict[str, str]]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        self_supervised_ratio: float = 0.3,
        episode_length: int = 4,  # M examples per episode: (M-1) support + 1 query
    ):
        self.task_clusters = task_clusters
        self.task_names = list(task_clusters.keys())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.self_supervised_ratio = self_supervised_ratio
        self.episode_length = max(2, episode_length)

    def __len__(self) -> int:
        return sum(len(cluster) for cluster in self.task_clusters.values())

    def __getitem__(self, idx: int) -> EpisodicTaskBatch:
        # Select random task cluster
        task_name = random.choice(self.task_names)
        cluster = self.task_clusters[task_name]

        is_self_sup = (random.random() < self.self_supervised_ratio)

        if is_self_sup:
            # 1 item: prompt prefix forms support set, full prompt+response forms query set
            item = random.choice(cluster)
            query_item = item

            # Query format
            q_prompt = f"<|im_start|>user\n{query_item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            q_full = q_prompt + f"{query_item['response']}<|im_end|>"

            q_prompt_ids = self.tokenizer.encode(q_prompt, add_special_tokens=False)
            q_full_ids = self.tokenizer.encode(q_full, add_special_tokens=False)[: self.max_length]

            q_input_ids = torch.tensor(q_full_ids, dtype=torch.long)
            q_mask = torch.ones_like(q_input_ids)
            q_labels = q_input_ids.clone()
            q_labels[: min(len(q_prompt_ids), len(q_input_ids))] = -100

            # Support format: Unlabeled prompt only, self-supervised next-token prediction
            s_input_ids = torch.tensor(q_prompt_ids[: self.max_length], dtype=torch.long)
            s_mask = torch.ones_like(s_input_ids)
            # In self-supervised mode, labels are the prompt tokens themselves (standard causal LM)
            s_labels = s_input_ids.clone()

            support_batch = {
                "input_ids": s_input_ids.unsqueeze(0),
                "attention_mask": s_mask.unsqueeze(0),
                "labels": s_labels.unsqueeze(0),
            }
            query_batch = {
                "input_ids": q_input_ids.unsqueeze(0),
                "attention_mask": q_mask.unsqueeze(0),
                "labels": q_labels.unsqueeze(0),
            }

            return EpisodicTaskBatch(
                support_batch=support_batch,
                query_batch=query_batch,
                is_self_supervised=True,
                task_name=task_name,
            )

        # Few-Shot Mode: Sample M examples
        if len(cluster) >= self.episode_length:
            items = random.sample(cluster, self.episode_length)
        else:
            items = [random.choice(cluster) for _ in range(self.episode_length)]

        support_items = items[:-1]
        query_item = items[-1]

        # Process support items
        support_samples = []
        for s_item in support_items:
            s_prompt = f"<|im_start|>user\n{s_item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            s_full = s_prompt + f"{s_item['response']}<|im_end|>"
            s_p_ids = self.tokenizer.encode(s_prompt, add_special_tokens=False)
            s_f_ids = self.tokenizer.encode(s_full, add_special_tokens=False)[: self.max_length]
            s_in = torch.tensor(s_f_ids, dtype=torch.long)
            s_m = torch.ones_like(s_in)
            s_lbl = s_in.clone()
            s_lbl[: min(len(s_p_ids), len(s_in))] = -100
            support_samples.append({"input_ids": s_in, "attention_mask": s_m, "labels": s_lbl})

        # Process query item
        q_prompt = f"<|im_start|>user\n{query_item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        q_full = q_prompt + f"{query_item['response']}<|im_end|>"
        q_p_ids = self.tokenizer.encode(q_prompt, add_special_tokens=False)
        q_f_ids = self.tokenizer.encode(q_full, add_special_tokens=False)[: self.max_length]
        q_in = torch.tensor(q_f_ids, dtype=torch.long)
        q_m = torch.ones_like(q_in)
        q_lbl = q_in.clone()
        q_lbl[: min(len(q_p_ids), len(q_in))] = -100

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        support_batch = pad_collate_fn(support_samples, pad_token_id=pad_id)
        query_batch = pad_collate_fn([{"input_ids": q_in, "attention_mask": q_m, "labels": q_lbl}], pad_token_id=pad_id)

        return EpisodicTaskBatch(
            support_batch=support_batch,
            query_batch=query_batch,
            is_self_supervised=False,
            task_name=task_name,
        )


def create_stage2_episodic_loader(
    tokenizer: PreTrainedTokenizer,
    task_clusters: Optional[Dict[str, List[Dict[str, str]]]] = None,
    max_length: int = 512,
    self_supervised_ratio: float = 0.3,
    episode_length: int = 4,
    num_episodes: int = 2000,
) -> List[EpisodicTaskBatch]:
    """
    Creates a collection of episodic batches for Stage 2 FOMAML training.
    If task_clusters is not provided, builds clusters from multi-task sample prompts.
    """
    if task_clusters is None:
        # Group fallback templates by task_type
        task_clusters = {}
        for p in SAMPLE_MULTI_TASK_PROMPTS:
            t = p["task_type"]
            task_clusters.setdefault(t, []).append(p)

    dataset = EpisodicTaskDataset(
        task_clusters=task_clusters,
        tokenizer=tokenizer,
        max_length=max_length,
        self_supervised_ratio=self_supervised_ratio,
        episode_length=episode_length,
    )

    episodes = [dataset[i % len(dataset)] for i in range(num_episodes)]
    return episodes
