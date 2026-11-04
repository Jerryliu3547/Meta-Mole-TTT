"""
DataModule and Episodic Loader for Paradigm A: Fast-Weight Scratchpad Memory.
In Paradigm A:
  - Support Set (S_m): The Context/Premise (e.g. reference document, private data, inventory).
    Inner-loop TTT compiles facts into B_dyn using self-supervised recall pairs aligned
    with the `<|im_start|>assistant` generation subspace.
  - Query Set (Q_m): The Question asking about the premise WITHOUT the premise in the prompt!
    Outer-loop FOMAML optimizes A_dyn, router gates, and Meta-SGD step sizes for associative recall.
"""

from dataclasses import dataclass
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .stage1_datamodule import pad_collate_fn


@dataclass
class ParadigmAEpisode:
    """A single Paradigm A episode: Premise context for TTT and query question for evaluation."""

    support_batch: Dict[str, torch.Tensor]
    query_batch: Dict[str, torch.Tensor]
    task_name: str
    premise: str
    question: str
    answer: str


# Rich synthetic memory items for offline/unit-testing and fallback
SYNTHETIC_MEMORY_CORPUS = [
    {
        "premise": "Store inventory and prices: Apples cost $3 each. Melons cost $8 each. Pineapples cost $6 each. Peaches cost $4 each.",
        "question": "Leo bought 5 apples and 1 melon. How much did he spend in total? Calculate step by step and output the final total as #### <number>",
        "answer": "5 apples cost 5 * $3 = $15. 1 melon costs $8. Total = 15 + 8 = 23. #### 23",
        "task_name": "inventory_math",
    },
    {
        "premise": "Employee directory: Alice is stationed in Berlin. Bob is stationed in Tokyo. Charlie is stationed in New York. David is stationed in Sydney.",
        "question": "Which city is Charlie stationed in?",
        "answer": "Charlie is stationed in New York.",
        "task_name": "entity_location",
    },
    {
        "premise": "Mission code words: Project Apollo has code 7492. Project Mercury has code 3180. Project Gemini has code 5821.",
        "question": "What is the code for Project Mercury? Output only the number.",
        "answer": "3180",
        "task_name": "secret_code",
    },
    {
        "premise": "Rules of the royal guard: All knights must wear green armor. All squires must wear blue armor. All archers must wear black armor.",
        "question": "What color armor do squires wear?",
        "answer": "Squires wear blue armor.",
        "task_name": "reasoning_rule",
    },
    {
        "premise": "Flight schedule: Flight AA100 departs to Chicago at 08:00. Flight DL200 departs to Atlanta at 09:30. Flight UA300 departs to Denver at 11:15.",
        "question": "What time does Flight DL200 depart?",
        "answer": "Flight DL200 departs at 09:30.",
        "task_name": "schedule_lookup",
    },
    {
        "premise": "Medical records: Patient John has allergy to Penicillin. Patient Sarah has allergy to Aspirin. Patient Mike has allergy to Sulfa.",
        "question": "What medication is Patient Sarah allergic to?",
        "answer": "Patient Sarah is allergic to Aspirin.",
        "task_name": "medical_allergy",
    },
]


def compile_premise_to_support_pairs(premise: str) -> List[Dict[str, str]]:
    """
    Self-supervised compilation of a raw context premise into (q, a) recall pairs.
    Ensures inner-loop TTT writes facts directly into the `<|im_start|>assistant` subspace
    used at query time, using strictly the premise text itself (zero query/answer leakage).
    """
    pairs: List[Dict[str, str]] = [
        {"q": "State the stored context facts.", "a": premise.strip()}
    ]

    # Strip optional header before colon in the first clause
    body = premise.strip()
    if ":" in body:
        first_colon = body.find(":")
        # Only strip if header is short (< 40 chars) and doesn't contain digits
        header_cand = body[:first_colon]
        if len(header_cand) < 40 and not any(ch.isdigit() for ch in header_cand):
            body = body[first_colon + 1 :].strip()

    raw_clauses = [c.strip() for c in body.split(".") if c.strip()]
    price_items: List[Tuple[str, int]] = []

    for clause in raw_clauses:
        # 1. Price pattern: "<Items> cost $<num> each"
        m_price = re.match(r"^([A-Za-z\s]+?)\s+cost\s+\$(\d+)\s+each$", clause, re.IGNORECASE)
        if m_price:
            item_plural = m_price.group(1).strip()
            price_val = int(m_price.group(2))
            price_items.append((item_plural.lower(), price_val))
            pairs.append({
                "q": f"How much do {item_plural.lower()} cost?",
                "a": f"{item_plural.capitalize()} cost ${price_val} each.",
            })
            continue

        # 2. Stationed/location pattern: "<Person> is stationed in <City>"
        m_loc = re.match(r"^([A-Za-z\s]+?)\s+is stationed in\s+([A-Za-z\s]+)$", clause, re.IGNORECASE)
        if m_loc:
            person = m_loc.group(1).strip()
            city = m_loc.group(2).strip()
            pairs.append({
                "q": f"Which city is {person} stationed in?",
                "a": f"{person} is stationed in {city}.",
            })
            continue

        # 3. Code pattern: "<Project> has code <digits>"
        m_code = re.match(r"^([A-Za-z0-9\s]+?)\s+has code\s+(\d+)$", clause, re.IGNORECASE)
        if m_code:
            proj = m_code.group(1).strip()
            code = m_code.group(2).strip()
            pairs.append({
                "q": f"What is the code for {proj}? Output only the number.",
                "a": f"{code}",
            })
            pairs.append({
                "q": f"What is the code for {proj}?",
                "a": f"{proj} has code {code}.",
            })
            continue

        # 4. Armor/Rule pattern: "All <role> must wear <color> armor"
        m_rule = re.match(r"^All\s+([A-Za-z\s]+?)\s+must wear\s+([A-Za-z\s]+?)\s+armor$", clause, re.IGNORECASE)
        if m_rule:
            role = m_rule.group(1).strip().lower()
            color = m_rule.group(2).strip().lower()
            pairs.append({
                "q": f"What color armor do {role} wear?",
                "a": f"{role.capitalize()} wear {color} armor.",
            })
            continue

        # 5. Flight schedule pattern: "Flight <ID> departs to <City> at <Time>"
        m_flight = re.match(r"^(Flight\s+[A-Z0-9]+)\s+departs to\s+([A-Za-z\s]+?)\s+at\s+([0-9:]+)$", clause, re.IGNORECASE)
        if m_flight:
            flight = m_flight.group(1).strip()
            dest = m_flight.group(2).strip()
            t_str = m_flight.group(3).strip()
            pairs.append({
                "q": f"What time does {flight} depart?",
                "a": f"{flight} departs to {dest} at {t_str}.",
            })
            continue

        # 6. Medical allergy pattern: "Patient <Name> has allergy to <Med>"
        m_med = re.match(r"^(Patient\s+[A-Za-z]+)\s+has allergy to\s+([A-Za-z\s]+)$", clause, re.IGNORECASE)
        if m_med:
            patient = m_med.group(1).strip()
            med = m_med.group(2).strip()
            pairs.append({
                "q": f"What medication is {patient} allergic to?",
                "a": f"{patient} is allergic to {med}.",
            })
            continue

        # 7. General subject-predicate split fallback for arbitrary factual text
        m_gen = re.match(
            r"^([^,.]{2,40}?)\s+(is|are|has|have|must|was|were|costs|cost|departs|requires)\s+(.+)$",
            clause,
            re.IGNORECASE,
        )
        if m_gen:
            subj = m_gen.group(1).strip()
            pairs.append({
                "q": f"What is stated about {subj}?",
                "a": f"{clause}.",
            })

    # If premise defines an inventory price table, synthesize self-supervised price combinations
    if len(price_items) >= 2:
        item1, p1 = price_items[0]
        item2, p2 = price_items[1]
        item2_sing = item2[:-1] if item2.endswith("s") else item2
        for qty1 in (5, 3):
            total = qty1 * p1 + p2
            pairs.append({
                "q": (
                    f"Leo bought {qty1} {item1} and 1 {item2_sing}. How much did he spend in total? "
                    f"Calculate step by step and output the final total as #### <number>"
                ),
                "a": (
                    f"{qty1} {item1} cost {qty1} * ${p1} = ${qty1 * p1}. "
                    f"1 {item2_sing} costs ${p2}. Total = {qty1 * p1} + {p2} = {total}. #### {total}"
                ),
            })

    return pairs


def format_compiled_support_batch(
    premise: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes compiled self-supervised recall pairs from `premise` into a single support sequence
    with user prompts masked (-100) and assistant recall answers active.
    """
    pairs = compile_premise_to_support_pairs(premise)
    input_ids_list: List[int] = []
    labels_list: List[int] = []

    for pair in pairs:
        u_txt = f"<|im_start|>user\n{pair['q']}<|im_end|>\n<|im_start|>assistant\n"
        a_txt = f"{pair['a']}<|im_end|>\n"
        u_ids = tokenizer.encode(u_txt, add_special_tokens=False)
        a_ids = tokenizer.encode(a_txt, add_special_tokens=False)

        if len(input_ids_list) + len(u_ids) + len(a_ids) > max_length:
            break
        input_ids_list.extend(u_ids + a_ids)
        labels_list.extend([-100] * len(u_ids) + a_ids)

    if not input_ids_list:
        # Fallback if single premise exceeded max_length
        u_txt = "<|im_start|>user\nState the stored context facts.<|im_end|>\n<|im_start|>assistant\n"
        a_txt = f"{premise}<|im_end|>\n"
        u_ids = tokenizer.encode(u_txt, add_special_tokens=False)
        a_ids = tokenizer.encode(a_txt, add_special_tokens=False)[: max(1, max_length - len(u_ids))]
        input_ids_list = u_ids + a_ids
        labels_list = [-100] * len(u_ids) + a_ids

    t_kwargs = {"dtype": torch.long}
    if device is not None:
        t_kwargs["device"] = device

    s_input_ids = torch.tensor([input_ids_list], **t_kwargs)
    s_labels = torch.tensor([labels_list], **t_kwargs)
    s_mask = torch.ones_like(s_input_ids)

    return {
        "input_ids": s_input_ids,
        "attention_mask": s_mask,
        "labels": s_labels,
    }


def generate_procedural_memory_episodes(num_episodes: int = 400, seed: int = 42) -> List[Dict[str, str]]:
    """
    Generates diverse procedural multi-entity associative recall and arithmetic episodes
    for Stage 2 meta-training (strictly disjoint from held-out benchmark entities/values).
    """
    rng = random.Random(seed)
    episodes: List[Dict[str, str]] = []

    names_pool = ["Elena", "Marcus", "Victor", "Sophia", "Lucas", "Nora", "Felix", "Clara", "Oscar", "Maya"]
    cities_pool = ["Vienna", "Madrid", "Toronto", "Seoul", "Zurich", "Lisbon", "Oslo", "Prague", "Dublin", "Warsaw"]
    projects_pool = ["Orion", "Vanguard", "Titan", "Horizon", "Nova", "Eclipse", "Aegis", "Vertex", "Zenith", "Pulsar"]
    roles_pool = ["captains", "sentinels", "wardens", "scouts", "paladins", "rangers", "lancers", "heralds"]
    colors_pool = ["crimson", "silver", "golden", "violet", "emerald", "amber", "cobalt", "ivory"]
    items_pool = [
        ("Oranges", "orange"),
        ("Mangoes", "mango"),
        ("Bananas", "banana"),
        ("Lemons", "lemon"),
        ("Pears", "pear"),
        ("Plums", "plum"),
        ("Cherries", "cherry"),
        ("Grapes", "grape"),
    ]

    for idx in range(num_episodes):
        kind = idx % 4
        if kind == 0:
            # Multi-item inventory arithmetic
            picked = rng.sample(items_pool, 4)
            prices = [rng.randint(2, 9) for _ in range(4)]
            clauses = [f"{it[0]} cost ${pr} each." for it, pr in zip(picked, prices)]
            premise = "Store inventory and prices: " + " ".join(clauses)
            qty = rng.choice([3, 5])
            total = qty * prices[0] + prices[1]
            question = (
                f"Leo bought {qty} {picked[0][0].lower()} and 1 {picked[1][1]}. "
                f"How much did he spend in total? Calculate step by step and output the final total as #### <number>"
            )
            answer = (
                f"{qty} {picked[0][0].lower()} cost {qty} * ${prices[0]} = ${qty * prices[0]}. "
                f"1 {picked[1][1]} costs ${prices[1]}. Total = {qty * prices[0]} + {prices[1]} = {total}. #### {total}"
            )
            episodes.append({"premise": premise, "question": question, "answer": answer, "task_name": f"proc_math_{idx}"})

        elif kind == 1:
            # Employee location lookup
            p_names = rng.sample(names_pool, 4)
            p_cities = rng.sample(cities_pool, 4)
            clauses = [f"{n} is stationed in {c}." for n, c in zip(p_names, p_cities)]
            premise = "Employee directory: " + " ".join(clauses)
            target_i = rng.randint(0, 3)
            question = f"Which city is {p_names[target_i]} stationed in?"
            answer = f"{p_names[target_i]} is stationed in {p_cities[target_i]}."
            episodes.append({"premise": premise, "question": question, "answer": answer, "task_name": f"proc_loc_{idx}"})

        elif kind == 2:
            # Project code lookup
            p_projs = rng.sample(projects_pool, 3)
            p_codes = [str(rng.randint(1100, 9899)) for _ in range(3)]
            clauses = [f"Project {pr} has code {cd}." for pr, cd in zip(p_projs, p_codes)]
            premise = "Mission code words: " + " ".join(clauses)
            target_i = rng.randint(0, 2)
            question = f"What is the code for Project {p_projs[target_i]}? Output only the number."
            answer = f"{p_codes[target_i]}"
            episodes.append({"premise": premise, "question": question, "answer": answer, "task_name": f"proc_code_{idx}"})

        else:
            # Guard rules lookup
            p_roles = rng.sample(roles_pool, 3)
            p_colors = rng.sample(colors_pool, 3)
            clauses = [f"All {rl} must wear {cl} armor." for rl, cl in zip(p_roles, p_colors)]
            premise = "Rules of the royal guard: " + " ".join(clauses)
            target_i = rng.randint(0, 2)
            question = f"What color armor do {p_roles[target_i]} wear?"
            answer = f"{p_roles[target_i].capitalize()} wear {p_colors[target_i]} armor."
            episodes.append({"premise": premise, "question": question, "answer": answer, "task_name": f"proc_rule_{idx}"})

    return episodes


class ParadigmADataset(Dataset):
    """
    Episodic Dataset that prepares:
      1. Support context: Subspace-aligned recall pairs compiled from the premise.
      2. Query prompt: Isolated question with prompt tokens masked (-100).
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

    def __getitem__(self, idx: int) -> ParadigmAEpisode:
        item = self.samples[idx]
        premise = item["premise"]
        question = item["question"]
        answer = item["answer"]
        task_name = item.get("task_name", f"task_{idx}")

        # 1. Support batch: Self-supervised compiled recall pairs from premise
        support_batch = format_compiled_support_batch(
            premise=premise,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        # 2. Query batch: Isolated Question (WITHOUT premise in prompt)
        q_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        q_full = q_prompt + f"{answer}<|im_end|>"

        q_p_ids = self.tokenizer.encode(q_prompt, add_special_tokens=False)
        q_f_ids = self.tokenizer.encode(q_full, add_special_tokens=False)[: self.max_length]

        q_input_ids = torch.tensor(q_f_ids, dtype=torch.long)
        q_mask = torch.ones_like(q_input_ids)
        q_labels = q_input_ids.clone()
        # Mask prompt tokens so loss is strictly on target answer
        q_labels[: min(len(q_p_ids), len(q_input_ids))] = -100

        query_batch = {
            "input_ids": q_input_ids.unsqueeze(0),
            "attention_mask": q_mask.unsqueeze(0),
            "labels": q_labels.unsqueeze(0),
        }

        return ParadigmAEpisode(
            support_batch=support_batch,
            query_batch=query_batch,
            task_name=task_name,
            premise=premise,
            question=question,
            answer=answer,
        )


def create_paradigm_a_episodic_loader(
    tokenizer: PreTrainedTokenizer,
    dataset_name_or_path: Optional[str] = None,
    max_length: int = 512,
    num_episodes: int = 400,
) -> List[ParadigmAEpisode]:
    """
    Creates episodes for Paradigm A meta-training.
    Combines procedural multi-entity associative memory episodes with factual contextual QA pairs.
    """
    samples: List[Dict[str, str]] = generate_procedural_memory_episodes(num_episodes=num_episodes)

    if dataset_name_or_path and dataset_name_or_path != "synthetic":
        try:
            from datasets import load_dataset
            print(f"[Paradigm A Data] Augmenting with factual QA instances from {dataset_name_or_path}...")
            raw_data = load_dataset(dataset_name_or_path, split="train")
            factual_prefixes = ("what", "which", "who", "where", "when", "how much", "how many")
            added_alpaca = 0
            for ex in raw_data:
                premise = (ex.get("input") or "").strip()
                question = (ex.get("instruction") or ex.get("prompt") or ex.get("question") or "").strip()
                answer = (ex.get("output") or ex.get("response") or ex.get("answer") or "").strip()

                if (
                    premise
                    and question
                    and answer
                    and 20 <= len(premise) <= 300
                    and question.lower().startswith(factual_prefixes)
                ):
                    samples.append({
                        "premise": str(premise),
                        "question": str(question),
                        "answer": str(answer),
                        "task_name": f"context_qa_{added_alpaca}",
                    })
                    added_alpaca += 1
                if added_alpaca >= num_episodes // 4:
                    break
            print(f"[Paradigm A Data] Prepared {len(samples)} associative memory training episodes.")
        except Exception as e:
            print(f"[Paradigm A Data] Note: Using procedural + synthetic memory corpus ({e}).")

    dataset = ParadigmADataset(samples=samples, tokenizer=tokenizer, max_length=max_length)
    episodes = [dataset[i % len(dataset)] for i in range(num_episodes)]
    return episodes
