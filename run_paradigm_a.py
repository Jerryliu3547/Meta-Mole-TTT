"""
Dedicated Paradigm A Pipeline Runner for Meta-MoLE-TTT.
Focuses on Fast-Weight Scratchpad Working Memory & Context Compression.

Stages:
  1: (Optional) Train ERM static collaborative anchors.
  surgery: Gate surgery N -> N+1 (reuses tier1/stage1 anchors if available).
  2: FOMAML meta-training specifically for Paradigm A (compiles premise into B_dyn, evaluates on isolated query).
  eval: Head-to-head benchmark evaluation (Base Model vs. Meta-MoLE-TTT Memory Recall).
  full: Surgery -> Stage 2 -> Eval.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

import torch
import yaml

from src.data.benchmarks import PARADIGM_A_SAMPLE_ITEMS, extract_answer
from src.data.paradigm_a_datamodule import create_paradigm_a_episodic_loader
from src.data.stage1_datamodule import create_stage1_dataloader
from src.models.model_wrapper import MetaMoLEModel
from src.trainers.stage1_erm_trainer import Stage1ERMTrainer
from src.trainers.stage2_fomaml_trainer import Stage2FOMAMLTrainer
from src.trainers.ttt_inference_engine import TTTInferenceEngine
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: Dict[str, Any], device: torch.device):
    model_cfg = cfg["model"]
    model_name = model_cfg.get("name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

    print(f"[Paradigm A] Loading base model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        base_model.to(device)

    mole_model = MetaMoLEModel(
        base_model=base_model,
        target_modules=model_cfg.get("target_modules"),
        num_experts=model_cfg.get("num_experts", 6),
        rank=model_cfg.get("rank", 8),
        alpha=model_cfg.get("alpha", 0.4),
        router_temp=model_cfg.get("router_temp", 1.0),
        ttt_layer_indices=model_cfg.get("ttt_layer_indices", "upper_half"),
    )

    return mole_model, tokenizer


def run_stage1(cfg: Dict[str, Any], device: torch.device):
    print("=" * 70)
    print("PARADIGM A - STAGE 1: Static MoLE Anchor Stabilization (ERM)")
    print("=" * 70)
    mole_model, tokenizer = build_model_and_tokenizer(cfg, device)
    s1_cfg = cfg["stage1"]

    train_loader = create_stage1_dataloader(
        tokenizer=tokenizer,
        dataset_name_or_path=s1_cfg.get("dataset_name_or_path", "yahma/alpaca-cleaned"),
        batch_size=s1_cfg.get("batch_size", 4),
        max_length=s1_cfg.get("max_length", 512),
        num_samples=s1_cfg.get("total_steps", 500) * s1_cfg.get("batch_size", 4),
    )

    trainer = Stage1ERMTrainer(
        model=mole_model,
        train_dataloader=train_loader,
        lr_experts=float(s1_cfg.get("lr_experts", 1e-4)),
        lr_router=float(s1_cfg.get("lr_router", 5e-4)),
        weight_decay=float(s1_cfg.get("weight_decay", 0.01)),
        total_steps=s1_cfg.get("total_steps", 500),
        warmup_steps=s1_cfg.get("warmup_steps", 50),
        lambda_bal=float(s1_cfg.get("lambda_bal", 0.01)),
        lambda_z=float(s1_cfg.get("lambda_z", 1e-3)),
        lambda_ortho=float(s1_cfg.get("lambda_ortho", 0.05)),
        temp_start=float(s1_cfg.get("temp_start", 1.5)),
        temp_end=float(s1_cfg.get("temp_end", 0.7)),
        logging_steps=s1_cfg.get("logging_steps", 25),
        save_steps=s1_cfg.get("save_steps", 250),
        output_dir=s1_cfg.get("output_dir", "checkpoints/paradigm_a/stage1"),
        device=device,
    )
    trainer.train()
    print("[Paradigm A] Stage 1 Completed.")


def run_surgery(cfg: Dict[str, Any], device: torch.device):
    print("=" * 70)
    print("PARADIGM A - GATE SURGERY: N -> N+1 Gate Expansion")
    print("=" * 70)
    mole_model, _ = build_model_and_tokenizer(cfg, device)
    surg_cfg = cfg["surgery"]

    # Check if we can reuse existing Stage 1 anchors (e.g. from tier1/stage1)
    s1_src = surg_cfg.get("stage1_source_dir", "checkpoints/tier1/stage1")
    s1_fallback = cfg["stage1"].get("output_dir", "checkpoints/paradigm_a/stage1")

    chosen_s1 = None
    for cand in [s1_src, s1_fallback]:
        if os.path.exists(os.path.join(cand, "meta_mole_weights.pt")):
            chosen_s1 = cand
            break

    if chosen_s1:
        print(f"[Paradigm A] Loading static anchor weights from {chosen_s1}...")
        mole_model.load_mole_weights(chosen_s1, strict=False)
    else:
        print("[Paradigm A] Warning: No Stage 1 checkpoint found. Using fresh random initialization for anchors.")

    print("[Paradigm A] Performing expand_for_stage2_surgery()...")
    mole_model.expand_for_stage2_surgery(
        freeze_static_experts=surg_cfg.get("freeze_static_experts", True),
        scratchpad_bias_init=float(surg_cfg.get("scratchpad_bias_init", -2.0)),
    )

    save_dir = surg_cfg.get("output_dir", "checkpoints/paradigm_a/stage2_init")
    print(f"[Paradigm A] Saving surgery checkpoint to {save_dir}...")
    mole_model.save_mole_weights(save_dir)
    print("[Paradigm A] Gate Surgery Completed.")


def run_stage2(cfg: Dict[str, Any], device: torch.device):
    print("=" * 70)
    print("PARADIGM A - STAGE 2: FOMAML Fast-Weight Memory Meta-Training")
    print("Inner loop ingests premise into B_dyn; outer loop evaluates isolated query")
    print("=" * 70)
    mole_model, tokenizer = build_model_and_tokenizer(cfg, device)
    surg_ckpt = cfg["surgery"].get("output_dir", "checkpoints/paradigm_a/stage2_init")
    s2_cfg = cfg["stage2"]

    mole_model.expand_for_stage2_surgery(
        freeze_static_experts=cfg["surgery"].get("freeze_static_experts", True),
        scratchpad_bias_init=float(cfg["surgery"].get("scratchpad_bias_init", 0.8)),
    )

    s2_out_dir = s2_cfg.get("output_dir", "checkpoints/paradigm_a/stage2")
    if os.path.exists(os.path.join(s2_out_dir, "meta_mole_weights.pt")):
        print(f"[Paradigm A] Resuming continuous meta-training from existing Stage 2 checkpoint: {s2_out_dir}...")
        mole_model.load_mole_weights(s2_out_dir, strict=False)
        # Ensure step sizes are at least in the fast-weight memory regime if resuming from an older run
        with torch.no_grad():
            for l in mole_model.mole_layers:
                if l.is_ttt_layer and l.raw_eta_B_dyn.item() < -2.5:
                    l.raw_eta_B_dyn.fill_(-1.8)
    elif os.path.exists(os.path.join(surg_ckpt, "meta_mole_weights.pt")):
        print(f"[Paradigm A] Loading surgery checkpoint from {surg_ckpt}...")
        mole_model.load_mole_weights(surg_ckpt, strict=False)

    episodes = create_paradigm_a_episodic_loader(
        tokenizer=tokenizer,
        dataset_name_or_path=s2_cfg.get("dataset_name_or_path", "yahma/alpaca-cleaned"),
        max_length=s2_cfg.get("max_length", 512),
        num_episodes=s2_cfg.get("total_episodes", 200),
    )

    trainer = Stage2FOMAMLTrainer(
        model=mole_model,
        episodes=episodes,
        inner_steps=s2_cfg.get("inner_steps", 3),
        meta_lr_gate=float(s2_cfg.get("meta_lr_gate", 2e-4)),
        meta_lr_scratchpad=float(s2_cfg.get("meta_lr_scratchpad", 2e-4)),
        meta_lr_eta=float(s2_cfg.get("meta_lr_eta", 1e-3)),
        gamma_kl=float(s2_cfg.get("gamma_kl", 0.1)),
        gamma_ortho=float(s2_cfg.get("gamma_ortho", 0.05)),
        lambda_kl=float(s2_cfg.get("lambda_kl", 0.1)),
        meta_batch_size=s2_cfg.get("meta_batch_size", 2),
        total_episodes=s2_cfg.get("total_episodes", 200),
        logging_episodes=s2_cfg.get("logging_episodes", 20),
        save_episodes=s2_cfg.get("save_episodes", 100),
        output_dir=s2_cfg.get("output_dir", "checkpoints/paradigm_a/stage2"),
        device=device,
    )
    trainer.train()
    print("[Paradigm A] Stage 2 Meta-Training Completed.")


def run_eval(cfg: Dict[str, Any], device: torch.device):
    print("=" * 70)
    print("PARADIGM A - EVALUATION: Fast-Weight Scratchpad Memory Retrieval")
    print("Testing if B_dyn compiled facts can be retrieved from an isolated question")
    print("=" * 70)
    mole_model, tokenizer = build_model_and_tokenizer(cfg, device)
    s2_ckpt = cfg["stage2"].get("output_dir", "checkpoints/paradigm_a/stage2")

    mole_model.expand_for_stage2_surgery(
        freeze_static_experts=cfg["surgery"].get("freeze_static_experts", True),
        scratchpad_bias_init=float(cfg["surgery"].get("scratchpad_bias_init", 0.5)),
    )
    if os.path.exists(os.path.join(s2_ckpt, "meta_mole_weights.pt")):
        print(f"[Paradigm A] Loading trained checkpoint from {s2_ckpt}...")
        mole_model.load_mole_weights(s2_ckpt, strict=False)

    engine = TTTInferenceEngine(
        model=mole_model,
        tokenizer=tokenizer,
        inner_steps=cfg["inference"].get("inner_steps", 10),
        device=device,
    )

    items = PARADIGM_A_SAMPLE_ITEMS
    print(f"\nRunning {len(items)} held-out fast-weight memory retrieval tests:\n")

    base_correct = 0
    mole_mem_correct = 0
    mole_full_correct = 0

    for idx, item in enumerate(items):
        print("-" * 75)
        print(f"Test [{idx+1}/{len(items)}]: {item.task_name}")
        print(f"Context Premise : \"{item.context_premise}\"")
        print(f"Target Query    : \"{item.query_input}\"")
        print(f"Ground Truth    : {item.ground_truth}")

        full_context_query = f"{item.context_premise}\n{item.query_input}"

        # 1. Base Model Baseline (100% Information: Full Premise + Query in prompt)
        orig_scalings = [(l.scaling, l.dyn_scaling) for l in mole_model.mole_layers]
        orig_steps = engine.inner_steps
        for l in mole_model.mole_layers:
            l.scaling = 0.0
            l.dyn_scaling = 0.0
        engine.inner_steps = 0
        base_out = engine.adapt_and_generate(
            query_prompt=full_context_query,
            few_shot_demos=None,
            context_premise=None,
            max_new_tokens=64,
        ).strip().replace("\n", " ")

        # Restore scaling & inner steps
        for l_i, l in enumerate(mole_model.mole_layers):
            l.scaling, l.dyn_scaling = orig_scalings[l_i]
        engine.inner_steps = orig_steps

        # 2. Meta-MoLE-TTT (Paradigm A: Fast-Weight Memory Recall - Zero Context in Prompt)
        # Compiles premise into B_dyn via TTT, then answers query WITHOUT premise in prompt!
        mole_mem_out = engine.adapt_and_generate(
            query_prompt=item.query_input,
            few_shot_demos=None,
            context_premise=item.context_premise,
            max_new_tokens=64,
        ).strip().replace("\n", " ")

        # 3. Meta-MoLE-TTT (Full 100% Context + TTT Working Memory)
        mole_full_out = engine.adapt_and_generate(
            query_prompt=full_context_query,
            few_shot_demos=None,
            context_premise=item.context_premise,
            max_new_tokens=64,
        ).strip().replace("\n", " ")

        gt = item.ground_truth.lower().strip()
        base_pass = gt in base_out.lower()
        mole_mem_pass = gt in mole_mem_out.lower()
        mole_full_pass = gt in mole_full_out.lower()

        if base_pass:
            base_correct += 1
        if mole_mem_pass:
            mole_mem_correct += 1
        if mole_full_pass:
            mole_full_correct += 1

        base_status = "✓ PASS" if base_pass else "✗ FAIL"
        mole_mem_status = "✓ PASS" if mole_mem_pass else "✗ FAIL"
        mole_full_status = "✓ PASS" if mole_full_pass else "✗ FAIL"

        print(f"  [1. Base Model (100% Full Context in Prompt)]      : [{base_status}] Output: {base_out[:60]}")
        print(f"  [2. Meta-MoLE-TTT (Zero Context in Prompt Memory)] : [{mole_mem_status}] Output: {mole_mem_out[:60]}")
        print(f"  [3. Meta-MoLE-TTT (100% Full Context + TTT Memory)]: [{mole_full_status}] Output: {mole_full_out[:60]}")

    print("\n" + "=" * 75)
    print("PARADIGM A BENCHMARK SUMMARY")
    print(f"1. Base Model (100% Full Context in Prompt)       : {base_correct}/{len(items)} ({base_correct/len(items)*100:.1f}%)")
    print(f"2. Meta-MoLE-TTT (Zero Context in Prompt Memory)  : {mole_mem_correct}/{len(items)} ({mole_mem_correct/len(items)*100:.1f}%)")
    print(f"3. Meta-MoLE-TTT (100% Full Context + TTT Memory) : {mole_full_correct}/{len(items)} ({mole_full_correct/len(items)*100:.1f}%)")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Paradigm A Runner: Fast-Weight Working Memory")
    parser.add_argument("--config", type=str, default="configs/paradigm_a_qwen05b.yaml", help="Path to YAML config")
    parser.add_argument(
        "--stage",
        type=str,
        default="eval",
        choices=["1", "surgery", "2", "eval", "full"],
        help="Stage to execute",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--episodes", type=int, default=None, help="Override total_episodes for Stage 2")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Paradigm A] Using device: {device}")

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["stage2"]["total_episodes"] = args.episodes

    if args.stage == "1":
        run_stage1(cfg, device)
    elif args.stage == "surgery":
        run_surgery(cfg, device)
    elif args.stage == "2":
        run_stage2(cfg, device)
    elif args.stage == "eval":
        run_eval(cfg, device)
    elif args.stage == "full":
        s2_out = cfg["stage2"].get("output_dir", "checkpoints/paradigm_a/stage2")
        surg_out = cfg["surgery"].get("output_dir", "checkpoints/paradigm_a/stage2_init")
        if not os.path.exists(os.path.join(s2_out, "meta_mole_weights.pt")) and not os.path.exists(
            os.path.join(surg_out, "meta_mole_weights.pt")
        ):
            run_surgery(cfg, device)
        run_stage2(cfg, device)
        run_eval(cfg, device)


if __name__ == "__main__":
    main()

