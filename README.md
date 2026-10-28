# Meta-MoLE-TTT: Meta-Gated Mixture of LoRA Experts with Orthogonal Test-Time Scratchpad

This repository implements **Meta-MoLE-TTT**, a decoupled plasticity-stability architecture for large language models trained via a strict **Two-Stage Curriculum**:

1. **Stage 1 (Collaborative MoLE Stabilization — No Scratchpad):** Trains $N$ collaborative LoRA experts $\{A_i, B_i\}_{i=1}^N$ and an $N$-way gating router $W_g^{(1:N)}$ end-to-end via Empirical Risk Minimization (ERM) with load-balancing, router Z-loss, and pairwise subspace orthogonality regularization until the collaborative feature basis is stabilized.
2. **Smooth $N \to N+1$ Gate Expansion Surgery:** Freezes the converged $N$ collaborative experts (preserving our exact Static MoLE baseline) and expands the router by one row with zero weights and small negative bias ($b_{g, N+1} = -2.0$), attaching an $(N+1)$-th **Dynamic Scratchpad LoRA** $(A_{\text{dyn}}, B_{\text{dyn}})$.
3. **Stage 2 (Bi-Level FOMAML + Meta-SGD Scratchpad):** Meta-trains the expanded router $W_g^{(1:N+1)}$, the scratchpad basis $A_{\text{dyn}}^{(0)}$, and per-layer inner learning rates $\boldsymbol{\eta}$ so that 1–3 test-time gradient steps rapidly absorb instance-specific residuals into the scratchpad without corrupting the frozen collaborative anchors.
4. **Stateless TTT Inference:** Supports both Few-Shot Demonstration TTT and Zero-Shot Self-Supervised Prompt TTT, refreshes the KV cache cleanly, and strictly resets the model after generation.

---

## Repository Structure

```text
meta_mole_ttt/
├── configs/
│   ├── tier0_smollm_debug.yaml       # Fast 2-minute pipeline verification (SmolLM2-135M)
│   ├── tier1_qwen05b_ablation.yaml   # Fast ablation sweeps (Qwen2.5-0.5B)
│   └── tier2_llama8b_main.yaml       # Full conference experiment config (Llama-3.1-8B)
├── src/
│   ├── models/
│   │   ├── meta_mole_layer.py        # MetaMoLELinear with Stage 1 & 2 modes + N->N+1 surgery
│   │   ├── model_wrapper.py          # Injects MetaMoLELinear into HF causal LMs + upper-half targeting
│   │   └── regularizers.py           # Load balancing, Router Z-loss, Static & Dynamic Orthogonality
│   ├── data/
│   │   ├── stage1_datamodule.py      # Multi-task instruction dataset with prompt masking
│   │   ├── episodic_collator.py      # Support/Query episodic collator with 30% self-supervised mode
│   │   └── benchmarks.py             # BBH, ARC-AGI, and GSM-Symbolic evaluators
│   ├── trainers/
│   │   ├── stage1_erm_trainer.py     # Stage 1 collaborative MoLE trainer with temp annealing
│   │   ├── stage2_fomaml_trainer.py  # Stage 2 FOMAML + Meta-SGD trainer with detached gradient trick
│   │   └── ttt_inference_engine.py   # Stateless test-time adaptation + KV-cache refresh + reset
│   └── analysis/
│       ├── plot_gate_trajectories.py # Plots N+1 gate shifts across TTT steps k=0..5
│       └── plot_meta_sgd_lrs.py      # Plots learned inner learning rates across Transformer layers
├── tests/
│   ├── test_shapes_and_surgery.py    # Verifies Stage 1 -> Stage 2 lossless transition at t=0
│   └── test_fomaml_gradients.py      # Verifies B_dyn, A_dyn, W_g, and eta all receive non-zero grads for k>=2
├── scripts/
│   └── run_phoenix.sbatch            # SLURM batch script for Georgia Tech PACE Phoenix cluster
├── run_pipeline.py                   # CLI entrypoint for Stage 1 -> Surgery -> Stage 2 -> Eval
└── requirements.txt                  # Python dependencies
```

---

## Setup & Execution on Georgia Tech Phoenix Cluster

### 1. Environment Setup
```bash
# Connect to GT Phoenix Cluster
ssh <username>@login-phoenix.pace.gatech.edu

# Load environment modules
module load anaconda3
module load cuda/12.1

# Create and activate conda environment
conda create -n mole python=3.10 -y
conda activate mole

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests on Compute Node / Interactive Session
```bash
# Request an interactive GPU session on Phoenix
salloc -N 1 -n 1 -c 8 --gres=gpu:1 --mem=32G -t 02:00:00 -q inferno

# Run unit tests to verify shapes, surgery, and FOMAML gradients
python -m pytest tests/
```

### 3. Step-by-Step CLI Execution
You can execute individual stages or run the full pipeline:

#### Step 3.1: Stage 1 Collaborative MoLE Stabilization
```bash
python run_pipeline.py --config configs/tier1_qwen05b_ablation.yaml --stage 1
```

#### Step 3.2: Gate Expansion Surgery ($N \to N+1$)
```bash
python run_pipeline.py --config configs/tier1_qwen05b_ablation.yaml --stage surgery
```

#### Step 3.3: Stage 2 FOMAML Meta-Training
```bash
python run_pipeline.py --config configs/tier1_qwen05b_ablation.yaml --stage 2
```

#### Step 3.4: Evaluation (5-Row Ablation Table)
```bash
python run_pipeline.py --config configs/tier1_qwen05b_ablation.yaml --stage eval
```

#### Step 3.5: Visualizations
```bash
python run_pipeline.py --config configs/tier1_qwen05b_ablation.yaml --stage plot
```

#### Step 3.6: End-to-End Execution
```bash
python run_pipeline.py --config configs/tier1_qwen05b_ablation.yaml --stage full
```

### 4. Batch Submission via SLURM
Submit the preconfigured job script:
```bash
sbatch scripts/run_phoenix.sbatch
```
Monitor your job:
```bash
squeue -u $USER
tail -f logs/slurm_*.out
```

---

## 5-Row Conference Ablation Suite
The evaluation pipeline generates the complete conference ablation table across held-out tasks (**BIG-Bench Hard**, **ARC-AGI**, and **GSM-Symbolic**):
1. **Base Pretrained Model (Few-Shot):** Frozen base model baseline.
2. **Static Collaborative MoLE (No TTT):** Stage 1 stabilized MoLE baseline.
3. **MoLE + Ad-Hoc TTT (No Meta-Training):** Naive test-time gradient descent directly on static experts.
4. **Single LoRA + TTT (Rank-Matched):** Standard single LoRA adapter with rank $N \times r$ subjected to TTT.
5. **Meta-MoLE-TTT (Full Proposed):** Two-Stage Meta-MoLE with gate expansion surgery, FOMAML meta-learning, and orthogonal scratchpad adaptation.
