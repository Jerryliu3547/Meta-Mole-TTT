# Implementation Plan: Meta-Gated Mixture of LoRA Experts with Orthogonal Test-Time Scratchpad (Meta-MoLE-TTT)

## 1. Executive Summary & Core Thesis
**Target Venue:** Top-Tier CS/ML Conferences (ICML / NeurIPS / ICLR / ACL)  
**Core Scientific Claim:**  
In a standard Mixture of LoRA Experts (MoLE) trained end-to-end from random initialization, the $N$ LoRA experts do not partition into human-labeled buckets; rather, they self-organize into **collaborative, fine-grained feature extractors** that span a shared representation subspace. However, when faced with out-of-distribution (OOD) tasks at inference time, static interpolation of these $N$ collaborative experts fails to capture novel, instance-specific patterns, while naive Test-Time Training (TTT) of all experts causes catastrophic forgetting and destroys their stabilized collaborative equilibrium.

We propose **Meta-MoLE-TTT**, a decoupled plasticity-stability architecture trained via a strict **Two-Stage Curriculum**:
1. **Stage 1 (Collaborative MoLE Stabilization — No Scratchpad):** We train $N$ collaborative LoRA experts $\{A_i, B_i\}_{i=1}^N$ and an $N$-way gating router $W_g^{(1:N)}$ end-to-end via Empirical Risk Minimization (ERM) with load-balancing, router Z-loss, and subspace orthogonality regularization until the collaborative feature basis is fully stabilized.
2. **Stage 2 (Gate Expansion Surgery & Bi-Level Meta-TTT Scratchpad):** We **freeze** the converged $N$ collaborative experts (preserving our exact Static MoLE baseline) and perform **$N \to N+1$ Gate Expansion Surgery** to attach a $(N+1)$-th **Dynamic Scratchpad LoRA** $(A_{\text{dyn}}, B_{\text{dyn}})$. Using First-Order MAML (FOMAML) + Meta-SGD, we meta-train the expanded router $W_g^{(1:N+1)}$, the scratchpad basis $A_{\text{dyn}}^{(0)}$, and per-layer inner learning rates $\boldsymbol{\eta}$ so that 1–3 steps of test-time gradient descent rapidly absorb orthogonal, instance-specific residuals without corrupting the frozen collaborative anchor experts.

---

## 2. Mathematical Formulation & Architecture Specification

### 2.1 Layer-Level Forward Pass (`MetaMoLELinear`)
Applied to target linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`) of a frozen pretrained Transformer backbone.

Given frozen base weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and input hidden state $x \in \mathbb{R}^{B \times S \times d_{\text{in}}}$:

#### Phase A: Stage 1 Forward Pass ($N$ Collaborative Experts Only)
$$h_{\text{Stage1}} = W_0 x + \frac{\alpha}{r} \sum_{i=1}^{N} g_i(x) \cdot \text{RMSNorm}\left(B_i A_i x\right)$$
where $A_i \in \mathbb{R}^{r \times d_{\text{in}}}$, $B_i \in \mathbb{R}^{d_{\text{out}} \times r}$, and the $N$-way router probabilities are:
$$g(x) = \text{Softmax}\left(\frac{W_g^{(1:N)} x + b_g^{(1:N)}}{\tau}\right), \quad W_g^{(1:N)} \in \mathbb{R}^{N \times d_{\text{in}}}$$
*(Crucial detail: Applying parameter-free $\text{RMSNorm}$ to each expert's output before gating prevents an expert with larger weight Frobenius norm from dominating router gradients).*

#### Phase B: Stage 2 & Inference Forward Pass ($N$ Frozen Anchors + 1 Dynamic Scratchpad)
After $N \to N+1$ gate surgery, the router outputs $N+1$ probabilities $g(x) \in \Delta^{N+1}$:
$$h_{\text{Stage2}} = W_0 x + \frac{\alpha}{r} \sum_{i=1}^{N} g_i(x) \cdot \text{RMSNorm}\left(\underbrace{B_i A_i}_{\text{Frozen}} x\right) + \frac{\alpha}{r} g_{N+1}(x) \cdot \left(B_{\text{dyn}} A_{\text{dyn}} x\right)$$
*(Note: Do NOT apply RMSNorm to the dynamic scratchpad output $B_{\text{dyn}} A_{\text{dyn}} x$, because at inner-loop step $t=0$, $B_{\text{dyn}}^{(0)} = 0$, which would divide by zero).*

---

### 2.2 Smooth $N \to N+1$ Gate Expansion Surgery
Transitioning from Stage 1 ($W_g \in \mathbb{R}^{N \times d_{\text{in}}}$) to Stage 2 ($W_g \in \mathbb{R}^{(N+1) \times d_{\text{in}}}$) must not cause an initial loss spike or destroy the stabilized expert routing:
1. **Copy Converged Weights:** Copy $W_g^{(1:N)}$ and $b_g^{(1:N)}$ directly from the Stage 1 checkpoint into the first $N$ rows of the expanded router.
2. **Zero-Weight + Negative-Bias Init for Row $N+1$:**
   $$w_{g, N+1} = \mathbf{0} \in \mathbb{R}^{d_{\text{in}}}, \quad b_{g, N+1} = -2.0$$
   - **Why $b_{g, N+1} = -2.0$?** At inner-loop step $t=0$, $B_{\text{dyn}}^{(0)} = 0$. Having a small non-zero softmax probability $g_{N+1}(x) \approx 0.05\text{--}0.10$ ensures non-zero gradients flow into $B_{\text{dyn}}$ at step $t=0$ (avoiding the zero-gradient saddle point), while keeping $\sim 90\%+$ of the probability mass on the stabilized $N$ experts.

---

## 3. Training Curriculum & Loss Functions

### Parameter Freeze/Unfreeze Schedule Across Stages

| Parameter Group | Stage 1: Pure MoLE Stabilization (ERM) | Stage 2: Meta-TTT Scratchpad (Bi-Level FOMAML) | Stage 2.5 (Optional): Co-Adaptation (Final 10%) | Test-Time Inference (TTT Inner Loop) |
| :--- | :--- | :--- | :--- | :--- |
| **Base LLM ($\theta_{\text{base}}$)** | Frozen | Frozen | Frozen | Frozen |
| **Collaborative Experts ($\{A_i, B_i\}_{i=1}^N$)** | **Trainable** ($\text{lr}=2\text{e-}4$) | **Frozen** (Stationary Basis) | **Outer-Loop Only** ($\text{lr}=1\text{e-}5$) | Frozen |
| **Router Gate ($W_g, b_g$)** | **Trainable** ($N$-way, $\text{lr}=5\text{e-}4$) | **Meta-Trained** ($N+1$-way, Inner & Outer) | **Meta-Trained** (Inner & Outer) | **Adapted** ($k$ steps) |
| **Scratchpad Down-Proj ($A_{\text{dyn}}^{(0)}$)** | *Not instantiated* | **Meta-Trained** (Outer learns init; Inner adapts) | **Meta-Trained** (Inner & Outer) | **Adapted** ($k$ steps) |
| **Scratchpad Up-Proj ($B_{\text{dyn}}$)** | *Not instantiated* | **Zero-Reset** per episode ($B_{\text{dyn}}^{(0)}=0$); Inner adapts | **Zero-Reset** per episode; Inner adapts | **Adapted** (Starts at $0$, $k$ steps) |
| **Meta-SGD Step Sizes ($\boldsymbol{\eta}_{\text{gate}}, \boldsymbol{\eta}_{\text{dyn}}$)** | *Not instantiated* | **Meta-Trained** (Outer loop only, $\text{lr}=1\text{e-}3$) | **Meta-Trained** (Outer loop only) | Fixed learned scalars |

---

### Stage 1 Objective: End-to-End Collaborative MoLE Stabilization
Trained on a unified multi-task instruction/reasoning mixture so experts self-organize as fine-grained collaborative feature extractors:
$$\mathcal{L}_{\text{Stage1}} = \mathcal{L}_{\text{CE}}(x) + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{z} \mathcal{L}_{z} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho\_static}}$$

1. **Symmetry-Breaking Orthogonal Init:** Initialize stacked $A_{1:N} \in \mathbb{R}^{(N \cdot r) \times d_{\text{in}}}$ using `torch.nn.init.orthogonal_` before reshaping to $(N, r, d_{\text{in}})$, and initialize $B_{1:N} \sim \mathcal{N}(0, 10^{-4})$ (near-zero so initial forward pass matches base LLM, but non-zero so router gets gradient at Step 0).
2. **Load-Balancing Loss ($\lambda_{\text{bal}} = 0.01$):**
   $$\mathcal{L}_{\text{balance}} = N \sum_{i=1}^{N} f_i \cdot P_i$$
   where $f_i$ is the fraction of tokens where expert $i$ is top-1, and $P_i$ is the mean router probability for expert $i$.
3. **Router Z-Loss ($\lambda_z = 10^{-3}$):** Penalizes large logits $z = W_g x + b_g$ to prevent softmax saturation (critical so Stage 2 inner-loop gradients don't vanish):
   $$\mathcal{L}_z = \frac{1}{B \cdot S} \sum_{b, s} \left(\log \sum_{i=1}^N \exp(z_{b,s,i})\right)^2$$
4. **Pairwise Subspace Orthogonality ($\lambda_{\text{ortho}} = 0.05$):** Forces collaborative experts to span distinct feature directions:
   $$\mathcal{L}_{\text{ortho\_static}} = \sum_{1 \le i < j \le N} \left\Vert{} \hat{A}_i \hat{A}_j^\top \right\Vert{}_F^2, \quad \hat{A}_i = \frac{A_i}{\Vert{}A_i\Vert{}_F + \epsilon}$$

---

### Stage 2 Objective: Bi-Level FOMAML + Meta-SGD for Gate & Scratchpad
Each training task/episode $\mathcal{T}_m$ is partitioned into a **Support Context** $S_m$ (few-shot demonstrations or unlabeled test prompt prefix) and a **Query Target** $Q_m$ (target reasoning trace and answer).

Let the adaptable parameter tuple at episode start be $\phi^{(0)} = \{W_g, b_g, A_{\text{dyn}}^{(0)}, B_{\text{dyn}}^{(0)} = \mathbf{0}\}$.

#### Inner Loop ($k \in \{1, 2, 3\}$ steps on Support Set $S_m$)
At each inner step $t = 0, \dots, k-1$, compute the self-supervised/support loss with an **Orthogonal Residual Constraint** and **KL Trust Region**:
$$\mathcal{L}_{\text{inner}}(S_m; \phi^{(t)}) = \mathcal{L}_{\text{NTP}}(S_m; \phi^{(t)}) + \gamma_{\text{KL}} D_{\text{KL}}\big(g(S_m; W_g^{(t)}) \parallel g(S_m; W_g^{(0)})\big) + \gamma_{\text{ortho}} \sum_{i=1}^{N} \left\Vert{} A_{\text{dyn}}^{(t)} \hat{A}_i^\top \right\Vert{}_F^2$$
Update adaptable parameters using learnable per-layer, per-group Meta-SGD step sizes $\boldsymbol{\eta} = \{\eta_{\text{gate}}^{(l)}, \eta_{A_{\text{dyn}}}^{(l)}, \eta_{B_{\text{dyn}}}^{(l)}\}_{l=1}^L$ (clamped via $\text{Softplus}(\tilde{\eta})$ to stay strictly positive):
$$\phi_m^{(t+1)} = \phi_m^{(t)} - \text{Softplus}(\boldsymbol{\eta}) \odot \text{clip\_norm}\left(\nabla_{\phi} \mathcal{L}_{\text{inner}}\big(S_m; \phi_m^{(t)}\big), 1.0\right).\text{detach}()$$
*(Note: `.detach()` on the gradient graph implements First-Order MAML [FOMAML], avoiding Hessian-vector OOMs while allowing direct outer-loop gradients to flow into $\phi^{(0)}$ and $\boldsymbol{\eta}$ via the final query evaluation).*

**Wait — How does Meta-SGD learn $\boldsymbol{\eta}$ if gradients are detached?**
To allow outer-loop gradients to update $\boldsymbol{\eta}$ without computing full second-order Hessians over the LLM:
- Detach the gradient tensor $\mathbf{g}_t = \text{stop\_grad}\big(\nabla_{\phi} \mathcal{L}_{\text{inner}}(S_m; \phi_m^{(t)})\big)$, **but do NOT detach the multiplication by $\text{Softplus}(\boldsymbol{\eta})$**:
  $$\phi_m^{(t+1)} = \phi_m^{(t)} - \text{Softplus}(\boldsymbol{\eta}) \odot \mathbf{g}_t$$
  By the chain rule on the query set $Q_m$:
  $$\frac{\partial \mathcal{L}_{\text{outer}}(Q_m; \phi_m^{(k)})}{\partial \boldsymbol{\eta}} = -\sum_{t=0}^{k-1} \left\langle \nabla_{\phi^{(k)}} \mathcal{L}_{\text{outer}}(Q_m; \phi_m^{(k)}), \, \mathbf{g}_t \right\rangle \cdot \sigma(\tilde{\boldsymbol{\eta}})$$
  This exact first-order inner-product rule updates $\boldsymbol{\eta}$ whenever the inner gradient $\mathbf{g}_t$ aligns with the outer query gradient, requiring **zero second-order memory overhead**!

#### Outer Loop (Meta-Update on Query Set $Q_m$)
Minimize the query cross-entropy loss plus load-balancing regularization over meta-batches of episodes:
$$\min_{W_g^{(0)}, b_g^{(0)}, A_{\text{dyn}}^{(0)}, \tilde{\boldsymbol{\eta}}} \mathbb{E}_{\mathcal{T}_m} \left[ \mathcal{L}_{\text{CE}}\left(Q_m \mid S_m; \theta_{\text{base}}, \{A_i, B_i\}_{i=1}^N, \phi_m^{(k)}\right) + \lambda_{\text{KL}} D_{\text{KL}}\big(g(Q_m; W_{g,m}^{(k)}) \parallel g(Q_m; W_g^{(0)})\big) \right]$$

---

## 4. Critical Engineering Edge Cases (Must Implement)

1. **The $B_{\text{dyn}}^{(0)} = 0$ Zero-Gradient Trap at Inner Step $t=0$:**
   - Because output is $B_{\text{dyn}} A_{\text{dyn}} x$ and $B_{\text{dyn}}^{(0)} = 0$, at $t=0$:
     - $\nabla_{B_{\text{dyn}}} \mathcal{L} \neq 0$ (since $A_{\text{dyn}}^{(0)} x \neq 0$ and $g_{N+1}(x) > 0$).
     - $\nabla_{A_{\text{dyn}}} \mathcal{L} = B_{\text{dyn}}^{(0)\top} (\dots) = 0$!
     - $\nabla_{w_{g, N+1}} \mathcal{L} = 0$ (since the $(N+1)$-th expert outputs zero at $t=0$, changing its gate weight doesn't change the layer output at $t=0$!).
   - **Mandatory Solution:** 
     - **Option A (Minimum $k \ge 2$ Inner Steps):** At $t=0$, $B_{\text{dyn}}^{(1)}$ becomes non-zero. At $t=1$, both $A_{\text{dyn}}$ and $w_{g, N+1}$ immediately receive rich, non-zero gradients! Therefore, **always set inner steps $k \ge 2$** (default $k=3$).
     - **Option B (Auxiliary Cosine Routing Alignment Loss at $t=0$):** Add a small auxiliary routing alignment term in $\mathcal{L}_{\text{inner}}$ so $w_{g, N+1}$ moves at $t=0$ whenever the hidden state residual has high norm.
2. **KV-Cache Invalidation After TTT Inner Loop:**
   - During the $k$ inner-loop steps on the support/prompt tokens $S_m$, the weights of $\phi_m^{(k)}$ change at every layer.
   - **Never reuse a KV-cache computed before or during the inner loop.** Once $\phi_m^{(k)}$ is finalized after step $k$, run a single clean forward pass (prefill) over $S_m$ with `use_cache=True` using $\phi_m^{(k)}$, and then autoregressively decode the answer.
3. **Batched Vectorized Execution (`torch.einsum`):**
   - Never use a Python `for` loop over $N$ experts. Stack static experts into `A_static: [N, r, d_in]` and `B_static: [N, d_out, r]` as registered non-trainable buffers in Stage 2, and execute via `torch.einsum`.
4. **Selective Layer Placement (Upper $L/2$ Layers Only Option):**
   - Provide a config flag `ttt_layer_indices: "upper_half" | "all"`. Applying TTT adaptation only to the upper $L/2$ layers reduces inner-loop backpropagation activation memory and latency by $\sim 48\%$ because gradients do not need to propagate back to layer 0.

---

## 5. Model Scaling Ladder (For Fast Debugging to Publication)

Antigravity must support switching models via YAML config without changing code:

| Tier | Model ID | Purpose | Hardware / VRAM | Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0 (Unit Test)** | `HuggingFaceTB/SmolLM2-135M-Instruct` | Verify Stage 1 $\to$ Stage 2 surgery, FOMAML gradients, and $t=0 \to 2$ non-zero gradient flow in < 2 mins. | CPU or any 4GB+ GPU | $N=4, r=4, k=2$ |
| **Tier 1 (Dev / Ablation)** | `Qwen/Qwen2.5-0.5B-Instruct` or `1.5B` | Rapid hyperparameter sweeps ($\lambda_{\text{ortho}}, \gamma_{\text{KL}}, k$) and verifying Stage 2 beats Stage 1. | Single 16GB–24GB GPU | $N=6, r=8, k=3$ |
| **Tier 2 (Paper Main)** | `meta-llama/Llama-3.1-8B-Instruct` & `Qwen/Qwen2.5-7B-Instruct` | Main conference tables across two distinct model families. | 1x A100/H100 (40GB/80GB) | $N=8, r=8, k=3$ |

---

## 6. Datasets & Experimental Protocol

### 6.1 Stage 1 & Stage 2 Training Data (Unified Multi-Task Mixture)
- **Dataset:** A 50,000-sample stratified mixture drawn from **FLAN-v2 / Tulu-3 Mixture** (spanning symbolic math, Python coding, logical deduction, reading comprehension, and table QA).
- **Stage 1 Formatting:** Standard causal language modeling (instruction $\to$ response) with batch size 32.
- **Stage 2 Episodic Formatting (`EpisodicTaskCollator`):** Groups 3 to 5 examples from the same task cluster:
  - First $M-1$ examples form the **Support Set $S_m$** (used for inner-loop TTT steps).
  - Final example forms the **Query Set $Q_m$** (used for outer-loop FOMAML meta-update).
  - *Self-Supervised Zero-Shot Fallback Mode:* For 30% of Stage 2 episodes, $S_m$ is simply the **unlabeled input prompt/context** of $Q_m$ itself (training the inner loop to adapt using only self-supervised prompt perplexity + router entropy minimization when few-shot examples aren't available).

### 6.2 Evaluation Benchmarks (Held-Out OOD Tasks)
1. **BIG-Bench Hard (BBH - 27 Tasks):** Hold out 10 unseen reasoning tasks (e.g., *Boolean Expressions, Causal Judgment, Dyck Languages, Logical Deduction, Tracking Shuffled Objects*) never seen during Stage 1 or 2.
2. **ARC-AGI (Abstraction & Reasoning Corpus - Text/Grid Format):** Pure test-time adaptation benchmark where each test item provides 2–4 demonstration pairs ($S_m$) and 1 test input grid ($Q_m$).
3. **GSM-Symbolic / LiveCodeBench Hybrids:** Tests stability on math and algorithmic reasoning under distribution shift.

---

## 7. Repository Directory Structure

```text
meta_mole_ttt/
├── configs/
│   ├── tier0_smollm_debug.yaml       # Fast 2-minute pipeline verification
│   ├── tier1_qwen05b_ablation.yaml   # Fast ablation sweeps
│   └── tier2_llama8b_main.yaml       # Full conference experiment config
├── src/
│   ├── models/
│   │   ├── meta_mole_layer.py        # MetaMoLELinear with Stage 1 & Stage 2 modes + N->N+1 surgery
│   │   ├── model_wrapper.py          # Injects MetaMoLELinear into HF PreTrainedModel + upper-half layer targeting
│   │   └── regularizers.py           # Switch balance loss, Router Z-loss, Static & Dynamic Orthogonality losses
│   ├── data/
│   │   ├── stage1_datamodule.py      # Unified multi-task instruction stream for Stage 1
│   │   ├── episodic_collator.py      # Support/Query episodic packer for Stage 2 FOMAML
│   │   └── benchmarks.py             # BBH, ARC-AGI, and GSM-Symbolic evaluators
│   ├── trainers/
│   │   ├── stage1_erm_trainer.py     # Stage 1 collaborative MoLE trainer
│   │   ├── stage2_fomaml_trainer.py  # Stage 2 FOMAML + Meta-SGD trainer with detached inner-product trick
│   │   └── ttt_inference_engine.py   # Stateless test-time adaptation + KV-cache refresh + reset
│   └── analysis/
│       ├── plot_gate_trajectories.py # Plots N+1 gate shifts across TTT steps k=0..5
│       └── plot_meta_sgd_lrs.py      # Plots learned inner learning rates across Transformer layers
├── tests/
│   ├── test_shapes_and_surgery.py    # Verifies Stage 1 -> Stage 2 lossless transition at t=0
│   └── test_fomaml_gradients.py      # Verifies B_dyn, A_dyn, W_g, and eta all receive non-zero grads for k>=2
└── run_pipeline.py                   # CLI entrypoint for Stage 1 -> Surgery -> Stage 2 -> Eval
```

---

## 8. Core Reference Implementation Spec (`MetaMoLELinear` & Surgery)

Antigravity should use the following exact PyTorch module design for `src/models/meta_mole_layer.py`:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterFreeRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

class MetaMoLELinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        router_temp: float = 1.0,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        self.router_temp = router_temp
        self.stage = 1  # 1 = Pure Collaborative MoLE, 2 = Frozen Anchors + Meta-TTT Scratchpad

        # Freeze base projection
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # N Collaborative Static Experts: A_static [N, r, d_in], B_static [N, d_out, r]
        self.A_static = nn.Parameter(torch.empty(num_experts, rank, self.in_features))
        self.B_static = nn.Parameter(torch.empty(num_experts, self.out_features, rank))
        self.expert_norm = ParameterFreeRMSNorm()

        # Stage 1 Router: [N, d_in]
        self.router_weight = nn.Parameter(torch.empty(num_experts, self.in_features))
        self.router_bias = nn.Parameter(torch.zeros(num_experts))

        # Stage 2 Dynamic Scratchpad & Meta-SGD Learning Rates (initialized during surgery)
        self.A_dyn = nn.Parameter(torch.empty(rank, self.in_features), requires_grad=False)
        self.B_dyn = nn.Parameter(torch.zeros(self.out_features, rank), requires_grad=False)
        
        # Meta-SGD inverse-softplus parameters (softplus(-4.6) ~= 0.01)
        self.raw_eta_gate = nn.Parameter(torch.tensor(-4.6), requires_grad=False)
        self.raw_eta_A_dyn = nn.Parameter(torch.tensor(-5.3), requires_grad=False)
        self.raw_eta_B_dyn = nn.Parameter(torch.tensor(-4.6), requires_grad=False)

        self._init_stage1_weights()

    def _init_stage1_weights(self):
        # Orthogonal init across all N*r rows to break symmetry
        flat_A = torch.empty(self.num_experts * self.rank, self.in_features)
        nn.init.orthogonal_(flat_A)
        with torch.no_grad():
            self.A_static.copy_(flat_A.view(self.num_experts, self.rank, self.in_features))
            # Small non-zero init for B_static so router gets gradient at step 0
            nn.init.normal_(self.B_static, mean=0.0, std=1e-4)
            nn.init.kaiming_uniform_(self.router_weight, a=math.sqrt(5))
            nn.init.zeros_(self.router_bias)
            nn.init.orthogonal_(self.A_dyn)

    @torch.no_grad()
    def expand_for_stage2_surgery(self, freeze_static_experts: bool = True, scratchpad_bias_init: float = -2.0):
        """
        Transitions layer from Stage 1 (N experts) to Stage 2 (N frozen anchors + 1 dynamic scratchpad).
        Guarantees zero forward-pass discrepancy at inner step t=0 because B_dyn = 0.
        """
        assert self.stage == 1, "Layer is already in Stage 2."
        self.stage = 2
        device, dtype = self.router_weight.device, self.router_weight.dtype

        if freeze_static_experts:
            self.A_static.requires_grad = False
            self.B_static.requires_grad = False

        # Expand router from N to N+1
        new_w = torch.zeros(self.num_experts + 1, self.in_features, device=device, dtype=dtype)
        new_b = torch.zeros(self.num_experts + 1, device=device, dtype=dtype)
        new_w[:self.num_experts].copy_(self.router_weight.data)
        new_b[:self.num_experts].copy_(self.router_bias.data)
        new_b[self.num_experts] = scratchpad_bias_init  # ~8-10% initial softmax probe weight

        self.router_weight = nn.Parameter(new_w, requires_grad=True)
        self.router_bias = nn.Parameter(new_b, requires_grad=True)

        # Activate Scratchpad & Meta-SGD step sizes
        nn.init.orthogonal_(self.A_dyn)
        self.B_dyn.zero_()
        self.A_dyn.requires_grad = True
        self.B_dyn.requires_grad = True
        self.raw_eta_gate.requires_grad = True
        self.raw_eta_A_dyn.requires_grad = True
        self.raw_eta_B_dyn.requires_grad = True

    def forward(self, x: torch.Tensor):
        base_out = self.base_layer(x)  # [B, S, d_out]
        logits = (F.linear(x, self.router_weight, self.router_bias)) / self.router_temp
        gates = F.softmax(logits, dim=-1)  # [B, S, N] or [B, S, N+1]

        # Batched static experts: x [B, S, d_in] -> [B, S, N, r] -> [B, S, N, d_out]
        static_hidden = torch.einsum("bsd,nrd->bsnr", x, self.A_static)
        static_out = torch.einsum("bsnr,ndr->bsnd", static_hidden, self.B_static)
        static_out = self.expert_norm(static_out)

        if self.stage == 1:
            routed_static = torch.einsum("bsnd,bsn->bsd", static_out, gates)
            return base_out + self.scaling * routed_static, logits, gates

        # Stage 2: Route first N gates to static experts, (N+1)-th gate to dynamic scratchpad
        gates_static = gates[..., :self.num_experts]
        gate_dyn = gates[..., self.num_experts:]  # [B, S, 1]

        routed_static = torch.einsum("bsnd,bsn->bsd", static_out, gates_static)
        dyn_hidden = F.linear(x, self.A_dyn)       # [B, S, r]
        dyn_out = F.linear(dyn_hidden, self.B_dyn) # [B, S, d_out]
        routed_dyn = gate_dyn * dyn_out

        return base_out + self.scaling * (routed_static + routed_dyn), logits, gates
```

---

## 9. Step-by-Step Execution Checklist for Antigravity

Execute and verify each milestone sequentially before moving to the next:

- [ ] **Milestone 1: Core Layer & Unit Tests (`tests/test_shapes_and_surgery.py`)**
  - Implement `MetaMoLELinear` and `regularizers.py`.
  - Write and run a unit test verifying:
    1. Stage 1 forward pass shapes and non-zero gradients on `A_static`, `B_static`, and `router_weight`.
    2. `expand_for_stage2_surgery()` runs cleanly.
    3. In Stage 2 with $k=2$ inner steps, `B_dyn` gets non-zero grad at $t=0$, and `A_dyn`, `router_weight[N]`, and `raw_eta_*` all get non-zero outer-loop gradients after $t=1$.
- [ ] **Milestone 2: Stage 1 End-to-End Collaborative MoLE Trainer**
  - Build `stage1_erm_trainer.py` with Cosine LR schedule, router temperature annealing ($\tau: 1.5 \to 0.7$), load-balancing loss, Z-loss, and static orthogonality loss.
  - Run Tier-0 (`SmolLM2-135M`) for 200 steps and verify expert utilization entropy remains $> 0.85 \times \log(N)$ (no expert collapse).
- [ ] **Milestone 3: Stage 2 FOMAML + Meta-SGD Bi-Level Trainer**
  - Implement `stage2_fomaml_trainer.py` loading the Stage 1 checkpoint, performing `expand_for_stage2_surgery()`, and executing the $k$-step inner loop + outer query update.
  - Ensure fast episodic reset (`B_dyn.zero_()`, restoring `A_dyn`, `router_weight`, `router_bias` from outer-loop meta-parameters) between episodes.
- [ ] **Milestone 4: Stateless TTT Inference Engine & KV-Cache Refresh**
  - Implement `ttt_inference_engine.py` supporting both **Few-Shot Support TTT** and **Zero-Shot Self-Supervised Prompt TTT**.
  - Verify that after $k$ inner steps, the model performs a clean prefill forward pass to populate the KV cache before calling `.generate()`.
- [ ] **Milestone 5: Automated Ablation & Plotting Suite**
  - Generate the 5-row conference ablation table automatically:
    1. Base Model (Zero/Few-Shot)
    2. Stage 1 Static Collaborative MoLE (No TTT)
    3. Stage 1 MoLE + Ad-Hoc TTT (Without Stage 2 Meta-Learning)
    4. Single Rank-Matched LoRA + TTT
    5. **Full Meta-MoLE-TTT (Stage 1 + Stage 2 FOMAML Scratchpad)**