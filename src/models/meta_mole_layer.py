"""
Meta-Gated Mixture of LoRA Experts (Meta-MoLE) Layer Implementation.
Supports Stage 1 (collaborative MoLE stabilization) and Stage 2 (bi-level FOMAML with orthogonal scratchpad).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterFreeRMSNorm(nn.Module):
    """
    Parameter-free Root Mean Square Normalization.
    Normalizes expert outputs before gating to prevent experts with larger weight
    Frobenius norms from dominating router gradients.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)


class MetaMoLELinear(nn.Module):
    """
    MetaMoLELinear replaces a standard nn.Linear layer in a Transformer model.
    It wraps the frozen base linear layer with:
      - N collaborative static LoRA experts (A_static, B_static)
      - An N-way router (Stage 1) or (N+1)-way router (Stage 2)
      - An optional (N+1)-th Dynamic Scratchpad LoRA (A_dyn, B_dyn)
      - Per-layer Meta-SGD step sizes (raw_eta_gate, raw_eta_A_dyn, raw_eta_B_dyn)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        router_temp: float = 1.0,
        is_ttt_layer: bool = True,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        self.dyn_scaling = 1.0
        self.router_temp = router_temp
        self.is_ttt_layer = is_ttt_layer
        self.stage = 1  # 1 = Pure Collaborative MoLE, 2 = Frozen Anchors + Meta-TTT Scratchpad

        # Freeze base projection layer
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # N Collaborative Static Experts: A_static [N, r, d_in], B_static [N, d_out, r]
        self.A_static = nn.Parameter(torch.empty(num_experts, rank, self.in_features, device=device, dtype=dtype))
        self.B_static = nn.Parameter(torch.empty(num_experts, self.out_features, rank, device=device, dtype=dtype))
        self.expert_norm = ParameterFreeRMSNorm()

        # Stage 1 Router: [N, d_in] and bias [N]
        self.router_weight = nn.Parameter(torch.empty(num_experts, self.in_features, device=device, dtype=dtype))
        self.router_bias = nn.Parameter(torch.zeros(num_experts, device=device, dtype=dtype))

        # Stage 2 Dynamic Scratchpad & Meta-SGD Learning Rates (activated during surgery)
        self.A_dyn = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype), requires_grad=False)
        self.B_dyn = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype), requires_grad=False)

        # Meta-SGD inverse-softplus parameters (softplus(-4.6) ~= 0.01)
        self.raw_eta_gate = nn.Parameter(torch.tensor(-4.6, device=device, dtype=dtype), requires_grad=False)
        self.raw_eta_A_dyn = nn.Parameter(torch.tensor(-5.3, device=device, dtype=dtype), requires_grad=False)
        self.raw_eta_B_dyn = nn.Parameter(torch.tensor(-4.6, device=device, dtype=dtype), requires_grad=False)

        # Holder for adapted parameters during inner-loop TTT steps
        self._adapted_params: Optional[Dict[str, torch.Tensor]] = None

        self._init_stage1_weights()

    def _init_stage1_weights(self):
        """Initializes static experts and router with symmetry-breaking orthogonal init."""
        device = self.A_static.device
        dtype = self.A_static.dtype
        flat_A = torch.empty(self.num_experts * self.rank, self.in_features, dtype=torch.float32)
        nn.init.orthogonal_(flat_A)
        with torch.no_grad():
            self.A_static.copy_(flat_A.view(self.num_experts, self.rank, self.in_features).to(device=device, dtype=dtype))
            # Small non-zero Gaussian initialization for B_static so router gets gradient at step 0
            b_init = torch.empty_like(self.B_static, dtype=torch.float32)
            nn.init.normal_(b_init, mean=0.0, std=1e-4)
            self.B_static.copy_(b_init.to(device=device, dtype=dtype))

            rw_init = torch.empty_like(self.router_weight, dtype=torch.float32)
            nn.init.kaiming_uniform_(rw_init, a=math.sqrt(5))
            self.router_weight.copy_(rw_init.to(device=device, dtype=dtype))
            self.router_bias.zero_()

            # Initialize A_dyn orthogonally as well
            flat_A_dyn = torch.empty(self.rank, self.in_features, dtype=torch.float32)
            nn.init.orthogonal_(flat_A_dyn)
            self.A_dyn.copy_(flat_A_dyn.to(device=device, dtype=dtype))

    @torch.no_grad()
    def expand_for_stage2_surgery(
        self,
        freeze_static_experts: bool = True,
        scratchpad_bias_init: float = -2.0,
    ):
        """
        Transitions layer from Stage 1 (N experts) to Stage 2 (N frozen anchors + 1 dynamic scratchpad).
        Guarantees zero forward-pass discrepancy at inner step t=0 because B_dyn = 0.
        """
        if self.stage == 2:
            return  # Already expanded

        self.stage = 2
        device, dtype = self.router_weight.device, self.router_weight.dtype

        if freeze_static_experts:
            self.A_static.requires_grad = False
            self.B_static.requires_grad = False

        # Expand router from N to N+1
        new_w = torch.zeros(self.num_experts + 1, self.in_features, device=device, dtype=dtype)
        new_b = torch.zeros(self.num_experts + 1, device=device, dtype=dtype)
        new_w[: self.num_experts].copy_(self.router_weight.data)
        new_b[: self.num_experts].copy_(self.router_bias.data)
        new_b[self.num_experts] = scratchpad_bias_init  # Small initial softmax weight probe

        self.router_weight = nn.Parameter(new_w, requires_grad=True)
        self.router_bias = nn.Parameter(new_b, requires_grad=True)

        # Activate Scratchpad & Meta-SGD step sizes if this layer participates in TTT
        if self.is_ttt_layer:
            flat_A_dyn = torch.empty(self.rank, self.in_features, dtype=torch.float32)
            nn.init.orthogonal_(flat_A_dyn)
            self.A_dyn.copy_(flat_A_dyn.to(device=device, dtype=dtype))
            self.B_dyn.zero_()
            if scratchpad_bias_init >= 0.0:
                # Paradigm A fast-weight memory mode: initialize higher write step size (softplus(-1.8) ~= 0.15)
                self.raw_eta_gate.fill_(-3.0)
                self.raw_eta_A_dyn.fill_(-3.5)
                self.raw_eta_B_dyn.fill_(-1.8)
            self.A_dyn.requires_grad = True
            self.B_dyn.requires_grad = True
            self.raw_eta_gate.requires_grad = True
            self.raw_eta_A_dyn.requires_grad = True
            self.raw_eta_B_dyn.requires_grad = True
        else:
            self.A_dyn.requires_grad = False
            self.B_dyn.requires_grad = False
            self.raw_eta_gate.requires_grad = False
            self.raw_eta_A_dyn.requires_grad = False
            self.raw_eta_B_dyn.requires_grad = False

    def get_adaptable_params(self) -> Dict[str, nn.Parameter]:
        """Returns the dictionary of parameters that can be adapted during TTT."""
        if self.stage == 1 or not self.is_ttt_layer:
            return {}
        return {
            "router_weight": self.router_weight,
            "router_bias": self.router_bias,
            "A_dyn": self.A_dyn,
            "B_dyn": self.B_dyn,
        }

    def get_meta_step_sizes(self) -> Dict[str, torch.Tensor]:
        """Returns Softplus-transformed positive step sizes for each adaptable group."""
        if not self.is_ttt_layer:
            return {}
        return {
            "router_weight": F.softplus(self.raw_eta_gate),
            "router_bias": F.softplus(self.raw_eta_gate),
            "A_dyn": F.softplus(self.raw_eta_A_dyn),
            "B_dyn": F.softplus(self.raw_eta_B_dyn),
        }

    def set_adapted_params(self, params: Optional[Dict[str, torch.Tensor]]):
        """Temporarily sets adapted parameters during inner-loop TTT steps."""
        self._adapted_params = params

    def reset_adapted_params(self):
        """Clears adapted parameters and resets B_dyn to zero for next episode."""
        self._adapted_params = None
        if self.stage == 2 and self.is_ttt_layer:
            with torch.no_grad():
                self.B_dyn.zero_()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            output: [..., d_out]
            logits: [..., N] or [..., N+1]
            gates:  [..., N] or [..., N+1]
        """
        base_out = self.base_layer(x)

        # Retrieve active router parameters
        if self._adapted_params is not None and "router_weight" in self._adapted_params:
            r_weight = self._adapted_params["router_weight"]
            r_bias = self._adapted_params["router_bias"]
        else:
            r_weight = self.router_weight
            r_bias = self.router_bias

        if r_weight.dtype != x.dtype:
            r_weight = r_weight.to(x.dtype)
            r_bias = r_bias.to(x.dtype)

        logits = F.linear(x, r_weight, r_bias) / self.router_temp
        gates = F.softmax(logits, dim=-1)

        # Batched static experts: x [..., d_in] -> [..., N, r] -> [..., N, d_out]
        A_static = self.A_static.to(x.dtype) if self.A_static.dtype != x.dtype else self.A_static
        B_static = self.B_static.to(x.dtype) if self.B_static.dtype != x.dtype else self.B_static

        static_hidden = torch.einsum("...d,nrd->...nr", x, A_static)
        static_out = torch.einsum("...nr,ndr->...nd", static_hidden, B_static)
        static_out = self.expert_norm(static_out)

        if self.stage == 1:
            routed_static = torch.einsum("...nd,...n->...d", static_out, gates)
            return base_out + self.scaling * routed_static, logits, gates

        # Stage 2: Route first N gates to static experts, (N+1)-th gate to dynamic scratchpad
        gates_static = gates[..., : self.num_experts]
        gate_dyn = gates[..., self.num_experts :]  # [..., 1]

        routed_static = torch.einsum("...nd,...n->...d", static_out, gates_static)

        # Dynamic Scratchpad
        if self._adapted_params is not None and "A_dyn" in self._adapted_params:
            curr_A_dyn = self._adapted_params["A_dyn"]
            curr_B_dyn = self._adapted_params["B_dyn"]
        else:
            curr_A_dyn = self.A_dyn
            curr_B_dyn = self.B_dyn

        curr_A_dyn = curr_A_dyn.to(x.dtype) if curr_A_dyn.dtype != x.dtype else curr_A_dyn
        curr_B_dyn = curr_B_dyn.to(x.dtype) if curr_B_dyn.dtype != x.dtype else curr_B_dyn

        dyn_hidden = self.expert_norm(F.linear(x, curr_A_dyn))  # [..., r] unit-RMS key activation
        dyn_out = F.linear(dyn_hidden, curr_B_dyn)  # [..., d_out]
        routed_dyn = gate_dyn * dyn_out

        total_out = base_out + self.scaling * routed_static + self.dyn_scaling * routed_dyn
        return total_out, logits, gates
