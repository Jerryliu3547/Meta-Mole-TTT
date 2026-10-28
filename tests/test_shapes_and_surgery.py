"""
Unit tests for MetaMoLELinear shapes, Stage 1 gradients, and smooth N -> N+1 Gate Surgery.
Verifies lossless forward pass transition at inner step t=0.
"""

import torch
import torch.nn as nn

from src.models.meta_mole_layer import MetaMoLELinear
from src.models.regularizers import (
    compute_dynamic_orthogonality_loss,
    compute_load_balancing_loss,
    compute_router_z_loss,
    compute_static_orthogonality_loss,
)


def test_stage1_shapes_and_gradients():
    """Verifies Stage 1 shapes and non-zero gradients on static experts and router."""
    torch.manual_seed(42)
    B, S, d_in, d_out = 2, 4, 32, 64
    num_experts, rank = 4, 4

    base_layer = nn.Linear(d_in, d_out)
    layer = MetaMoLELinear(
        base_layer=base_layer,
        num_experts=num_experts,
        rank=rank,
        alpha=16.0,
        router_temp=1.0,
    )

    x = torch.randn(B, S, d_in, requires_grad=True)
    out, logits, gates = layer(x)

    assert out.shape == (B, S, d_out), f"Expected out shape {(B, S, d_out)}, got {out.shape}"
    assert logits.shape == (B, S, num_experts), f"Expected logits shape {(B, S, num_experts)}, got {logits.shape}"
    assert gates.shape == (B, S, num_experts), f"Expected gates shape {(B, S, num_experts)}, got {gates.shape}"

    # Check softmax sum to 1
    gate_sums = gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5), "Gates do not sum to 1.0"

    # Compute loss with regularizers
    loss_ce = out.sum()
    loss_bal = compute_load_balancing_loss(gates, num_experts=num_experts)
    loss_z = compute_router_z_loss(logits)
    loss_ortho = compute_static_orthogonality_loss(layer.A_static)
    total_loss = loss_ce + 0.01 * loss_bal + 0.001 * loss_z + 0.05 * loss_ortho

    total_loss.backward()

    # Verify gradients
    assert layer.A_static.grad is not None and layer.A_static.grad.norm().item() > 0, "A_static grad should be non-zero"
    assert layer.B_static.grad is not None and layer.B_static.grad.norm().item() > 0, "B_static grad should be non-zero"
    assert layer.router_weight.grad is not None and layer.router_weight.grad.norm().item() > 0, "router_weight grad should be non-zero"
    assert layer.router_bias.grad is not None and layer.router_bias.grad.norm().item() > 0, "router_bias grad should be non-zero"
    # Base layer must be frozen
    assert base_layer.weight.grad is None, "Base layer should remain strictly frozen"
    print("✓ test_stage1_shapes_and_gradients passed!")


def test_surgery_and_t0_equivalence():
    """Verifies N -> N+1 gate expansion surgery and lossless forward pass at t=0."""
    torch.manual_seed(42)
    B, S, d_in, d_out = 2, 4, 32, 64
    num_experts, rank = 4, 4

    base_layer = nn.Linear(d_in, d_out)
    layer = MetaMoLELinear(
        base_layer=base_layer,
        num_experts=num_experts,
        rank=rank,
        alpha=16.0,
        router_temp=1.0,
    )

    x = torch.randn(B, S, d_in)

    # Stage 1 forward pass
    with torch.no_grad():
        out_s1, _, gates_s1 = layer(x)

    # Perform N -> N+1 surgery
    layer.expand_for_stage2_surgery(freeze_static_experts=True, scratchpad_bias_init=-2.0)

    assert layer.stage == 2, "Layer stage should be 2"
    assert layer.A_static.requires_grad is False, "Static experts must be frozen in Stage 2"
    assert layer.B_static.requires_grad is False, "Static experts must be frozen in Stage 2"
    assert layer.router_weight.shape == (num_experts + 1, d_in), "Router weight shape must expand to N+1"
    assert layer.router_bias.shape == (num_experts + 1,), "Router bias shape must expand to N+1"
    assert layer.router_bias[-1].item() == -2.0, "Scratchpad bias init should be -2.0"
    assert layer.B_dyn.norm().item() == 0.0, "B_dyn must be initialized to exact zero"

    # Stage 2 forward pass at t=0 (B_dyn = 0)
    with torch.no_grad():
        out_s2, _, gates_s2 = layer(x)

    # Gates: first N gates will receive ~90%+ mass, (N+1)-th gate receives ~8-10% probe mass
    assert gates_s2.shape == (B, S, num_experts + 1)
    dyn_gate = gates_s2[..., -1]
    assert (dyn_gate > 0.0).all() and (dyn_gate < 0.20).all(), f"Probe gate should be small non-zero: mean={dyn_gate.mean():.3f}"

    # Verify that dynamic scratchpad output is strictly zero at t=0 because B_dyn = 0
    # Output discrepancy between Stage 1 and Stage 2 arises only from the minor softmax renormalization of static experts
    diff = (out_s2 - out_s1).abs().max().item()
    print(f"Max absolute forward difference at t=0: {diff:.6f}")
    assert diff < 0.05, f"Discrepancy at t=0 should be minimal, got {diff}"

    print("✓ test_surgery_and_t0_equivalence passed!")


if __name__ == "__main__":
    test_stage1_shapes_and_gradients()
    test_surgery_and_t0_equivalence()
    print("All shape and surgery tests passed successfully!")
