"""
Unit tests for Stage 2 FOMAML and Meta-SGD Gradient Flow.
Verifies:
  1. Inner step t=0: B_dyn gets non-zero gradient, resolving the zero-gradient trap.
  2. Inner step t=1 (k >= 2): A_dyn and row N+1 of router get rich non-zero gradients.
  3. Outer-loop query backward populates non-zero gradients into A_dyn^(0), router_weight^(0),
     and Meta-SGD step sizes (raw_eta_gate, raw_eta_A_dyn, raw_eta_B_dyn).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.meta_mole_layer import MetaMoLELinear
from src.models.regularizers import compute_dynamic_orthogonality_loss


def test_fomaml_and_meta_sgd_gradient_flow():
    """Verifies that k >= 2 inner steps and FOMAML + Meta-SGD populate all meta-gradients."""
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
    layer.expand_for_stage2_surgery(freeze_static_experts=True, scratchpad_bias_init=-2.0)

    # Support input (inner loop)
    x_s = torch.randn(B, S, d_in)
    target_s = torch.randn(B, S, d_out)

    # Step t=0
    # Current active parameters
    r_w_0 = layer.router_weight
    r_b_0 = layer.router_bias
    a_d_0 = layer.A_dyn
    b_d_0 = torch.zeros_like(layer.B_dyn)

    layer.set_adapted_params({
        "router_weight": r_w_0,
        "router_bias": r_b_0,
        "A_dyn": a_d_0,
        "B_dyn": b_d_0,
    })

    out_s0, _, gates_s0 = layer(x_s)
    loss_s0 = F.mse_loss(out_s0, target_s)

    # Compute inner gradients at t=0
    grads_0 = torch.autograd.grad(
        loss_s0,
        [r_w_0, r_b_0, a_d_0, b_d_0],
        create_graph=False,
        allow_unused=True,
    )
    g_rw_0, g_rb_0, g_ad_0, g_bd_0 = grads_0

    # At t=0, B_dyn must have non-zero gradient!
    assert g_bd_0 is not None and g_bd_0.norm().item() > 0, "B_dyn must receive non-zero gradient at t=0"
    print(f"Step t=0: ||grad_B_dyn|| = {g_bd_0.norm().item():.6f} > 0")

    # Meta-SGD update for step t=1
    eta_gate = F.softplus(layer.raw_eta_gate)
    eta_A = F.softplus(layer.raw_eta_A_dyn)
    eta_B = F.softplus(layer.raw_eta_B_dyn)

    g_bd_0_det = g_bd_0.detach()
    b_d_1 = b_d_0 - eta_B * g_bd_0_det  # Non-zero!
    assert b_d_1.norm().item() > 0, "B_dyn at step 1 must be non-zero"

    g_rw_0_val = torch.zeros_like(r_w_0) if g_rw_0 is None else g_rw_0.detach()
    g_rb_0_val = torch.zeros_like(r_b_0) if g_rb_0 is None else g_rb_0.detach()
    g_ad_0_val = torch.zeros_like(a_d_0) if g_ad_0 is None else g_ad_0.detach()

    r_w_1 = r_w_0 - eta_gate * g_rw_0_val
    r_b_1 = r_b_0 - eta_gate * g_rb_0_val
    a_d_1 = a_d_0 - eta_A * g_ad_0_val

    # Step t=1: Forward pass with non-zero B_dyn^(1)
    layer.set_adapted_params({
        "router_weight": r_w_1,
        "router_bias": r_b_1,
        "A_dyn": a_d_1,
        "B_dyn": b_d_1,
    })

    out_s1, _, gates_s1 = layer(x_s)
    loss_s1 = F.mse_loss(out_s1, target_s)

    grads_1 = torch.autograd.grad(
        loss_s1,
        [r_w_1, r_b_1, a_d_1, b_d_1],
        create_graph=False,
        allow_unused=True,
    )
    g_rw_1, g_rb_1, g_ad_1, g_bd_1 = grads_1

    # At t=1, A_dyn and row N+1 of router MUST receive non-zero gradient!
    assert g_ad_1 is not None and g_ad_1.norm().item() > 0, "A_dyn must receive non-zero gradient at step t=1"
    assert g_rw_1 is not None and g_rw_1[-1].norm().item() > 0, "Router row N+1 must receive non-zero gradient at step t=1"
    print(f"Step t=1: ||grad_A_dyn|| = {g_ad_1.norm().item():.6f} > 0")
    print(f"Step t=1: ||grad_W_g[N+1]|| = {g_rw_1[-1].norm().item():.6f} > 0")

    # Meta-SGD update for step t=2 (k=2 final inner parameters)
    r_w_2 = r_w_1 - eta_gate * g_rw_1.detach()
    r_b_2 = r_b_1 - eta_gate * g_rb_1.detach()
    a_d_2 = a_d_1 - eta_A * g_ad_1.detach()
    b_d_2 = b_d_1 - eta_B * g_bd_1.detach()

    # Outer-Loop Evaluation on Query Set Q_m
    x_q = torch.randn(B, S, d_in)
    target_q = torch.randn(B, S, d_out)

    layer.set_adapted_params({
        "router_weight": r_w_2,
        "router_bias": r_b_2,
        "A_dyn": a_d_2,
        "B_dyn": b_d_2,
    })

    out_q, _, _ = layer(x_q)
    loss_outer = F.mse_loss(out_q, target_q)

    # Backpropagate outer query loss
    loss_outer.backward()

    # Verify that outer gradients reached all meta-parameters!
    assert layer.A_dyn.grad is not None and layer.A_dyn.grad.norm().item() > 0, "A_dyn^(0) must receive non-zero outer gradient"
    assert layer.router_weight.grad is not None and layer.router_weight.grad.norm().item() > 0, "router_weight^(0) must receive non-zero outer gradient"
    assert layer.raw_eta_gate.grad is not None and layer.raw_eta_gate.grad.abs().item() > 0, "raw_eta_gate must receive non-zero outer gradient"
    assert layer.raw_eta_A_dyn.grad is not None and layer.raw_eta_A_dyn.grad.abs().item() > 0, "raw_eta_A_dyn must receive non-zero outer gradient"
    assert layer.raw_eta_B_dyn.grad is not None and layer.raw_eta_B_dyn.grad.abs().item() > 0, "raw_eta_B_dyn must receive non-zero outer gradient"

    print(f"Outer Grad on A_dyn^(0): norm = {layer.A_dyn.grad.norm().item():.6f}")
    print(f"Outer Grad on router_weight^(0): norm = {layer.router_weight.grad.norm().item():.6f}")
    print(f"Outer Grad on raw_eta_gate: {layer.raw_eta_gate.grad.item():.6f}")
    print(f"Outer Grad on raw_eta_A_dyn: {layer.raw_eta_A_dyn.grad.item():.6f}")
    print(f"Outer Grad on raw_eta_B_dyn: {layer.raw_eta_B_dyn.grad.item():.6f}")
    print("✓ test_fomaml_and_meta_sgd_gradient_flow passed!")


if __name__ == "__main__":
    test_fomaml_and_meta_sgd_gradient_flow()
    print("All FOMAML gradient tests passed successfully!")
