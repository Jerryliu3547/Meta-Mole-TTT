"""
Loss functions and regularizers for Meta-MoLE-TTT.
Includes:
  - Load-balancing loss (Switch / GShard style)
  - Router Z-loss (prevents logit saturation)
  - Pairwise Subspace Orthogonality for collaborative static experts (Stage 1)
  - Dynamic Orthogonality constraint for scratchpad vs. static experts (Stage 2)
  - KL Trust Region divergence for router stability during TTT
  - Expert utilization entropy for collapse monitoring
"""

from typing import Optional

import torch
import torch.nn.functional as F


def compute_load_balancing_loss(
    gates: torch.Tensor,
    num_experts: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes Switch/GShard load-balancing auxiliary loss:
      L_bal = N * sum_{i=1}^N (f_i * P_i)
    where:
      f_i is the fraction of tokens routed to expert i as top-1 (detached),
      P_i is the average routing probability for expert i.

    Args:
        gates: Router softmax probabilities [..., N] or [..., N+1]
        num_experts: Number of experts to consider (defaults to gates.shape[-1])
    """
    if num_experts is None:
        num_experts = gates.shape[-1]

    # Focus on the specified experts
    curr_gates = gates[..., :num_experts].reshape(-1, num_experts)
    num_tokens = curr_gates.shape[0]

    if num_tokens == 0:
        return torch.tensor(0.0, device=gates.device)

    # Top-1 expert per token
    top1 = torch.argmax(curr_gates, dim=-1)  # [num_tokens]

    # Fraction of tokens assigned to each expert
    counts = torch.bincount(top1, minlength=num_experts).float()
    f = (counts / num_tokens).detach()

    # Mean probability per expert
    P = curr_gates.mean(dim=0)

    # Loss is N * sum(f_i * P_i)
    return float(num_experts) * torch.sum(f * P)


def compute_router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Router Z-Loss penalizes large log-sum-exp logits to prevent softmax saturation:
      L_z = (1 / (B * S)) * sum_{b, s} (log sum_i exp(z_{b, s, i}))^2

    Args:
        logits: Unnormalized or temperature-scaled router logits [...]
    """
    log_z = torch.logsumexp(logits, dim=-1)
    return torch.mean(log_z**2)


def compute_static_orthogonality_loss(
    A_static: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Pairwise Subspace Orthogonality Loss for Stage 1 static collaborative experts:
      L_ortho_static = sum_{1 <= i < j <= N} || A_hat_i @ A_hat_j^T ||_F^2
    where:
      A_hat_i = A_i / (||A_i||_F + eps)

    Args:
        A_static: Static expert down-projections of shape [N, r, d_in]
    """
    N, r, d_in = A_static.shape
    if N <= 1:
        return torch.tensor(0.0, device=A_static.device)

    # Compute Frobenius norms per expert: [N, 1, 1]
    norms = torch.norm(A_static, p="fro", dim=(1, 2), keepdim=True) + eps
    A_hat = A_static / norms  # [N, r, d_in]

    # Pairwise cross-expert product: [N, N, r, r]
    # cross[i, j] = A_hat[i] @ A_hat[j].T
    cross = torch.einsum("ird,jsd->ijsr", A_hat, A_hat)

    # Squared Frobenius norm for each pair (i, j): [N, N]
    frob_sq = torch.sum(cross**2, dim=(-2, -1))

    # Mask upper triangular (i < j)
    mask = torch.triu(torch.ones(N, N, device=A_static.device, dtype=torch.bool), diagonal=1)
    return frob_sq[mask].sum()


def compute_dynamic_orthogonality_loss(
    A_dyn: torch.Tensor,
    A_static: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Dynamic Orthogonality Constraint for Stage 2 inner loop:
      L_ortho_dyn = sum_{i=1}^N || A_dyn @ A_hat_i^T ||_F^2
    Forces the dynamic scratchpad down-projection to be orthogonal to the span of
    all static collaborative experts.

    Args:
        A_dyn: Dynamic scratchpad down-projection of shape [r, d_in]
        A_static: Static expert down-projections of shape [N, r, d_in]
    """
    norms = torch.norm(A_static, p="fro", dim=(1, 2), keepdim=True) + eps
    A_hat = A_static / norms  # [N, r, d_in]

    # cross[i] = A_dyn @ A_hat[i].T -> shape [N, r, r]
    cross = torch.einsum("rd,nid->nir", A_dyn, A_hat)
    return torch.sum(cross**2)


def compute_kl_trust_region_loss(
    gates_curr: torch.Tensor,
    gates_ref: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    KL Trust Region Loss between current router probabilities and reference router:
      D_KL(gates_curr || gates_ref) = sum_i gates_curr_i * log(gates_curr_i / gates_ref_i)

    Args:
        gates_curr: Current router probabilities [..., N]
        gates_ref: Reference router probabilities at episode start [..., N]
    """
    p = gates_curr.clamp(min=eps)
    q = gates_ref.clamp(min=eps)
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()
    return kl


def compute_expert_entropy(gates: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes the entropy of the mean routing distribution across tokens:
      H = -sum_i P_i * log(P_i)
    Useful for detecting and monitoring expert collapse.
    """
    P = gates.reshape(-1, gates.shape[-1]).mean(dim=0).clamp(min=eps)
    return -torch.sum(P * torch.log(P))
