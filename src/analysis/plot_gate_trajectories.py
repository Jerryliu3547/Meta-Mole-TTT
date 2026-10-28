"""
Plotting utility for Meta-MoLE-TTT:
Visualizes the gate probability trajectory shifts across TTT inner steps k = 0, 1, 2, ..., K.
Shows how the (N+1)-th dynamic scratchpad probe activates and how collaborative expert weights adjust.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.model_wrapper import MetaMoLEModel


def record_gate_trajectories(
    model: MetaMoLEModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    adapted_phis: List[List[Dict[str, torch.Tensor]]],
) -> np.ndarray:
    """
    Computes average gate distribution across tokens for each inner step.

    Args:
        model: MetaMoLEModel in Stage 2
        input_ids: Input tokens [B, S]
        attention_mask: Mask [B, S]
        adapted_phis: List of length (K+1), where each element is the list of layer parameter dicts at step k

    Returns:
        trajectories: Array of shape [K+1, num_experts + 1]
    """
    trajectories = []

    for step_phi in adapted_phis:
        model.set_adapted_params(step_phi)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # Average gates across all layers and all tokens
            step_gates = []
            for g in output.router_gates_list:
                mean_g = g.reshape(-1, g.shape[-1]).mean(dim=0).cpu().numpy()
                step_gates.append(mean_g)
            avg_layer_gates = np.mean(step_gates, axis=0)
            trajectories.append(avg_layer_gates)

    model.reset_adapted_params()
    return np.array(trajectories)  # [K+1, num_experts + 1]


def plot_gate_trajectories(
    trajectories: np.ndarray,
    output_path: str = "artifacts/gate_trajectories.png",
    title: str = "Meta-MoLE Router Gating Trajectories Across TTT Steps",
):
    """
    Plots gate probability trajectory across inner adaptation steps.
    Highlights the dynamic scratchpad (expert N+1) vs. static collaborative experts.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_steps, num_experts_total = trajectories.shape
    num_static = num_experts_total - 1

    plt.figure(figsize=(10, 6))

    steps = list(range(num_steps))

    # Plot static experts
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_static))
    for i in range(num_static):
        plt.plot(
            steps,
            trajectories[:, i],
            marker="o",
            linestyle="--",
            alpha=0.7,
            color=colors[i],
            label=f"Static Expert {i+1}",
        )

    # Plot dynamic scratchpad in bold vibrant red
    plt.plot(
        steps,
        trajectories[:, -1],
        marker="s",
        linestyle="-",
        linewidth=3.0,
        color="#d62728",
        label=f"Dynamic Scratchpad (Expert {num_static + 1})",
    )

    plt.xlabel("TTT Inner Adaptation Steps (k)", fontsize=13)
    plt.ylabel("Mean Router Softmax Probability $g_i(x)$", fontsize=13)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(steps)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Analysis] Saved gate trajectory plot to {output_path}")
