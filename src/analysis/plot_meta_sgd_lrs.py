"""
Plotting utility for Meta-MoLE-TTT:
Extracts and visualizes learned Meta-SGD inner learning rates (eta_gate, eta_A_dyn, eta_B_dyn)
across Transformer layers to discover layer-wise plasticity profiles.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..models.model_wrapper import MetaMoLEModel


def extract_meta_sgd_lrs(model: MetaMoLEModel) -> Dict[str, List[float]]:
    """Extracts softplus(raw_eta) for all TTT layers in the model."""
    eta_gate_list = []
    eta_A_list = []
    eta_B_list = []

    for layer in model.mole_layers:
        if not layer.is_ttt_layer:
            continue
        with torch.no_grad():
            e_g = F.softplus(layer.raw_eta_gate).item()
            e_a = F.softplus(layer.raw_eta_A_dyn).item()
            e_b = F.softplus(layer.raw_eta_B_dyn).item()
        eta_gate_list.append(e_g)
        eta_A_list.append(e_a)
        eta_B_list.append(e_b)

    return {
        "eta_gate": eta_gate_list,
        "eta_A_dyn": eta_A_list,
        "eta_B_dyn": eta_B_list,
    }


def plot_meta_sgd_lrs(
    lr_dict: Dict[str, List[float]],
    output_path: str = "artifacts/meta_sgd_lrs.png",
    title: str = "Learned Meta-SGD Inner Learning Rates Across Layers",
):
    """Generates a grouped bar chart of learned inner learning rates by depth."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    num_layers = len(lr_dict["eta_gate"])
    if num_layers == 0:
        print("[Analysis] No TTT layers found to plot.")
        return

    x = np.arange(num_layers)
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, lr_dict["eta_gate"], width, label=r"Router Gate $\eta_{\rm gate}$", color="#1f77b4")
    plt.bar(x, lr_dict["eta_A_dyn"], width, label=r"Scratchpad Down $\eta_{A_{\rm dyn}}$", color="#2ca02c")
    plt.bar(x + width, lr_dict["eta_B_dyn"], width, label=r"Scratchpad Up $\eta_{B_{\rm dyn}}$", color="#ff7f0e")

    plt.xlabel("Layer Index (Depth)", fontsize=13)
    plt.ylabel("Learned Step Size $\eta = \mathrm{Softplus}(\widetilde{\eta})$", fontsize=13)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(x, [f"L{i+1}" for i in x])
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Analysis] Saved Meta-SGD learning rate plot to {output_path}")
