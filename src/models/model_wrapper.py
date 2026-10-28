"""
Model wrapper to inject MetaMoLELinear into HuggingFace PreTrainedModels.
Supports selective layer placement (e.g., upper half vs. all layers),
loss aggregation, surgery expansion, and parameter adaptation.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .meta_mole_layer import MetaMoLELinear
from .regularizers import (
    compute_expert_entropy,
    compute_load_balancing_loss,
    compute_router_z_loss,
    compute_static_orthogonality_loss,
)


@dataclass
class MetaMoLEOutput:
    """Output dataclass returned by MetaMoLEModel.forward()."""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    router_logits_list: List[torch.Tensor]
    router_gates_list: List[torch.Tensor]
    base_outputs: Any


class MetaMoLEModel(nn.Module):
    """
    Wraps a HuggingFace causal LM and replaces target linear projections with MetaMoLELinear.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        target_modules: Optional[List[str]] = None,
        num_experts: int = 8,
        rank: int = 8,
        alpha: float = 16.0,
        router_temp: float = 1.0,
        ttt_layer_indices: Union[str, List[int]] = "all",
    ):
        super().__init__()
        self.base_model = base_model
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.router_temp = router_temp
        self.ttt_layer_indices_spec = ttt_layer_indices

        # Default targets for standard Transformer architectures (Llama, Qwen, Mistral, SmolLM)
        if target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
        else:
            self.target_modules = target_modules

        # Freeze all base model parameters
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Injected MetaMoLELinear layers
        self.mole_layers: List[MetaMoLELinear] = []
        self._inject_mole_layers()

    def _get_transformer_layers(self) -> List[nn.Module]:
        """Detects Transformer decoder block list across standard HF models."""
        for attr in ["model.layers", "transformer.h", "layers", "decoder.layers"]:
            curr = self.base_model
            parts = attr.split(".")
            found = True
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found and isinstance(curr, (nn.ModuleList, list)):
                return list(curr)
        return []

    def _resolve_ttt_layers(self, total_layers: int) -> set:
        """Determines which layer indices participate in test-time adaptation."""
        if isinstance(self.ttt_layer_indices_spec, list):
            return set(self.ttt_layer_indices_spec)
        elif self.ttt_layer_indices_spec == "upper_half":
            return set(range(total_layers // 2, total_layers))
        else:  # "all"
            return set(range(total_layers))

    def _inject_mole_layers(self):
        """Recursively replaces target linear layers with MetaMoLELinear."""
        blocks = self._get_transformer_layers()
        total_layers = len(blocks)
        ttt_layer_set = self._resolve_ttt_layers(total_layers) if total_layers > 0 else set()

        if total_layers > 0:
            for l_idx, block in enumerate(blocks):
                is_ttt = (l_idx in ttt_layer_set)
                self._replace_modules(block, is_ttt=is_ttt, layer_idx=l_idx)
        else:
            # Fallback if block structure is atypical
            self._replace_modules(self.base_model, is_ttt=True, layer_idx=0)

    def _replace_modules(self, module: nn.Module, is_ttt: bool, layer_idx: int):
        """Helper to replace target nn.Linear inside module."""
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(child_name == tm or child_name.endswith(f".{tm}") for tm in self.target_modules):
                # Replace with MetaMoLELinear
                mole_linear = MetaMoLELinear(
                    base_layer=child,
                    num_experts=self.num_experts,
                    rank=self.rank,
                    alpha=self.alpha,
                    router_temp=self.router_temp,
                    is_ttt_layer=is_ttt,
                )
                setattr(module, child_name, mole_linear)
                self.mole_layers.append(mole_linear)
            else:
                self._replace_modules(child, is_ttt, layer_idx)

    def expand_for_stage2_surgery(
        self,
        freeze_static_experts: bool = True,
        scratchpad_bias_init: float = -2.0,
    ):
        """Performs N -> N+1 gate expansion surgery on all injected layers."""
        for layer in self.mole_layers:
            layer.expand_for_stage2_surgery(
                freeze_static_experts=freeze_static_experts,
                scratchpad_bias_init=scratchpad_bias_init,
            )

    def set_router_temperature(self, temp: float):
        """Updates router temperature for all layers (for temperature annealing)."""
        self.router_temp = temp
        for layer in self.mole_layers:
            layer.router_temp = temp

    def get_adaptable_parameters(self) -> List[Dict[str, nn.Parameter]]:
        """Returns adaptable parameter dicts for all TTT layers."""
        return [layer.get_adaptable_params() for layer in self.mole_layers if layer.is_ttt_layer]

    def set_adapted_params(self, adapted_params_list: List[Optional[Dict[str, torch.Tensor]]]):
        """Sets adapted parameters for all TTT layers during inner loop."""
        ttt_idx = 0
        for layer in self.mole_layers:
            if layer.is_ttt_layer:
                layer.set_adapted_params(adapted_params_list[ttt_idx])
                ttt_idx += 1

    def reset_adapted_params(self):
        """Clears all adapted parameter caches and zeros out B_dyn."""
        for layer in self.mole_layers:
            layer.reset_adapted_params()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MetaMoLEOutput:
        """
        Forward pass through base model with MetaMoLE layers.
        Collects router logits and gates from all layers.
        """
        # Hook/intercept to collect router logits and gates
        router_logits_list: List[torch.Tensor] = []
        router_gates_list: List[torch.Tensor] = []

        handles = []

        def forward_hook(module, args, output):
            # output is (total_out, logits, gates)
            total_out, logits, gates = output
            router_logits_list.append(logits)
            router_gates_list.append(gates)
            return total_out

        for layer in self.mole_layers:
            handles.append(layer.register_forward_hook(forward_hook))

        try:
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            for h in handles:
                h.remove()

        logits = base_outputs.logits if hasattr(base_outputs, "logits") else base_outputs[0]
        loss = getattr(base_outputs, "loss", None)

        return MetaMoLEOutput(
            loss=loss,
            logits=logits,
            router_logits_list=router_logits_list,
            router_gates_list=router_gates_list,
            base_outputs=base_outputs,
        )

    def compute_stage1_regularizers(
        self,
        router_logits_list: List[torch.Tensor],
        router_gates_list: List[torch.Tensor],
        lambda_bal: float = 0.01,
        lambda_z: float = 1e-3,
        lambda_ortho: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Stage 1 regularization losses:
          - Load-balancing loss
          - Router Z-loss
          - Static pairwise subspace orthogonality
        """
        total_bal = torch.tensor(0.0, device=self.base_model.device)
        total_z = torch.tensor(0.0, device=self.base_model.device)
        total_ortho = torch.tensor(0.0, device=self.base_model.device)
        total_entropy = torch.tensor(0.0, device=self.base_model.device)

        num_layers = len(self.mole_layers)
        if num_layers == 0:
            return {
                "loss_bal": total_bal,
                "loss_z": total_z,
                "loss_ortho": total_ortho,
                "total_reg": total_bal,
                "entropy": total_entropy,
            }

        for idx, layer in enumerate(self.mole_layers):
            if idx < len(router_gates_list):
                gates = router_gates_list[idx]
                total_bal = total_bal + compute_load_balancing_loss(gates, self.num_experts)
                total_entropy = total_entropy + compute_expert_entropy(gates)
            if idx < len(router_logits_list):
                r_logits = router_logits_list[idx]
                total_z = total_z + compute_router_z_loss(r_logits)
            total_ortho = total_ortho + compute_static_orthogonality_loss(layer.A_static)

        avg_bal = total_bal / num_layers
        avg_z = total_z / num_layers
        avg_ortho = total_ortho / num_layers
        avg_entropy = total_entropy / num_layers

        total_reg = (lambda_bal * avg_bal) + (lambda_z * avg_z) + (lambda_ortho * avg_ortho)

        return {
            "loss_bal": avg_bal,
            "loss_z": avg_z,
            "loss_ortho": avg_ortho,
            "total_reg": total_reg,
            "entropy": avg_entropy,
        }

    def generate(self, *args, **kwargs):
        """Pass-through to base_model.generate()."""
        return self.base_model.generate(*args, **kwargs)

    def save_mole_weights(self, output_dir: str):
        """Saves only the Meta-MoLE trainable weights (lightweight checkpoint)."""
        os.makedirs(output_dir, exist_ok=True)
        mole_state = {}
        for name, param in self.named_parameters():
            if any(k in name for k in ["A_static", "B_static", "router_weight", "router_bias", "A_dyn", "B_dyn", "raw_eta"]):
                mole_state[name] = param.data.cpu()

        torch.save(mole_state, os.path.join(output_dir, "meta_mole_weights.pt"))

        # Save config metadata
        config_data = {
            "num_experts": self.num_experts,
            "rank": self.rank,
            "alpha": self.alpha,
            "router_temp": self.router_temp,
            "ttt_layer_indices_spec": self.ttt_layer_indices_spec,
            "stage": self.mole_layers[0].stage if self.mole_layers else 1,
        }
        torch.save(config_data, os.path.join(output_dir, "meta_mole_config.pt"))

    def load_mole_weights(self, checkpoint_dir: str, strict: bool = False):
        """Loads lightweight Meta-MoLE weights from checkpoint."""
        weight_path = os.path.join(checkpoint_dir, "meta_mole_weights.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Checkpoint not found at {weight_path}")

        mole_state = torch.load(weight_path, map_location=self.base_model.device)
        self.load_state_dict(mole_state, strict=strict)
