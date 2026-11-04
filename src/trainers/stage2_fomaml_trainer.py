"""
Stage 2 Bi-Level FOMAML + Meta-SGD Trainer for Meta-MoLE-TTT.
Implements the First-Order MAML inner loop with detached gradients and
Meta-SGD step size learning without second-order Hessian overhead.
"""

from contextlib import nullcontext
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..data.episodic_collator import EpisodicTaskBatch
from ..models.model_wrapper import MetaMoLEModel
from ..models.regularizers import (
    compute_dynamic_orthogonality_loss,
    compute_kl_trust_region_loss,
)


def _math_sdpa_context():
    """
    Solution 2: Forces PyTorch's MATH SDPA backend so second-order backward (`create_graph=True`)
    through the final inner-loop write step succeeds without triggering FlashAttention's
    unimplemented double-backward kernel.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        return nullcontext()


class Stage2FOMAMLTrainer:
    """
    Bi-Level FOMAML + Meta-SGD Trainer for Stage 2.
    Adapts the router gate and dynamic scratchpad on the support set,
    and meta-updates the gate initialization, scratchpad basis, and inner step sizes on the query set.
    """

    def __init__(
        self,
        model: MetaMoLEModel,
        episodes: List[EpisodicTaskBatch],
        inner_steps: int = 3,
        meta_lr_gate: float = 2e-4,
        meta_lr_scratchpad: float = 2e-4,
        meta_lr_eta: float = 1e-3,
        gamma_kl: float = 0.1,
        gamma_ortho: float = 0.05,
        lambda_kl: float = 0.1,
        meta_batch_size: int = 2,
        total_episodes: int = 1000,
        inner_grad_clip: float = 1.0,
        outer_grad_clip: float = 1.0,
        logging_episodes: int = 10,
        save_episodes: int = 200,
        output_dir: str = "checkpoints/stage2",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.episodes = episodes
        self.inner_steps = inner_steps
        self.gamma_kl = gamma_kl
        self.gamma_ortho = gamma_ortho
        self.lambda_kl = lambda_kl
        self.meta_batch_size = meta_batch_size
        self.total_episodes = total_episodes
        self.inner_grad_clip = inner_grad_clip
        self.outer_grad_clip = outer_grad_clip
        self.logging_episodes = logging_episodes
        self.save_episodes = save_episodes
        self.output_dir = output_dir

        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = device
            self.model.to(self.device)

        # Build meta-optimizer for outer loop
        # Parameters to meta-learn: router_weight, router_bias, A_dyn, raw_eta_*
        gate_meta_params = []
        scratchpad_meta_params = []
        eta_meta_params = []

        for layer in self.model.mole_layers:
            if not layer.is_ttt_layer:
                continue
            gate_meta_params.extend([layer.router_weight, layer.router_bias])
            scratchpad_meta_params.append(layer.A_dyn)
            eta_meta_params.extend([layer.raw_eta_gate, layer.raw_eta_A_dyn, layer.raw_eta_B_dyn])

        self.outer_optimizer = AdamW(
            [
                {"params": gate_meta_params, "lr": meta_lr_gate},
                {"params": scratchpad_meta_params, "lr": meta_lr_scratchpad},
                {"params": eta_meta_params, "lr": meta_lr_eta},
            ]
        )
        total_opt_steps = max(1, self.total_episodes // max(1, self.meta_batch_size))
        self.outer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.outer_optimizer,
            T_max=total_opt_steps,
            eta_min=meta_lr_gate * 0.1,
        )

    def _execute_inner_loop(
        self,
        support_batch: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Executes k inner-loop adaptation steps on the support context S_m.
        Returns the adapted parameter dictionary per TTT layer.
        Uses detached gradients with softplus(eta) to enable first-order Meta-SGD gradient flow.
        """
        s_input_ids = support_batch["input_ids"].to(self.device)
        s_mask = support_batch["attention_mask"].to(self.device)
        s_labels = support_batch["labels"].to(self.device)

        # Step 0: Record reference routing distribution and initial adaptable parameters
        with torch.no_grad():
            self.model.reset_adapted_params()
            ref_output = self.model(input_ids=s_input_ids, attention_mask=s_mask)
            ref_gates_list = [g.detach() for g in ref_output.router_gates_list]

        # Initialize current state phi^(0) and Solution 3 inner-loop momentum buffers
        current_phi: List[Dict[str, torch.Tensor]] = []
        velocity_phi: List[Dict[str, torch.Tensor]] = []
        for layer in self.model.mole_layers:
            if not layer.is_ttt_layer:
                continue
            current_phi.append({
                "router_weight": layer.router_weight,
                "router_bias": layer.router_bias,
                "A_dyn": layer.A_dyn,
                "B_dyn": torch.zeros_like(layer.B_dyn).requires_grad_(True),  # B_dyn^(0) = 0
            })
            velocity_phi.append({
                "router_weight": torch.zeros_like(layer.router_weight),
                "router_bias": torch.zeros_like(layer.router_bias),
                "A_dyn": torch.zeros_like(layer.A_dyn),
                "B_dyn": torch.zeros_like(layer.B_dyn),
            })

        momentum_beta = 0.5

        # Run k inner steps with Solution 2 (SDPBackend.MATH for create_graph=True on final step)
        # and Solution 3 (Inner-Loop Momentum + Cosine Step Decay)
        for step_t in range(self.inner_steps):
            is_last_step = (step_t == self.inner_steps - 1)
            if self.inner_steps > 1:
                cos_decay = 0.35 + 0.65 * 0.5 * (1.0 + math.cos(math.pi * step_t / (self.inner_steps - 1)))
            else:
                cos_decay = 1.0

            # On the final step, re-attach meta-parameters via straight-through bridge
            if is_last_step and step_t > 0:
                bridged_phi: List[Dict[str, torch.Tensor]] = []
                ttt_i = 0
                for layer in self.model.mole_layers:
                    if not layer.is_ttt_layer:
                        continue
                    c_phi = current_phi[ttt_i]
                    bridged_phi.append({
                        "router_weight": layer.router_weight + (c_phi["router_weight"] - layer.router_weight).detach(),
                        "router_bias": layer.router_bias + (c_phi["router_bias"] - layer.router_bias).detach(),
                        "A_dyn": layer.A_dyn + (c_phi["A_dyn"] - layer.A_dyn).detach(),
                        "B_dyn": c_phi["B_dyn"].detach().requires_grad_(True),
                    })
                    ttt_i += 1
                current_phi = bridged_phi

            # Apply current phi to model
            self.model.set_adapted_params(current_phi)

            # Solution 2: Use MATH SDPA kernel on the final step so create_graph=True is supported
            sdpa_ctx = _math_sdpa_context() if is_last_step else nullcontext()
            with sdpa_ctx:
                # Forward pass on support context
                s_out = self.model(
                    input_ids=s_input_ids,
                    attention_mask=s_mask,
                    labels=s_labels,
                )
                s_ce_loss = s_out.loss

                # Compute regularizers: Anchor-preserving KL divergence and Dynamic Subspace Orthogonality
                total_inner_kl = torch.tensor(0.0, device=self.device)
                total_inner_ortho = torch.tensor(0.0, device=self.device)

                ttt_idx = 0
                active_tensors: List[torch.Tensor] = []
                active_etas: List[torch.Tensor] = []

                for l_idx, layer in enumerate(self.model.mole_layers):
                    if not layer.is_ttt_layer:
                        continue

                    curr_layer_phi = current_phi[ttt_idx]
                    r_w = curr_layer_phi["router_weight"]
                    r_b = curr_layer_phi["router_bias"]
                    a_d = curr_layer_phi["A_dyn"]
                    b_d = curr_layer_phi["B_dyn"]

                    active_tensors.extend([r_w, r_b, a_d, b_d])
                    active_etas.extend([
                        F.softplus(layer.raw_eta_gate) * cos_decay,
                        F.softplus(layer.raw_eta_gate) * cos_decay,
                        F.softplus(layer.raw_eta_A_dyn) * cos_decay,
                        F.softplus(layer.raw_eta_B_dyn) * cos_decay,
                    ])

                    # Anchor-Preserving KL Trust Region over the N static experts
                    if l_idx < len(s_out.router_gates_list) and l_idx < len(ref_gates_list):
                        total_inner_kl = total_inner_kl + compute_kl_trust_region_loss(
                            s_out.router_gates_list[l_idx],
                            ref_gates_list[l_idx],
                            num_static_experts=layer.num_experts,
                        )

                    # Dynamic Orthogonality constraint
                    total_inner_ortho = total_inner_ortho + compute_dynamic_orthogonality_loss(
                        a_d, layer.A_static
                    )

                    ttt_idx += 1

                num_ttt = max(1, ttt_idx)
                inner_loss = s_ce_loss + (self.gamma_kl * total_inner_kl / num_ttt) + (self.gamma_ortho * total_inner_ortho / num_ttt)

                # Retain graph on the final inner step under MATH SDPA so B_dyn^(k) carries write gradients
                inner_grads = torch.autograd.grad(
                    inner_loss,
                    active_tensors,
                    create_graph=is_last_step,
                    allow_unused=True,
                )

            next_phi: List[Dict[str, torch.Tensor]] = []
            tensor_ptr = 0

            ttt_idx = 0
            for layer in self.model.mole_layers:
                if not layer.is_ttt_layer:
                    continue

                layer_phi_next = {}
                for key in ["router_weight", "router_bias", "A_dyn", "B_dyn"]:
                    p_curr = active_tensors[tensor_ptr]
                    eta_curr = active_etas[tensor_ptr]
                    g_curr = inner_grads[tensor_ptr]
                    tensor_ptr += 1

                    if g_curr is None:
                        g_curr = torch.zeros_like(p_curr)

                    g_norm = torch.norm(g_curr.detach(), p="fro") + 1e-6
                    v_prev = velocity_phi[ttt_idx][key]

                    if is_last_step and key == "B_dyn":
                        # Solution 2 + 3: Differentiable write update with momentum
                        v_live = momentum_beta * v_prev + (1.0 - momentum_beta) * (g_curr / g_norm)
                        v_live_norm = torch.norm(v_live.detach(), p="fro") + 1e-6
                        layer_phi_next[key] = p_curr.detach() - eta_curr * (v_live / v_live_norm)
                        velocity_phi[ttt_idx][key] = v_live.detach()
                    else:
                        g_unit = g_curr.detach() / g_norm
                        v_next = momentum_beta * v_prev + (1.0 - momentum_beta) * g_unit
                        v_next_norm = torch.norm(v_next, p="fro") + 1e-6
                        v_normalized = torch.clamp(v_next / v_next_norm, -self.inner_grad_clip, self.inner_grad_clip)
                        velocity_phi[ttt_idx][key] = v_next.detach()
                        layer_phi_next[key] = p_curr - eta_curr * v_normalized

                next_phi.append(layer_phi_next)
                ttt_idx += 1

            current_phi = next_phi

        return current_phi

    def train_episode(self, episode: EpisodicTaskBatch) -> Dict[str, float]:
        """Runs inner adaptation on support set, then computes outer loss on query set."""
        # 1. Inner loop adaptation
        phi_k = self._execute_inner_loop(episode.support_batch)

        # 2. Outer loop evaluation on Query set (inside MATH SDPA context for second-order backward)
        self.model.set_adapted_params(phi_k)

        q_input_ids = episode.query_batch["input_ids"].to(self.device)
        q_mask = episode.query_batch["attention_mask"].to(self.device)
        q_labels = episode.query_batch["labels"].to(self.device)

        with _math_sdpa_context():
            q_out = self.model(
                input_ids=q_input_ids,
                attention_mask=q_mask,
                labels=q_labels,
            )
            outer_ce = q_out.loss

            # Outer anchor-preserving KL divergence and reference unadapted query loss
            outer_kl = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                self.model.reset_adapted_params()
                q_ref_out = self.model(input_ids=q_input_ids, attention_mask=q_mask, labels=q_labels)
                ref_ce = q_ref_out.loss.detach() if q_ref_out.loss is not None else outer_ce.detach()

            num_static = self.model.mole_layers[0].num_experts if self.model.mole_layers else None
            for g_k, g_ref in zip(q_out.router_gates_list, q_ref_out.router_gates_list):
                outer_kl = outer_kl + compute_kl_trust_region_loss(
                    g_k, g_ref, num_static_experts=num_static
                )

            # Contrastive memory margin: encourage adapted model to strictly beat unadapted reference
            margin_loss = F.relu(outer_ce - ref_ce + 0.15)
            outer_loss = (
                outer_ce
                + 0.25 * margin_loss
                + self.lambda_kl * (outer_kl / max(1, len(q_out.router_gates_list)))
            )

            # Backward on outer loss (accumulate)
            (outer_loss / self.meta_batch_size).backward()

        # Reset model parameter temporary state
        self.model.reset_adapted_params()

        return {
            "outer_ce": outer_ce.item(),
            "ref_ce": ref_ce.item(),
            "outer_kl": outer_kl.item(),
            "outer_loss": outer_loss.item(),
        }

    def train(self) -> Dict[str, Any]:
        """Runs Stage 2 meta-training loop over episodes."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.train()

        start_time = time.time()
        running_ce = 0.0
        running_ref_ce = 0.0
        running_kl = 0.0
        running_loss = 0.0

        self.outer_optimizer.zero_grad()
        num_episodes = min(self.total_episodes, len(self.episodes))

        print(f"[Stage 2 Trainer] Starting FOMAML meta-training for {num_episodes} episodes...")

        for ep_idx in range(1, num_episodes + 1):
            episode = self.episodes[ep_idx - 1]
            metrics = self.train_episode(episode)

            running_ce += metrics["outer_ce"]
            running_ref_ce += metrics.get("ref_ce", metrics["outer_ce"])
            running_kl += metrics["outer_kl"]
            running_loss += metrics["outer_loss"]

            # Meta-batch optimization step
            if ep_idx % self.meta_batch_size == 0 or ep_idx == num_episodes:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.outer_grad_clip)
                self.outer_optimizer.step()
                self.outer_scheduler.step()
                self.outer_optimizer.zero_grad()

            if ep_idx % self.logging_episodes == 0 or ep_idx == num_episodes:
                avg_ce = running_ce / self.logging_episodes
                avg_ref_ce = running_ref_ce / self.logging_episodes
                avg_kl = running_kl / self.logging_episodes
                avg_loss = running_loss / self.logging_episodes
                mem_gain = avg_ref_ce - avg_ce
                elapsed = time.time() - start_time

                print(
                    f"Episode [{ep_idx:4d}/{num_episodes:4d}] | "
                    f"Outer CE: {avg_ce:.4f} (Ref: {avg_ref_ce:.4f}, Gain: {mem_gain:+.4f}) | "
                    f"Outer KL: {avg_kl:.4f} | "
                    f"Total Outer: {avg_loss:.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                running_ce = 0.0
                running_ref_ce = 0.0
                running_kl = 0.0
                running_loss = 0.0

            if ep_idx % self.save_episodes == 0 or ep_idx == num_episodes:
                ckpt_path = os.path.join(self.output_dir, f"episode_{ep_idx}")
                print(f"[Stage 2 Trainer] Saving meta-checkpoint to {ckpt_path}...")
                self.model.save_mole_weights(ckpt_path)
                self.model.save_mole_weights(self.output_dir)

        print("[Stage 2 Trainer] Stage 2 FOMAML Meta-Training Complete.")
        return {
            "status": "success",
            "episodes": num_episodes,
            "output_dir": self.output_dir,
        }
