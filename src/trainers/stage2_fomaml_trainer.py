"""
Stage 2 Bi-Level FOMAML + Meta-SGD Trainer for Meta-MoLE-TTT.
Implements the First-Order MAML inner loop with detached gradients and
Meta-SGD step size learning without second-order Hessian overhead.
"""

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

        # Initialize current state phi^(0)
        current_phi: List[Dict[str, torch.Tensor]] = []
        for layer in self.model.mole_layers:
            if not layer.is_ttt_layer:
                continue
            current_phi.append({
                "router_weight": layer.router_weight,
                "router_bias": layer.router_bias,
                "A_dyn": layer.A_dyn,
                "B_dyn": torch.zeros_like(layer.B_dyn),  # B_dyn^(0) = 0
            })

        # Run k inner steps
        for step_t in range(self.inner_steps):
            # Apply current phi to model
            self.model.set_adapted_params(current_phi)

            # Forward pass on support context
            s_out = self.model(
                input_ids=s_input_ids,
                attention_mask=s_mask,
                labels=s_labels,
            )
            s_ce_loss = s_out.loss

            # Compute regularizers: KL divergence from step 0 and Dynamic Subspace Orthogonality
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
                    F.softplus(layer.raw_eta_gate),
                    F.softplus(layer.raw_eta_gate),
                    F.softplus(layer.raw_eta_A_dyn),
                    F.softplus(layer.raw_eta_B_dyn),
                ])

                # KL Trust Region
                if l_idx < len(s_out.router_gates_list) and l_idx < len(ref_gates_list):
                    total_inner_kl = total_inner_kl + compute_kl_trust_region_loss(
                        s_out.router_gates_list[l_idx], ref_gates_list[l_idx]
                    )

                # Dynamic Orthogonality constraint
                total_inner_ortho = total_inner_ortho + compute_dynamic_orthogonality_loss(
                    a_d, layer.A_static
                )

                ttt_idx += 1

            num_ttt = max(1, ttt_idx)
            inner_loss = s_ce_loss + (self.gamma_kl * total_inner_kl / num_ttt) + (self.gamma_ortho * total_inner_ortho / num_ttt)

            # Compute gradients of inner loss w.r.t active parameters
            inner_grads = torch.autograd.grad(
                inner_loss,
                active_tensors,
                create_graph=False,  # First-order MAML (FOMAML)
                allow_unused=True,
            )

            # Detach gradients and perform Meta-SGD step: phi^(t+1) = phi^(t) - softplus(eta) * g_t
            # Keeping softplus(eta) in computation graph ensures outer-loop gradients update eta!
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

                    # Detach gradient for FOMAML and clip norm
                    g_detached = torch.clamp(g_curr.detach(), -self.inner_grad_clip, self.inner_grad_clip)

                    # Meta-SGD update
                    layer_phi_next[key] = p_curr - eta_curr * g_detached

                next_phi.append(layer_phi_next)
                ttt_idx += 1

            current_phi = next_phi

        return current_phi

    def train_episode(self, episode: EpisodicTaskBatch) -> Dict[str, float]:
        """Runs inner adaptation on support set, then computes outer loss on query set."""
        # 1. Inner loop adaptation
        phi_k = self._execute_inner_loop(episode.support_batch)

        # 2. Outer loop evaluation on Query set
        self.model.set_adapted_params(phi_k)

        q_input_ids = episode.query_batch["input_ids"].to(self.device)
        q_mask = episode.query_batch["attention_mask"].to(self.device)
        q_labels = episode.query_batch["labels"].to(self.device)

        q_out = self.model(
            input_ids=q_input_ids,
            attention_mask=q_mask,
            labels=q_labels,
        )
        outer_ce = q_out.loss

        # Outer KL divergence regularization
        outer_kl = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            self.model.reset_adapted_params()
            q_ref_out = self.model(input_ids=q_input_ids, attention_mask=q_mask)

        for g_k, g_ref in zip(q_out.router_gates_list, q_ref_out.router_gates_list):
            outer_kl = outer_kl + compute_kl_trust_region_loss(g_k, g_ref)

        outer_loss = outer_ce + self.lambda_kl * (outer_kl / max(1, len(q_out.router_gates_list)))

        # Backward on outer loss (accumulate)
        (outer_loss / self.meta_batch_size).backward()

        # Reset model parameter temporary state
        self.model.reset_adapted_params()

        return {
            "outer_ce": outer_ce.item(),
            "outer_kl": outer_kl.item(),
            "outer_loss": outer_loss.item(),
        }

    def train(self) -> Dict[str, Any]:
        """Runs Stage 2 meta-training loop over episodes."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.train()

        start_time = time.time()
        running_ce = 0.0
        running_kl = 0.0
        running_loss = 0.0

        self.outer_optimizer.zero_grad()
        num_episodes = min(self.total_episodes, len(self.episodes))

        print(f"[Stage 2 Trainer] Starting FOMAML meta-training for {num_episodes} episodes...")

        for ep_idx in range(1, num_episodes + 1):
            episode = self.episodes[ep_idx - 1]
            metrics = self.train_episode(episode)

            running_ce += metrics["outer_ce"]
            running_kl += metrics["outer_kl"]
            running_loss += metrics["outer_loss"]

            # Meta-batch optimization step
            if ep_idx % self.meta_batch_size == 0 or ep_idx == num_episodes:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.outer_grad_clip)
                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()

            if ep_idx % self.logging_episodes == 0 or ep_idx == num_episodes:
                avg_ce = running_ce / self.logging_episodes
                avg_kl = running_kl / self.logging_episodes
                avg_loss = running_loss / self.logging_episodes
                elapsed = time.time() - start_time

                print(
                    f"Episode [{ep_idx:4d}/{num_episodes:4d}] | "
                    f"Outer CE: {avg_ce:.4f} | "
                    f"Outer KL: {avg_kl:.4f} | "
                    f"Total Outer: {avg_loss:.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                running_ce = 0.0
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
