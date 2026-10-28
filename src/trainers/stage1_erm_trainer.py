"""
Stage 1 Collaborative MoLE Stabilization Trainer (ERM).
Trains N static collaborative LoRA experts and an N-way router end-to-end
with load-balancing loss, router Z-loss, and pairwise subspace orthogonality.
Includes router temperature annealing (tau: 1.5 -> 0.7) and expert collapse monitoring.
"""

import math
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..models.model_wrapper import MetaMoLEModel


class Stage1ERMTrainer:
    """
    Trainer for Stage 1 Collaborative MoLE.
    """

    def __init__(
        self,
        model: MetaMoLEModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr_experts: float = 2e-4,
        lr_router: float = 5e-4,
        weight_decay: float = 0.01,
        total_steps: int = 1000,
        warmup_steps: int = 100,
        lambda_bal: float = 0.01,
        lambda_z: float = 1e-3,
        lambda_ortho: float = 0.05,
        temp_start: float = 1.5,
        temp_end: float = 0.7,
        grad_clip: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 200,
        output_dir: str = "checkpoints/stage1",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lambda_bal = lambda_bal
        self.lambda_z = lambda_z
        self.lambda_ortho = lambda_ortho
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.grad_clip = grad_clip
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir

        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = device
            self.model.to(self.device)

        # Build optimizer with separate parameter groups for static experts and router
        expert_params = []
        router_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "A_static" in name or "B_static" in name:
                expert_params.append(param)
            elif "router_weight" in name or "router_bias" in name:
                router_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": expert_params, "lr": lr_experts, "weight_decay": weight_decay},
                {"params": router_params, "lr": lr_router, "weight_decay": 0.0},
            ]
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)

    def _get_current_temperature(self, step: int) -> float:
        """Linear annealing of router temperature from temp_start to temp_end."""
        progress = min(1.0, max(0.0, step / float(self.total_steps)))
        return self.temp_start + progress * (self.temp_end - self.temp_start)

    def train(self) -> Dict[str, Any]:
        """Runs the main Stage 1 ERM training loop."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.train()

        step = 0
        running_ce = 0.0
        running_bal = 0.0
        running_z = 0.0
        running_ortho = 0.0
        running_entropy = 0.0
        start_time = time.time()

        print(f"[Stage 1 Trainer] Starting training for {self.total_steps} steps on device {self.device}...")
        data_iter = iter(self.train_dataloader)

        while step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            step += 1

            # Update router temperature
            curr_temp = self._get_current_temperature(step)
            self.model.set_router_temperature(curr_temp)

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            ce_loss = output.loss

            # Regularization losses
            reg_dict = self.model.compute_stage1_regularizers(
                router_logits_list=output.router_logits_list,
                router_gates_list=output.router_gates_list,
                lambda_bal=self.lambda_bal,
                lambda_z=self.lambda_z,
                lambda_ortho=self.lambda_ortho,
            )

            total_loss = ce_loss + reg_dict["total_reg"]

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Warmup and LR schedule
            if step > self.warmup_steps:
                self.scheduler.step()

            # Accumulate logs
            running_ce += ce_loss.item()
            running_bal += reg_dict["loss_bal"].item()
            running_z += reg_dict["loss_z"].item()
            running_ortho += reg_dict["loss_ortho"].item()
            running_entropy += reg_dict["entropy"].item()

            if step % self.logging_steps == 0 or step == self.total_steps:
                avg_ce = running_ce / self.logging_steps
                avg_bal = running_bal / self.logging_steps
                avg_z = running_z / self.logging_steps
                avg_ortho = running_ortho / self.logging_steps
                avg_ent = running_entropy / self.logging_steps
                target_min_entropy = 0.85 * math.log(self.model.num_experts)

                elapsed = time.time() - start_time
                print(
                    f"Step [{step:4d}/{self.total_steps:4d}] | "
                    f"CE Loss: {avg_ce:.4f} | "
                    f"Bal: {avg_bal:.4f} | "
                    f"Z: {avg_z:.4f} | "
                    f"Ortho: {avg_ortho:.4f} | "
                    f"Entropy: {avg_ent:.3f} (target > {target_min_entropy:.3f}) | "
                    f"Temp: {curr_temp:.2f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                running_ce = 0.0
                running_bal = 0.0
                running_z = 0.0
                running_ortho = 0.0
                running_entropy = 0.0

            if step % self.save_steps == 0 or step == self.total_steps:
                ckpt_path = os.path.join(self.output_dir, f"step_{step}")
                print(f"[Stage 1 Trainer] Saving checkpoint to {ckpt_path}...")
                self.model.save_mole_weights(ckpt_path)
                # Also save to root of output_dir
                self.model.save_mole_weights(self.output_dir)

        print("[Stage 1 Trainer] Stage 1 ERM Training Complete.")
        return {
            "status": "success",
            "final_step": step,
            "output_dir": self.output_dir,
        }
