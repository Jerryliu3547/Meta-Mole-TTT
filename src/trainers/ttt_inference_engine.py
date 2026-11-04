"""
Stateless Test-Time Training (TTT) Inference Engine.
Performs k inner-loop gradient adaptation steps on support demonstrations or prompt prefix,
invalidates and refreshes the KV cache, generates responses autoregressively,
and strictly resets the model to guarantee test-time statelessness.
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from ..data.benchmarks import BenchmarkItem
from ..data.paradigm_a_datamodule import format_compiled_support_batch
from ..models.model_wrapper import MetaMoLEModel
from ..models.regularizers import (
    compute_dynamic_orthogonality_loss,
    compute_kl_trust_region_loss,
)


class TTTInferenceEngine:
    """
    Inference Engine with Test-Time Adaptation.
    """

    def __init__(
        self,
        model: MetaMoLEModel,
        tokenizer: PreTrainedTokenizer,
        inner_steps: int = 3,
        gamma_kl: float = 0.1,
        gamma_ortho: float = 0.05,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.inner_steps = inner_steps
        self.gamma_kl = gamma_kl
        self.gamma_ortho = gamma_ortho
        self.grad_clip = grad_clip
        self.ad_hoc_eta: Optional[float] = None

        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = device
            self.model.to(self.device)

    def _prepare_support_tensors(
        self,
        few_shot_demos: Optional[List[Dict[str, str]]] = None,
        query_prompt: str = "",
        context_premise: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Formats and tokenizes support examples, context premise, or zero-shot prompt prefix."""
        if context_premise:
            # PARADIGM A: Self-supervised memory compilation of premise into scratchpad weights
            # Compiles premise into subspace-aligned recall pairs matching Stage 2 training
            compiled = format_compiled_support_batch(
                premise=context_premise,
                tokenizer=self.tokenizer,
                max_length=512,
                device=self.device,
            )
            compiled["is_memory"] = torch.tensor(1.0, device=self.device)
            return compiled
        elif few_shot_demos and len(few_shot_demos) > 0:
            is_memory = False
            # PARADIGM B: Few-shot demonstration adaptation
            input_ids_list = []
            labels_list = []
            for demo in few_shot_demos:
                p_text = f"<|im_start|>user\n{demo['input']}<|im_end|>\n<|im_start|>assistant\n"
                r_text = f"{demo['output']}<|im_end|>\n"
                p_ids = self.tokenizer.encode(p_text, add_special_tokens=False)
                r_ids = self.tokenizer.encode(r_text, add_special_tokens=False)
                input_ids_list.extend(p_ids + r_ids)
                labels_list.extend([-100] * len(p_ids) + r_ids)

            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
            labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)
        else:
            is_memory = False
            # Zero-shot self-supervised mode: Use prompt prefix
            text = f"<|im_start|>user\n{query_prompt}<|im_end|>\n<|im_start|>assistant\n"
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            labels = input_ids.clone()

        attention_mask = torch.ones_like(input_ids)
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if is_memory:
            res["is_memory"] = torch.tensor(1.0, device=self.device)
        return res

    def _adapt_parameters(
        self,
        support_tensors: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Executes k inner-loop adaptation steps on the test instance support data.
        Returns the adapted parameter dictionary.
        """
        input_ids = support_tensors["input_ids"]
        attention_mask = support_tensors["attention_mask"]
        labels = support_tensors["labels"]

        # Step 0: Record reference routing distribution
        with torch.no_grad():
            self.model.reset_adapted_params()
            ref_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            ref_gates = [g.detach() for g in ref_out.router_gates_list]

        # Initialize adaptable parameters and Solution 3 momentum buffers
        current_phi: List[Dict[str, torch.Tensor]] = []
        velocity_phi: List[Dict[str, torch.Tensor]] = []
        for layer in self.model.mole_layers:
            if not layer.is_ttt_layer:
                continue
            current_phi.append({
                "router_weight": layer.router_weight.clone().detach().requires_grad_(True),
                "router_bias": layer.router_bias.clone().detach().requires_grad_(True),
                "A_dyn": layer.A_dyn.clone().detach().requires_grad_(True),
                "B_dyn": torch.zeros_like(layer.B_dyn).requires_grad_(True),
            })
            velocity_phi.append({
                "router_weight": torch.zeros_like(layer.router_weight),
                "router_bias": torch.zeros_like(layer.router_bias),
                "A_dyn": torch.zeros_like(layer.A_dyn),
                "B_dyn": torch.zeros_like(layer.B_dyn),
            })

        momentum_beta = 0.5

        for step_t in range(self.inner_steps):
            self.model.set_adapted_params(current_phi)

            if self.inner_steps > 1:
                cos_decay = 0.35 + 0.65 * 0.5 * (1.0 + math.cos(math.pi * step_t / (self.inner_steps - 1)))
            else:
                cos_decay = 1.0

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_ce = output.loss

            # Solution 3: Soft brake when support batch CE loss is already saturated (< 0.08)
            if loss_ce is not None and loss_ce.item() < 0.08:
                cos_decay *= 0.25

            total_kl = torch.tensor(0.0, device=self.device)
            total_ortho = torch.tensor(0.0, device=self.device)

            active_tensors: List[torch.Tensor] = []
            active_etas: List[torch.Tensor] = []
            ttt_idx = 0

            for l_idx, layer in enumerate(self.model.mole_layers):
                if not layer.is_ttt_layer:
                    continue
                layer_phi = current_phi[ttt_idx]
                r_w = layer_phi["router_weight"]
                r_b = layer_phi["router_bias"]
                a_d = layer_phi["A_dyn"]
                b_d = layer_phi["B_dyn"]

                active_tensors.extend([r_w, r_b, a_d, b_d])
                if self.ad_hoc_eta is not None:
                    eta_val = torch.tensor(self.ad_hoc_eta * cos_decay, device=self.device)
                    active_etas.extend([eta_val, eta_val, eta_val, eta_val])
                elif "is_memory" in support_tensors:
                    # Use meta-learned step sizes with cosine decay for high-fidelity fast-weight storage
                    active_etas.extend([
                        torch.clamp(F.softplus(layer.raw_eta_gate).detach(), min=0.05) * cos_decay,
                        torch.clamp(F.softplus(layer.raw_eta_gate).detach(), min=0.05) * cos_decay,
                        torch.clamp(F.softplus(layer.raw_eta_A_dyn).detach(), min=0.05) * cos_decay,
                        torch.clamp(F.softplus(layer.raw_eta_B_dyn).detach(), min=0.18) * cos_decay,
                    ])
                else:
                    active_etas.extend([
                        F.softplus(layer.raw_eta_gate).detach() * cos_decay,
                        F.softplus(layer.raw_eta_gate).detach() * cos_decay,
                        F.softplus(layer.raw_eta_A_dyn).detach() * cos_decay,
                        F.softplus(layer.raw_eta_B_dyn).detach() * cos_decay,
                    ])

                if l_idx < len(output.router_gates_list) and l_idx < len(ref_gates):
                    total_kl = total_kl + compute_kl_trust_region_loss(
                        output.router_gates_list[l_idx],
                        ref_gates[l_idx],
                        num_static_experts=layer.num_experts,
                    )

                total_ortho = total_ortho + compute_dynamic_orthogonality_loss(a_d, layer.A_static)
                ttt_idx += 1

            num_ttt = max(1, ttt_idx)
            inner_loss = loss_ce + (self.gamma_kl * total_kl / num_ttt) + (self.gamma_ortho * total_ortho / num_ttt)

            grads = torch.autograd.grad(
                inner_loss,
                active_tensors,
                allow_unused=True,
            )

            next_phi: List[Dict[str, torch.Tensor]] = []
            ptr = 0
            ttt_i = 0
            for layer in self.model.mole_layers:
                if not layer.is_ttt_layer:
                    continue
                next_layer_phi = {}
                for key in ["router_weight", "router_bias", "A_dyn", "B_dyn"]:
                    p_curr = active_tensors[ptr]
                    eta = active_etas[ptr]
                    g = grads[ptr]
                    ptr += 1

                    if g is None:
                        g = torch.zeros_like(p_curr)

                    g_norm = torch.norm(g.detach(), p="fro") + 1e-6
                    g_unit = g.detach() / g_norm
                    v_prev = velocity_phi[ttt_i][key]
                    v_next = momentum_beta * v_prev + (1.0 - momentum_beta) * g_unit
                    v_next_norm = torch.norm(v_next, p="fro") + 1e-6
                    v_clipped = torch.clamp(v_next / v_next_norm, -self.grad_clip, self.grad_clip)
                    velocity_phi[ttt_i][key] = v_next.detach()

                    # Test-time update with momentum + cosine decay
                    p_next = (p_curr.detach() - eta * v_clipped).detach().requires_grad_(True)
                    next_layer_phi[key] = p_next

                next_phi.append(next_layer_phi)
                ttt_i += 1

            current_phi = next_phi

        return current_phi

    def adapt_and_generate(
        self,
        query_prompt: str,
        few_shot_demos: Optional[List[Dict[str, str]]] = None,
        context_premise: Optional[str] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Adapts parameters on the support context (demos or premise), invalidates KV cache,
        generates continuation with adapted weights, and strictly resets model state.
        """
        try:
            # 1. Adapt on support context
            support_tensors = self._prepare_support_tensors(
                few_shot_demos=few_shot_demos,
                query_prompt=query_prompt,
                context_premise=context_premise,
            )
            adapted_phi = self._adapt_parameters(support_tensors)

            # 2. Apply adapted parameters
            self.model.set_adapted_params(adapted_phi)

            # 3. Format target prompt for generation
            # Zero context in prompt for memory retrieval; isolated question directly targets memory
            full_prompt = ""
            if few_shot_demos:
                for demo in few_shot_demos:
                    full_prompt += f"<|im_start|>user\n{demo['input']}<|im_end|>\n<|im_start|>assistant\n{demo['output']}<|im_end|>\n"
            full_prompt += f"<|im_start|>user\n{query_prompt}<|im_end|>\n<|im_start|>assistant\n"

            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # 4. Generate with clean KV cache (past_key_values=None)
            generate_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
            else:
                generate_kwargs.update({"do_sample": False})

            with torch.no_grad():
                output_ids = self.model.generate(**generate_kwargs)

            # Decode only the generated completion tokens
            gen_tokens = output_ids[0][input_ids.shape[1] :]
            response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            return response_text

        finally:
            # 5. Guaranteed stateless reset
            self.model.reset_adapted_params()

    def evaluate_benchmark_item(self, item: Any) -> str:
        """Adapts on item.few_shot_demos or item.context_premise and answers item.query_input."""
        return self.adapt_and_generate(
            query_prompt=item.query_input,
            few_shot_demos=getattr(item, "few_shot_demos", None),
            context_premise=getattr(item, "context_premise", None),
            max_new_tokens=128,
            do_sample=False,
        )
