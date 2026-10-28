"""
Stateless Test-Time Training (TTT) Inference Engine.
Performs k inner-loop gradient adaptation steps on support demonstrations or prompt prefix,
invalidates and refreshes the KV cache, generates responses autoregressively,
and strictly resets the model to guarantee test-time statelessness.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from ..data.benchmarks import BenchmarkItem
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

        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = device
            self.model.to(self.device)

    def _prepare_support_tensors(
        self,
        few_shot_demos: Optional[List[Dict[str, str]]] = None,
        query_prompt: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Formats and tokenizes support examples or zero-shot prompt prefix."""
        if few_shot_demos and len(few_shot_demos) > 0:
            # Few-shot mode: Concatenate demonstrations
            text = ""
            for demo in few_shot_demos:
                text += f"<|im_start|>user\n{demo['input']}<|im_end|>\n<|im_start|>assistant\n{demo['output']}<|im_end|>\n"
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            labels = input_ids.clone()
        else:
            # Zero-shot self-supervised mode: Use prompt prefix
            text = f"<|im_start|>user\n{query_prompt}<|im_end|>\n<|im_start|>assistant\n"
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            labels = input_ids.clone()

        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

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

        # Initialize adaptable parameters
        current_phi: List[Dict[str, torch.Tensor]] = []
        for layer in self.model.mole_layers:
            if not layer.is_ttt_layer:
                continue
            current_phi.append({
                "router_weight": layer.router_weight.clone().detach().requires_grad_(True),
                "router_bias": layer.router_bias.clone().detach().requires_grad_(True),
                "A_dyn": layer.A_dyn.clone().detach().requires_grad_(True),
                "B_dyn": torch.zeros_like(layer.B_dyn).requires_grad_(True),
            })

        for step_t in range(self.inner_steps):
            self.model.set_adapted_params(current_phi)

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_ce = output.loss

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
                active_etas.extend([
                    F.softplus(layer.raw_eta_gate).detach(),
                    F.softplus(layer.raw_eta_gate).detach(),
                    F.softplus(layer.raw_eta_A_dyn).detach(),
                    F.softplus(layer.raw_eta_B_dyn).detach(),
                ])

                if l_idx < len(output.router_gates_list) and l_idx < len(ref_gates):
                    total_kl = total_kl + compute_kl_trust_region_loss(output.router_gates_list[l_idx], ref_gates[l_idx])

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

                    g_clipped = torch.clamp(g.detach(), -self.grad_clip, self.grad_clip)
                    # Test-time update
                    p_next = (p_curr.detach() - eta * g_clipped).detach().requires_grad_(True)
                    next_layer_phi[key] = p_next

                next_phi.append(next_layer_phi)

            current_phi = next_phi

        return current_phi

    @torch.no_grad()
    def adapt_and_generate(
        self,
        query_prompt: str,
        few_shot_demos: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Adapts parameters on the support context, invalidates KV cache,
        generates continuation with adapted weights, and strictly resets model state.
        """
        try:
            # 1. Adapt on support context
            support_tensors = self._prepare_support_tensors(few_shot_demos, query_prompt)
            adapted_phi = self._adapt_parameters(support_tensors)

            # 2. Apply adapted parameters
            self.model.set_adapted_params(adapted_phi)

            # 3. Format target prompt for generation
            # Prepend few-shot demonstrations to context if present
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

            output_ids = self.model.generate(**generate_kwargs)

            # Decode only the generated completion tokens
            gen_tokens = output_ids[0][input_ids.shape[1] :]
            response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            return response_text

        finally:
            # 5. Guaranteed stateless reset
            self.model.reset_adapted_params()

    def evaluate_benchmark_item(self, item: BenchmarkItem) -> str:
        """Adapts on item.few_shot_demos and answers item.query_input."""
        return self.adapt_and_generate(
            query_prompt=item.query_input,
            few_shot_demos=item.few_shot_demos,
            max_new_tokens=128,
            do_sample=False,
        )
