## Description

Backward pass for complete MoE layer with shared expert. Computes gradients through router (sigmoid + top-k selection), 128 routed experts with SwiGLU, and shared expert with SwiGLU. Gradients flow back through scatter-gather routing pattern.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_seq_len, hidden_size] | bfloat16 |
| hidden_states | [batch_seq_len, hidden_size] | bfloat16 |
| router_weight | [n_routed_experts, hidden_size] | bfloat16 |
| e_score_correction_bias | [n_routed_experts] | float32 |
| router_logits | [batch_seq_len, n_routed_experts] | float32 |
| scores | [batch_seq_len, n_routed_experts] | float32 |
| topk_indices | [batch_seq_len, num_experts_per_tok] | int64 |
| topk_weights | [batch_seq_len, num_experts_per_tok] | float32 |
| score_mask | [batch_seq_len, n_routed_experts] | float32 |
| shared_expert_gate_weight | [moe_intermediate_size, hidden_size] | bfloat16 |
| shared_expert_up_weight | [moe_intermediate_size, hidden_size] | bfloat16 |
| shared_expert_down_weight | [hidden_size, moe_intermediate_size] | bfloat16 |
| shared_gate_output | [batch_seq_len, moe_intermediate_size] | bfloat16 |
| shared_up_output | [batch_seq_len, moe_intermediate_size] | bfloat16 |
| shared_activated | [batch_seq_len, moe_intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_seq_len, hidden_size] | bfloat16 |
| grad_router_weight | [n_routed_experts, hidden_size] | float32 |
| grad_shared_expert_gate_weight | [moe_intermediate_size, hidden_size] | bfloat16 |
| grad_shared_expert_up_weight | [moe_intermediate_size, hidden_size] | bfloat16 |
| grad_shared_expert_down_weight | [hidden_size, moe_intermediate_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | n_routed_experts | num_experts_per_tok | batch_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 1408 | 128 | 8 | 384 | 0.2685 | 0.0155 |
| 2 | 4096 | 1408 | 128 | 8 | 6144 | 0.6169 | 0.2423 |
| 3 | 4096 | 1408 | 128 | 8 | 8192 | 0.7691 | 0.3229 |
| 4 | 4096 | 1408 | 128 | 8 | 3719 | 0.4753 | 0.1468 |
| 5 | 4096 | 1408 | 128 | 8 | 512 | 0.3104 | 0.0206 |
| 6 | 4096 | 1408 | 128 | 8 | 192 | 0.2984 | 0.0087 |
| 7 | 4096 | 1408 | 128 | 8 | 211 | 0.3012 | 0.0089 |
| 8 | 4096 | 1408 | 128 | 8 | 3072 | 0.4265 | 0.1213 |
| 9 | 4096 | 1408 | 128 | 8 | 4093 | 0.4995 | 0.1615 |
| 10 | 4096 | 1408 | 128 | 8 | 1879 | 0.3368 | 0.0744 |
| 11 | 4096 | 1408 | 128 | 8 | 3089 | 0.4202 | 0.1220 |
| 12 | 4096 | 1408 | 128 | 8 | 2053 | 0.3503 | 0.0812 |
| 13 | 4096 | 1408 | 128 | 8 | 997 | 0.2756 | 0.0396 |
| 14 | 4096 | 1408 | 128 | 8 | 1571 | 0.3148 | 0.0622 |
| 15 | 4096 | 1408 | 128 | 8 | 1321 | 0.2972 | 0.0524 |
| 16 | 4096 | 1408 | 128 | 8 | 293 | 0.2692 | 0.0119 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = 4096
    moe_intermediate_size = 1408
    n_routed_experts = 128
    num_experts_per_tok = 8
    routed_scaling_factor = 1.0
    
    # Gradient from next layer
    grad_output = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Original hidden states
    hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Router weights
    router_weight = torch.randn(n_routed_experts, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    
    # Score correction bias
    e_score_correction_bias = torch.zeros(n_routed_experts, dtype=torch.float32, device=device)
    
    # Compute router logits and scores for realistic saved tensors
    router_logits = F.linear(hidden_states.to(torch.float32), router_weight.to(torch.float32))
    scores = torch.sigmoid(router_logits)
    
    # Compute top-k selection
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    topk_weights, topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)
    
    # Normalize weights
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = (topk_weights / denominator) * routed_scaling_factor
    
    # Score mask (all ones for n_group=1, topk_group=1)
    score_mask = torch.ones(batch_seq_len, n_routed_experts, dtype=torch.float32, device=device)
    
    # Shared expert weights
    shared_expert_gate_weight = torch.randn(moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    shared_expert_up_weight = torch.randn(moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    shared_expert_down_weight = torch.randn(hidden_size, moe_intermediate_size, dtype=torch.bfloat16, device=device) * 0.02
    
    # Compute shared expert forward pass for saved tensors
    shared_gate_output = F.linear(hidden_states, shared_expert_gate_weight)
    shared_up_output = F.linear(hidden_states, shared_expert_up_weight)
    shared_activated = F.silu(shared_gate_output) * shared_up_output
    
    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "router_weight": router_weight,
        "e_score_correction_bias": e_score_correction_bias,
        "router_logits": router_logits,
        "scores": scores,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "score_mask": score_mask,
        "shared_expert_gate_weight": shared_expert_gate_weight,
        "shared_expert_up_weight": shared_expert_up_weight,
        "shared_expert_down_weight": shared_expert_down_weight,
        "shared_gate_output": shared_gate_output,
        "shared_up_output": shared_up_output,
        "shared_activated": shared_activated,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    router_logits: torch.Tensor,
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    score_mask: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    shared_expert_up_weight: torch.Tensor,
    shared_expert_down_weight: torch.Tensor,
    shared_gate_output: torch.Tensor,
    shared_up_output: torch.Tensor,
    shared_activated: torch.Tensor,
):
    """
    Backward pass for MoE layer with shared expert.
    
    Computes gradients for:
    - hidden_states (input)
    - router_weight
    - shared_expert_gate_weight, shared_expert_up_weight, shared_expert_down_weight
    
    Note: Routed expert weight gradients are omitted for simplicity as they would
    require passing all 128*3 weight tensors and their saved activations.
    """
    batch_seq_len = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = 128
    norm_topk_prob = True
    routed_scaling_factor = 1.0
    
    # Initialize gradients
    grad_hidden_states = torch.zeros_like(hidden_states)
    
    # Gradient flows through addition: split to routed and shared paths
    grad_shared_output = grad_output.clone()
    
    # ===== Backward through shared expert =====
    # Gradient through shared_expert_down: output = down(activated)
    # down_weight shape: [hidden_size, moe_intermediate_size]
    # grad_shared_output shape: [batch_seq_len, hidden_size]
    # grad_shared_activated = grad_shared_output @ down_weight (need transpose for backward)
    grad_shared_activated = grad_shared_output @ shared_expert_down_weight
    
    # grad_shared_expert_down_weight = grad_shared_output.T @ shared_activated
    # Shape: [hidden_size, batch_seq_len] @ [batch_seq_len, moe_intermediate_size] = [hidden_size, moe_intermediate_size]
    grad_shared_expert_down_weight = grad_shared_output.t().to(torch.float32) @ shared_activated.to(torch.float32)
    grad_shared_expert_down_weight = grad_shared_expert_down_weight.to(torch.bfloat16)
    
    # Gradient through SwiGLU: activated = silu(gate) * up
    grad_shared_gate_silu = grad_shared_activated * shared_up_output
    grad_shared_up_output = grad_shared_activated * F.silu(shared_gate_output)
    
    # Gradient through silu: silu(x) = x * sigmoid(x)
    sigmoid_gate = torch.sigmoid(shared_gate_output.to(torch.float32))
    shared_gate_float = shared_gate_output.to(torch.float32)
    grad_shared_gate_output = grad_shared_gate_silu.to(torch.float32) * (
        sigmoid_gate * (1 + shared_gate_float * (1 - sigmoid_gate))
    )
    grad_shared_gate_output = grad_shared_gate_output.to(torch.bfloat16)
    
    # Gradient through shared_expert_up and shared_expert_gate
    # up_weight shape: [moe_intermediate_size, hidden_size]
    # grad_shared_up_output shape: [batch_seq_len, moe_intermediate_size]
    # grad_hidden_from_shared_up = grad_shared_up_output @ up_weight
    grad_hidden_from_shared_up = grad_shared_up_output @ shared_expert_up_weight
    grad_hidden_from_shared_gate = grad_shared_gate_output @ shared_expert_gate_weight
    
    # grad_shared_expert_up_weight = grad_shared_up_output.T @ hidden_states
    # Shape: [moe_intermediate_size, batch_seq_len] @ [batch_seq_len, hidden_size] = [moe_intermediate_size, hidden_size]
    grad_shared_expert_up_weight = grad_shared_up_output.t().to(torch.float32) @ hidden_states.to(torch.float32)
    grad_shared_expert_up_weight = grad_shared_expert_up_weight.to(torch.bfloat16)
    
    grad_shared_expert_gate_weight = grad_shared_gate_output.t().to(torch.float32) @ hidden_states.to(torch.float32)
    grad_shared_expert_gate_weight = grad_shared_expert_gate_weight.to(torch.bfloat16)
    
    grad_hidden_states = grad_hidden_states + grad_hidden_from_shared_up + grad_hidden_from_shared_gate
    
    # ===== Backward through routing =====
    # Routed expert output: y_routed = sum_k(w_norm_k * expert_k(x))
    # grad_topk_weights_norm[token, k] = dot(grad_output[token], expert_k_output[token])
    # Since routed expert outputs are not available as saved tensors (would require
    # 128*3 weight matrices and activations), we approximate using grad_output directly.
    # The routed contribution per token is: y_routed = sum_k(w_k * e_k), so
    # grad_w_k = <grad_output, e_k>. Without e_k, we use a straight-through estimator:
    # grad_w_k ≈ ||grad_output||^2 / num_experts_per_tok as an isotropic approximation.
    num_experts_per_tok = topk_weights.shape[-1]

    # Approximate grad_topk_weights using the norm of grad_output as a proxy
    # This captures the magnitude of the gradient signal flowing through routing
    grad_output_f32 = grad_output.to(torch.float32)
    grad_norm_sq = (grad_output_f32 * grad_output_f32).sum(dim=-1, keepdim=True)  # [batch_seq_len, 1]
    grad_topk_weights = grad_norm_sq.expand_as(topk_weights) / num_experts_per_tok

    # Gradient through routing weight normalization and scaling
    if norm_topk_prob:
        # Gradient through normalization: w_norm = w / sum(w) * routed_scaling_factor
        topk_weights_unnorm = topk_weights / routed_scaling_factor
        denominator = topk_weights_unnorm.sum(dim=-1, keepdim=True) + 1e-20
        grad_topk_weights_unnorm = grad_topk_weights / routed_scaling_factor

        # Quotient rule: d/dw_i (w_i / S) = (S - w_i) / S^2
        sum_grad = (grad_topk_weights_unnorm * topk_weights_unnorm).sum(dim=-1, keepdim=True) / denominator
        grad_topk_weights_before_norm = (grad_topk_weights_unnorm - sum_grad) / denominator
    else:
        grad_topk_weights_before_norm = grad_topk_weights / routed_scaling_factor

    # Gradient through top-k selection (sparse gradient)
    grad_scores_for_choice = torch.zeros(batch_seq_len, n_routed_experts, dtype=torch.float32, device=hidden_states.device)
    grad_scores_for_choice.scatter_add_(
        1,
        topk_indices,
        grad_topk_weights_before_norm
    )

    # Gradient through masking (only selected groups receive gradient)
    grad_scores_for_choice = grad_scores_for_choice * score_mask

    # Gradient through score correction (bias is non-trainable, so only propagate to scores)
    grad_scores = grad_scores_for_choice

    # Gradient through sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    grad_router_logits = grad_scores * scores * (1 - scores)

    # Gradient through router linear projection
    # router_weight shape: [n_routed_experts, hidden_size]
    # grad_router_logits shape: [batch_seq_len, n_routed_experts]
    grad_hidden_from_router = grad_router_logits.to(torch.bfloat16) @ router_weight

    # grad_router_weight = grad_router_logits.T @ hidden_states
    grad_router_weight = grad_router_logits.t() @ hidden_states.to(torch.float32)

    grad_hidden_states = grad_hidden_states + grad_hidden_from_router
    
    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.054632 ms |
| - | Scoring Baseline | 0.500000 | 0.370086 ms |
| - | Reference Implementation | 0.151703 | 2.026464 ms |
