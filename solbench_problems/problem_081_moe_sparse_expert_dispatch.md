## Description

Sparse MoE expert computation with top-k routing across 128 routed experts. Each token is routed to 8 experts via sigmoid-based gating with group-wise selection. Includes SwiGLU activation (gate * up projection) and shared expert computation.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | bfloat16 |
| router_weight | [n_routed_experts, hidden_size] | bfloat16 |
| expert_gate_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_up_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_down_weights | [n_routed_experts, hidden_size, moe_intermediate_size] | bfloat16 |
| shared_gate_weight | [shared_intermediate, hidden_size] | bfloat16 |
| shared_up_weight | [shared_intermediate, hidden_size] | bfloat16 |
| shared_down_weight | [hidden_size, shared_intermediate] | bfloat16 |
| e_score_correction_bias | [n_routed_experts] | float32 |
| routed_scaling_factor | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | n_routed_experts | num_experts_per_tok | n_group | topk_group | shared_intermediate | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 10240 | 8.1106 | 1.7670 |
| 2 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1024 | 2.2359 | 0.5867 |
| 3 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1321 | 2.3327 | 0.5883 |
| 4 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1056 | 1.9483 | 0.5869 |
| 5 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1088 | 1.9884 | 0.5870 |
| 6 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1120 | 2.3637 | 0.5872 |
| 7 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1087 | 2.3743 | 0.5870 |
| 8 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 2131 | 3.0814 | 0.5926 |
| 9 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1152 | 2.0212 | 0.5874 |
| 10 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 2767 | 3.1712 | 0.5960 |
| 11 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 4093 | 4.3146 | 0.7065 |
| 12 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 4096 | 3.5153 | 0.7070 |
| 13 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1184 | 2.4139 | 0.5875 |
| 14 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1216 | 2.4482 | 0.5877 |
| 15 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 8192 | 6.4595 | 1.4137 |
| 16 | 4096 | 1408 | 128 | 8 | 1 | 1 | 1408 | 1248 | 2.6747 | 0.5879 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    Sparse MoE expert computation with top-k routing.
    
    1. Compute routing scores via sigmoid activation
    2. Apply group-wise expert selection (n_group=1, topk_group=1 simplifies to global topk)
    3. Select top-8 experts per token
    4. Compute SwiGLU for each selected expert
    5. Combine with shared expert output
    """
    # Constants
    n_routed_experts = 128
    num_experts_per_tok = 8
    n_group = 1
    topk_group = 1
    norm_topk_prob = True
    
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    
    # Router logits: [num_tokens, n_routed_experts]
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    
    # Sigmoid activation for scores
    scores = torch.sigmoid(router_logits)
    
    # Apply score correction bias
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    
    # Group-wise expert selection
    experts_per_group = n_routed_experts // n_group
    group_scores = scores_for_choice.view(-1, n_group, experts_per_group)
    
    # Select top-2 experts per group and sum their scores
    group_scores_agg = group_scores.topk(2, dim=-1)[0].sum(dim=-1)  # [num_tokens, n_group]
    
    # Select top groups
    group_idx = torch.topk(group_scores_agg, k=topk_group, dim=-1, sorted=False)[1]
    
    # Create mask for selected groups
    group_mask = torch.zeros_like(group_scores_agg)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand mask to all experts
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, experts_per_group)
        .reshape(-1, n_routed_experts)
    )
    
    # Mask out non-selected groups
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    
    # Select top-k experts
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    
    # Normalize weights if required
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    
    # Apply routing scaling factor
    topk_weights = topk_weights * routed_scaling_factor
    topk_weights = topk_weights.to(hidden_states.dtype)
    
    # Compute routed expert outputs
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    
    # Create expert mask: [n_routed_experts, num_tokens, num_experts_per_tok]
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)  # [n_routed_experts, num_tokens, num_experts_per_tok]
    
    # Process each expert
    for expert_idx in range(n_routed_experts):
        mask = expert_mask[expert_idx]  # [num_tokens, num_experts_per_tok]
        token_indices, weight_indices = torch.where(mask)
        
        if token_indices.numel() > 0:
            # Get routing weights for this expert
            expert_weights = topk_weights[token_indices, weight_indices]
            
            # Get input tokens for this expert
            expert_input = hidden_states[token_indices]
            
            # Expert MLP computation: SwiGLU activation
            gate_output = F.silu(
                F.linear(expert_input, expert_gate_weights[expert_idx])
            )
            up_output = F.linear(expert_input, expert_up_weights[expert_idx])
            intermediate = gate_output * up_output
            expert_output = F.linear(intermediate, expert_down_weights[expert_idx])
            
            # Apply routing weights
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Accumulate to final output
            final_hidden_states.index_add_(0, token_indices, weighted_output)
    
    final_hidden_states = final_hidden_states.to(hidden_states.dtype)
    
    # Compute shared expert output
    shared_gate_output = F.silu(F.linear(hidden_states, shared_gate_weight))
    shared_up_output = F.linear(hidden_states, shared_up_weight)
    shared_intermediate = shared_gate_output * shared_up_output
    shared_output = F.linear(shared_intermediate, shared_down_weight)
    
    # Combine routed and shared experts
    output = final_hidden_states + shared_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.681275 ms |
| - | Scoring Baseline | 0.500000 | 2.921284 ms |
| - | Reference Implementation | 0.223837 | 8.543660 ms |
