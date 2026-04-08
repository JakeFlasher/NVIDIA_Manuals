## Description

Complete MoE layer forward pass that fuses routing, expert dispatch/computation, aggregation, shared expert computation. This represents the entire MoE layer computation from input hidden states through router sigmoid scoring, hierarchical group-based top-8 expert selection, parallel expert execution (160 routed experts with 1536 intermediate size), weighted aggregation, and shared expert computation (also 1536 intermediate size).
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_seq_len, hidden_size] | bfloat16 |
| router_weight | [n_routed_experts, hidden_size] | bfloat16 |
| e_score_correction_bias | [n_routed_experts] | float32 |
| expert_gate_projs | [n_routed_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_up_projs | [n_routed_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_down_projs | [n_routed_experts, hidden_size, moe_intermediate_size] | bfloat16 |
| shared_gate_proj_weight | [shared_intermediate, hidden_size] | bfloat16 |
| shared_up_proj_weight | [shared_intermediate, hidden_size] | bfloat16 |
| shared_down_proj_weight | [hidden_size, shared_intermediate] | bfloat16 |
| routed_scaling_factor | scalar | float32 |
| norm_topk_prob | scalar | bool |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_seq_len, hidden_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | n_routed_experts | num_experts_per_tok | n_group | topk_group | shared_intermediate | batch_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 4093 | 5.6051 | 1.0167 |
| 2 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 3072 | 4.6495 | 1.0099 |
| 3 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1280 | 2.7524 | 0.9979 |
| 4 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1312 | 2.5273 | 0.9981 |
| 5 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1344 | 2.7786 | 0.9983 |
| 6 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 2371 | 4.1808 | 1.0052 |
| 7 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1376 | 2.9636 | 0.9986 |
| 8 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1408 | 2.7125 | 0.9988 |
| 9 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1440 | 2.7333 | 0.9990 |
| 10 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 3011 | 4.1525 | 1.0095 |
| 11 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1472 | 2.8502 | 0.9992 |
| 12 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1504 | 2.7109 | 0.9994 |
| 13 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1536 | 2.9549 | 0.9996 |
| 14 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1568 | 3.1209 | 0.9998 |
| 15 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 1600 | 3.1911 | 1.0001 |
| 16 | 5120 | 1536 | 160 | 8 | 1 | 1 | 1536 | 7168 | 8.8742 | 1.6875 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
    shared_gate_proj_weight: torch.Tensor,
    shared_up_proj_weight: torch.Tensor,
    shared_down_proj_weight: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """
    Complete MoE layer forward pass.
    
    1. Router: Sigmoid-based hierarchical group selection for top-8 experts from 160
    2. Routed Experts: 160 experts with SwiGLU activation
    3. Weighted Aggregation: Normalized routing weights scaled by routed_scaling_factor
    4. Shared Expert: Single expert processing all tokens
    5. Output: routed_output + shared_output
    """
    # Constants
    n_routed_experts = 160
    num_experts_per_tok = 8
    n_group = 1
    topk_group = 1
    
    batch_seq_len = hidden_states.shape[0]
    
    # Step 1: Compute routing
    # Router forward: [batch_seq_len, n_routed_experts]
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    scores = torch.sigmoid(router_logits)  # [batch_seq_len, 160]
    
    # Apply expert score correction bias
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    
    # Hierarchical group-based selection
    experts_per_group = n_routed_experts // n_group
    group_scores_reshaped = scores_for_choice.view(-1, n_group, experts_per_group)
    
    # Get top-2 scores per group and sum them
    group_scores = group_scores_reshaped.topk(2, dim=-1)[0].sum(dim=-1)  # [batch_seq_len, n_group]
    
    # Select top groups
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    # Create group mask
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand group mask to expert mask
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, experts_per_group)
        .reshape(-1, n_routed_experts)
    )
    
    # Mask out experts outside selected groups
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    
    # Select top-k experts within selected groups
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    
    # Normalize weights if configured
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    
    # Apply routing scaling factor
    topk_weights = topk_weights * routed_scaling_factor
    
    # Step 2: Compute routed expert outputs
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    
    # Create expert mask: [n_routed_experts, batch_seq_len, num_experts_per_tok]
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)  # [160, batch_seq_len, 8]
    
    # Process each expert
    for expert_idx in range(n_routed_experts):
        mask = expert_mask[expert_idx]  # [batch_seq_len, num_experts_per_tok]
        token_indices, weight_indices = torch.where(mask)
        
        if token_indices.numel() > 0:
            # Gather tokens for this expert
            expert_input = hidden_states[token_indices]  # [num_tokens, hidden_size]
            expert_weights_sel = topk_weights[token_indices, weight_indices]  # [num_tokens]
            
            # Expert forward: SwiGLU activation
            gate_output = F.linear(expert_input, expert_gate_projs[expert_idx])
            up_output = F.linear(expert_input, expert_up_projs[expert_idx])
            intermediate = F.silu(gate_output) * up_output
            expert_output = F.linear(intermediate, expert_down_projs[expert_idx])
            
            # Apply routing weights
            weighted_output = expert_output * expert_weights_sel.unsqueeze(-1)
            
            # Scatter-add back to final output
            final_hidden_states.index_add_(0, token_indices, weighted_output)
    
    routed_output = final_hidden_states.to(hidden_states.dtype)
    
    # Step 3: Compute shared expert output
    gate_output = F.linear(hidden_states, shared_gate_proj_weight)
    up_output = F.linear(hidden_states, shared_up_proj_weight)
    intermediate = F.silu(gate_output) * up_output
    shared_output = F.linear(intermediate, shared_down_proj_weight)
    
    # Step 4: Combine outputs
    output = routed_output + shared_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 1.035180 ms |
| - | Scoring Baseline | 0.500000 | 3.441763 ms |
| - | Reference Implementation | 0.207068 | 10.307168 ms |
