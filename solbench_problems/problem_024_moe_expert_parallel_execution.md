## Description

Parallel execution of 256 routed MoE experts with token-to-expert dispatch and weighted aggregation. Each expert is a SwiGLU MLP (gate_proj, up_proj, down_proj). This operation performs expert masking, token gathering per expert, parallel expert computation, weight application, and scatter-add aggregation back to original token positions.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | bfloat16 |
| topk_indices | [num_tokens, num_experts_per_tok] | int64 |
| topk_weights | [num_tokens, num_experts_per_tok] | bfloat16 |
| gate_proj_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| up_proj_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| down_proj_weights | [n_routed_experts, hidden_size, moe_intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | n_routed_experts | num_experts_per_tok | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 2048 | 256 | 8 | 2447 | 23.9429 | 1.6861 |
| 2 | 4096 | 2048 | 256 | 8 | 2048 | 23.5183 | 1.6853 |
| 3 | 4096 | 2048 | 256 | 8 | 2080 | 23.5253 | 1.6853 |
| 4 | 4096 | 2048 | 256 | 8 | 2131 | 23.5650 | 1.6854 |
| 5 | 4096 | 2048 | 256 | 8 | 2112 | 23.5705 | 1.6854 |
| 6 | 4096 | 2048 | 256 | 8 | 3072 | 24.4856 | 1.6874 |
| 7 | 4096 | 2048 | 256 | 8 | 2144 | 23.6257 | 1.6855 |
| 8 | 4096 | 2048 | 256 | 8 | 2176 | 23.6218 | 1.6855 |
| 9 | 4096 | 2048 | 256 | 8 | 2208 | 23.6530 | 1.6856 |
| 10 | 4096 | 2048 | 256 | 8 | 2240 | 23.6694 | 1.6857 |
| 11 | 4096 | 2048 | 256 | 8 | 2767 | 24.2267 | 1.6868 |
| 12 | 4096 | 2048 | 256 | 8 | 2272 | 23.6967 | 1.6857 |
| 13 | 4096 | 2048 | 256 | 8 | 3803 | 24.9690 | 1.6890 |
| 14 | 4096 | 2048 | 256 | 8 | 8192 | 29.9026 | 1.8215 |
| 15 | 4096 | 2048 | 256 | 8 | 2304 | 23.7621 | 1.6858 |
| 16 | 4096 | 2048 | 256 | 8 | 2336 | 23.7756 | 1.6859 |

```python
import torch
import torch.nn.functional as F

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE expert parallel execution."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    n_routed_experts = axes_and_scalars["n_routed_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Input hidden states
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Expert selection indices - each token selects num_experts_per_tok experts
    topk_indices = torch.randint(
        0, n_routed_experts, (num_tokens, num_experts_per_tok), dtype=torch.int64, device=device
    )
    
    # Expert weights - softmax normalized
    topk_weights = torch.randn(num_tokens, num_experts_per_tok, dtype=torch.bfloat16, device=device)
    topk_weights = F.softmax(topk_weights.float(), dim=-1).to(torch.bfloat16)
    
    # Expert weight matrices
    gate_proj_weights = torch.randn(
        n_routed_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    up_proj_weights = torch.randn(
        n_routed_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    down_proj_weights = torch.randn(
        n_routed_experts, hidden_size, moe_intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    """Execute MoE experts in parallel with token dispatch and weighted aggregation.
    
    Args:
        hidden_states: Input tokens [num_tokens, hidden_size]
        topk_indices: Selected expert indices [num_tokens, num_experts_per_tok]
        topk_weights: Expert weights [num_tokens, num_experts_per_tok]
        gate_proj_weights: Gate projection weights [n_routed_experts, intermediate_size, hidden_size]
        up_proj_weights: Up projection weights [n_routed_experts, intermediate_size, hidden_size]
        down_proj_weights: Down projection weights [n_routed_experts, hidden_size, intermediate_size]
    
    Returns:
        Aggregated expert outputs [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = topk_indices.shape[1]
    
    # Initialize output buffer
    final_hidden_states = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device
    )
    
    # Create expert mask: [n_experts, num_tokens, num_experts_per_tok]
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)  # [n_experts, num_tokens, k]
    
    # Process each expert
    for expert_idx in range(n_routed_experts):
        mask = expert_mask[expert_idx]  # [num_tokens, k]
        
        # Find which tokens are routed to this expert
        token_indices, weight_indices = torch.where(mask)
        
        if token_indices.numel() > 0:
            # Gather tokens for this expert
            expert_input = hidden_states[token_indices].to(torch.float32)  # [num_expert_tokens, hidden_size]
            
            # Get corresponding weights
            expert_weights = topk_weights[token_indices, weight_indices].to(torch.float32)  # [num_expert_tokens]
            
            # Get expert weight matrices
            gate_w = gate_proj_weights[expert_idx].to(torch.float32)  # [intermediate_size, hidden_size]
            up_w = up_proj_weights[expert_idx].to(torch.float32)  # [intermediate_size, hidden_size]
            down_w = down_proj_weights[expert_idx].to(torch.float32)  # [hidden_size, intermediate_size]
            
            # SwiGLU computation: down(silu(gate(x)) * up(x))
            # gate_proj: [num_expert_tokens, hidden_size] @ [hidden_size, intermediate_size] = [num_expert_tokens, intermediate_size]
            gate_output = torch.matmul(expert_input, gate_w.t())  # [num_expert_tokens, intermediate_size]
            gate_output = F.silu(gate_output)
            
            # up_proj
            up_output = torch.matmul(expert_input, up_w.t())  # [num_expert_tokens, intermediate_size]
            
            # Element-wise multiply
            intermediate = gate_output * up_output  # [num_expert_tokens, intermediate_size]
            
            # down_proj
            expert_output = torch.matmul(intermediate, down_w.t())  # [num_expert_tokens, hidden_size]
            
            # Apply expert weights
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Scatter-add back to original positions
            final_hidden_states.index_add_(0, token_indices, weighted_output)
    
    return final_hidden_states.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 1.694193 ms |
| - | Scoring Baseline | 0.500000 | 24.177379 ms |
| - | Reference Implementation | 0.144876 | 134.408438 ms |
