## Description

Sparse expert dispatch and weighted combination for MoE training. Processes experts sequentially with sparse token extraction, expert computation (gate_up projection, gated GLU with clamping, down projection), and scattered accumulation using index_add_.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | float32 |
| router_indices | [num_tokens, num_experts_per_tok] | int64 |
| routing_weights | [num_tokens, num_local_experts] | float32 |
| gate_up_proj | [num_local_experts, hidden_size, intermediate_size_x2] | float32 |
| gate_up_proj_bias | [num_local_experts, intermediate_size_x2] | float32 |
| down_proj | [num_local_experts, intermediate_size, hidden_size] | float32 |
| down_proj_bias | [num_local_experts, hidden_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | float32 |

| # | hidden_size | intermediate_size | num_local_experts | num_experts_per_tok | intermediate_size_x2 | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2880 | 2880 | 128 | 4 | 5760 | 2048 | 3.4811 | 0.8346 |
| 2 | 2880 | 2880 | 128 | 4 | 5760 | 2080 | 3.4896 | 0.8346 |
| 3 | 2880 | 2880 | 128 | 4 | 5760 | 2112 | 3.3764 | 0.8347 |
| 4 | 2880 | 2880 | 128 | 4 | 5760 | 2144 | 3.3814 | 0.8347 |
| 5 | 2880 | 2880 | 128 | 4 | 5760 | 2176 | 3.5832 | 0.8348 |
| 6 | 2880 | 2880 | 128 | 4 | 5760 | 4096 | 4.9873 | 0.8376 |
| 7 | 2880 | 2880 | 128 | 4 | 5760 | 2208 | 3.6078 | 0.8348 |
| 8 | 2880 | 2880 | 128 | 4 | 5760 | 2240 | 3.6066 | 0.8349 |
| 9 | 2880 | 2880 | 128 | 4 | 5760 | 3011 | 4.0210 | 0.8360 |
| 10 | 2880 | 2880 | 128 | 4 | 5760 | 2293 | 3.6106 | 0.8349 |
| 11 | 2880 | 2880 | 128 | 4 | 5760 | 2272 | 3.5666 | 0.8349 |
| 12 | 2880 | 2880 | 128 | 4 | 5760 | 16384 | 12.8255 | 1.8011 |
| 13 | 2880 | 2880 | 128 | 4 | 5760 | 2304 | 3.6215 | 0.8350 |
| 14 | 2880 | 2880 | 128 | 4 | 5760 | 2336 | 3.6401 | 0.8350 |
| 15 | 2880 | 2880 | 128 | 4 | 5760 | 2368 | 3.7027 | 0.8350 |
| 16 | 2880 | 2880 | 128 | 4 | 5760 | 2400 | 3.6692 | 0.8351 |

```python
import torch
import torch.nn.functional as F

# Fixed constants for gated GLU activation
ALPHA = 1.702
LIMIT = 7.0

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for sparse expert dispatch and combine."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_local_experts = axes_and_scalars["num_local_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Hidden states
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device)
    
    # Router indices: each token selects top_k experts from [0, num_local_experts)
    # Generate random expert selections ensuring valid indices
    router_indices = torch.randint(
        0, num_local_experts, (num_tokens, num_experts_per_tok), dtype=torch.int64, device=device
    )
    
    # Routing weights: softmax-normalized weights for each expert
    routing_logits = torch.randn(num_tokens, num_local_experts, dtype=torch.float32, device=device)
    routing_weights = F.softmax(routing_logits, dim=-1)
    
    # Expert weights
    std = 0.02
    gate_up_proj = torch.randn(num_local_experts, hidden_size, 2 * intermediate_size, dtype=torch.float32, device=device) * std
    gate_up_proj_bias = torch.zeros(num_local_experts, 2 * intermediate_size, dtype=torch.float32, device=device)
    down_proj = torch.randn(num_local_experts, intermediate_size, hidden_size, dtype=torch.float32, device=device) * std
    down_proj_bias = torch.zeros(num_local_experts, hidden_size, dtype=torch.float32, device=device)
    
    return {
        "hidden_states": hidden_states,
        "router_indices": router_indices,
        "routing_weights": routing_weights,
        "gate_up_proj": gate_up_proj,
        "gate_up_proj_bias": gate_up_proj_bias,
        "down_proj": down_proj,
        "down_proj_bias": down_proj_bias,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse expert dispatch and weighted combination for MoE training.
    
    Args:
        hidden_states: Input tensor [num_tokens, hidden_size]
        router_indices: Selected expert indices [num_tokens, top_k]
        routing_weights: Routing weights [num_tokens, num_experts]
        gate_up_proj: Gate and up projection weights [num_experts, hidden_size, 2 * intermediate_size]
        gate_up_proj_bias: Gate and up projection bias [num_experts, 2 * intermediate_size]
        down_proj: Down projection weights [num_experts, intermediate_size, hidden_size]
        down_proj_bias: Down projection bias [num_experts, hidden_size]
    
    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    alpha = ALPHA
    limit = LIMIT
    
    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    
    # Initialize output accumulator
    output = torch.zeros_like(hidden_states)
    
    # Compute expert mask: [num_experts + 1, top_k, num_tokens]
    # +1 for masking class (tokens not assigned to any expert)
    expert_mask = F.one_hot(
        router_indices, 
        num_classes=num_experts + 1
    ).permute(2, 1, 0)
    
    # Identify which experts are used in this batch
    expert_hit = torch.greater(
        expert_mask.sum(dim=(-1, -2)), 0
    ).nonzero(as_tuple=False)
    
    # Process each active expert
    for expert_idx_tensor in expert_hit:
        expert_idx = expert_idx_tensor[0].item()
        
        # Skip masking index
        if expert_idx == num_experts:
            continue
        
        # Get token indices assigned to this expert
        _, token_idx = torch.where(expert_mask[expert_idx])
        
        # Extract tokens for this expert: [num_tokens_for_expert, hidden_size]
        current_state = hidden_states[token_idx]
        
        # Gate-up projection: [num_tokens_for_expert, 2 * intermediate_size]
        gate_up = (
            current_state @ gate_up_proj[expert_idx] + 
            gate_up_proj_bias[expert_idx]
        )
        
        # Split into gate and up: [num_tokens_for_expert, intermediate_size]
        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        
        # Apply clamping to prevent overflow
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        
        # Custom gated GLU activation
        # glu = gate * sigmoid(gate * alpha)
        glu = gate * torch.sigmoid(gate * alpha)
        gated_output = (up + 1) * glu
        
        # Down projection: [num_tokens_for_expert, hidden_size]
        expert_output = (
            gated_output @ down_proj[expert_idx] + 
            down_proj_bias[expert_idx]
        )
        
        # Apply routing weights: [num_tokens_for_expert, hidden_size]
        weighted_output = expert_output * routing_weights[token_idx, expert_idx, None]
        
        # Accumulate to output using scatter-add
        output.index_add_(0, token_idx, weighted_output)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.876201 ms |
| - | Scoring Baseline | 0.500000 | 3.971838 ms |
| - | Reference Implementation | 0.122569 | 23.611519 ms |
