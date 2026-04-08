## Description

Complete MoE expert computation pipeline including expert selection, token routing to experts, parallel expert forward passes (SwiGLU MLPs), and weighted accumulation of expert outputs. This is the core computational bottleneck in MoE models.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_seq_len, hidden_size] | bfloat16 |
| routing_weights | [batch_seq_len, num_experts_per_tok] | float32 |
| selected_experts | [batch_seq_len, num_experts_per_tok] | int64 |
| gate_proj_weights | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| up_proj_weights | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| down_proj_weights | [num_experts, hidden_size, moe_intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| final_hidden_states | [batch_seq_len, hidden_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | num_experts | num_experts_per_tok | batch_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 768 | 128 | 8 | 4093 | 1.7617 | 0.1710 |
| 2 | 2048 | 768 | 128 | 8 | 1024 | 9.6524 | 0.1590 |
| 3 | 2048 | 768 | 128 | 8 | 3089 | 1.4777 | 0.1612 |
| 4 | 2048 | 768 | 128 | 8 | 6144 | 2.3825 | 0.2565 |
| 5 | 2048 | 768 | 128 | 8 | 1056 | 0.8485 | 0.1591 |
| 6 | 2048 | 768 | 128 | 8 | 4096 | 1.7490 | 0.1711 |
| 7 | 2048 | 768 | 128 | 8 | 2048 | 1.1675 | 0.1601 |
| 8 | 2048 | 768 | 128 | 8 | 3719 | 1.6410 | 0.1619 |
| 9 | 2048 | 768 | 128 | 8 | 1087 | 0.8568 | 0.1591 |
| 10 | 2048 | 768 | 128 | 8 | 1088 | 0.8559 | 0.1591 |
| 11 | 2048 | 768 | 128 | 8 | 5120 | 2.0718 | 0.2138 |
| 12 | 2048 | 768 | 128 | 8 | 7168 | 2.6464 | 0.2992 |
| 13 | 2048 | 768 | 128 | 8 | 2447 | 1.2723 | 0.1606 |
| 14 | 2048 | 768 | 128 | 8 | 1120 | 9.7038 | 0.1591 |
| 15 | 2048 | 768 | 128 | 8 | 1571 | 0.9770 | 0.1596 |
| 16 | 2048 | 768 | 128 | 8 | 1152 | 0.8274 | 0.1592 |

```python
import torch
import torch.nn.functional as F

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Hidden states
    hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Routing weights (normalized per token)
    routing_weights_raw = torch.rand(batch_seq_len, num_experts_per_tok, dtype=torch.float32, device=device)
    routing_weights = routing_weights_raw / routing_weights_raw.sum(dim=-1, keepdim=True)
    
    # Selected experts - must be valid indices in [0, num_experts)
    # Each token selects num_experts_per_tok unique experts
    selected_experts = torch.zeros(batch_seq_len, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(batch_seq_len):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        selected_experts[i] = perm
    
    # Expert weights
    gate_proj_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    up_proj_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    down_proj_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=torch.bfloat16, device=device) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "selected_experts": selected_experts,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Complete MoE expert computation with token dispatch and weighted accumulation.
    
    Args:
        hidden_states: Input tokens [batch_seq_len, hidden_size]
        routing_weights: Normalized routing weights [batch_seq_len, num_experts_per_tok]
        selected_experts: Selected expert indices [batch_seq_len, num_experts_per_tok]
        gate_proj_weights: Gate projection weights [num_experts, moe_intermediate_size, hidden_size]
        up_proj_weights: Up projection weights [num_experts, moe_intermediate_size, hidden_size]
        down_proj_weights: Down projection weights [num_experts, hidden_size, moe_intermediate_size]
    
    Returns:
        final_hidden_states: Weighted expert outputs [batch_seq_len, hidden_size]
    """
    batch_seq_len, hidden_dim = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    
    # Initialize output accumulator
    final_hidden_states = torch.zeros(
        (batch_seq_len, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Create expert mask: [num_experts, num_experts_per_tok, batch_seq_len]
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Find which experts are actually used
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)
    
    # Process each active expert
    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()
        
        # Get expert weights
        gate_w = gate_proj_weights[expert_idx]  # [intermediate, hidden]
        up_w = up_proj_weights[expert_idx]      # [intermediate, hidden]
        down_w = down_proj_weights[expert_idx]  # [hidden, intermediate]
        
        # Find which tokens are assigned to this expert
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        
        if top_x.numel() == 0:
            continue
        
        # Gather tokens for this expert
        current_state = hidden_states[top_x].to(torch.float32)  # [num_tokens, hidden]
        
        # SwiGLU computation: down(silu(gate(x)) * up(x))
        gate_output = torch.matmul(current_state, gate_w.t().to(torch.float32))  # [num_tokens, intermediate]
        gate_activated = gate_output / (1.0 + torch.exp(-gate_output))  # SiLU
        up_output = torch.matmul(current_state, up_w.t().to(torch.float32))  # [num_tokens, intermediate]
        intermediate = gate_activated * up_output
        expert_output = torch.matmul(intermediate, down_w.t().to(torch.float32))  # [num_tokens, hidden]
        
        # Weight by routing weights
        weighted_output = expert_output * routing_weights[top_x, idx, None]
        
        # Scatter-add back to final output
        final_hidden_states.index_add_(0, top_x, weighted_output.to(hidden_states.dtype))
    
    return final_hidden_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.175831 ms |
| - | Scoring Baseline | 0.500000 | 1.736412 ms |
| - | Reference Implementation | 0.054735 | 37.489051 ms |
