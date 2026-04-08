## Description

Expert computation and weighted aggregation in Sparse MoE with shared expert. Performs parallel expert forward passes (SwiGLU MLPs), applies routing weights, aggregates outputs, and adds gated shared expert contribution.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_seq_len, hidden_size] | bfloat16 |
| routing_weights | [batch_seq_len, num_experts_per_tok] | bfloat16 |
| selected_experts | [batch_seq_len, num_experts_per_tok] | int64 |
| expert_gate_proj_weights | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_up_proj_weights | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_down_proj_weights | [num_experts, hidden_size, moe_intermediate_size] | bfloat16 |
| shared_expert_gate_proj_weight | [shared_expert_intermediate_size, hidden_size] | bfloat16 |
| shared_expert_up_proj_weight | [shared_expert_intermediate_size, hidden_size] | bfloat16 |
| shared_expert_down_proj_weight | [hidden_size, shared_expert_intermediate_size] | bfloat16 |
| shared_expert_gate_weight | [1, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_seq_len, hidden_size] | bfloat16 |

| # | hidden_size | moe_intermediate_size | shared_expert_intermediate_size | num_experts | num_experts_per_tok | batch_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 512 | 512 | 512 | 10 | 3296 | 1.6505 | 0.4249 |
| 2 | 2048 | 512 | 512 | 512 | 10 | 3328 | 1.6209 | 0.4249 |
| 3 | 2048 | 512 | 512 | 512 | 10 | 8192 | 2.4413 | 0.4301 |
| 4 | 2048 | 512 | 512 | 512 | 10 | 3360 | 1.6525 | 0.4249 |
| 5 | 2048 | 512 | 512 | 512 | 10 | 3392 | 1.6678 | 0.4250 |
| 6 | 2048 | 512 | 512 | 512 | 10 | 3424 | 1.6666 | 0.4250 |
| 7 | 2048 | 512 | 512 | 512 | 10 | 3456 | 1.6621 | 0.4250 |
| 8 | 2048 | 512 | 512 | 512 | 10 | 3488 | 1.6825 | 0.4251 |
| 9 | 2048 | 512 | 512 | 512 | 10 | 3520 | 1.6885 | 0.4251 |
| 10 | 2048 | 512 | 512 | 512 | 10 | 3552 | 1.6849 | 0.4251 |
| 11 | 2048 | 512 | 512 | 512 | 10 | 3889 | 1.7820 | 0.4255 |
| 12 | 2048 | 512 | 512 | 512 | 10 | 3584 | 1.7136 | 0.4252 |
| 13 | 2048 | 512 | 512 | 512 | 10 | 3616 | 1.7298 | 0.4252 |
| 14 | 2048 | 512 | 512 | 512 | 10 | 3648 | 1.7248 | 0.4252 |
| 15 | 2048 | 512 | 512 | 512 | 10 | 3680 | 1.7859 | 0.4253 |
| 16 | 2048 | 512 | 512 | 512 | 10 | 3712 | 1.7411 | 0.4253 |

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
    shared_expert_intermediate_size = axes_and_scalars["shared_expert_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    hidden_states = torch.randn(
        batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device
    )
    
    # Routing weights should be normalized (sum to 1 per token)
    routing_weights_raw = torch.rand(
        batch_seq_len, num_experts_per_tok, dtype=torch.bfloat16, device=device
    )
    routing_weights = routing_weights_raw / routing_weights_raw.sum(dim=-1, keepdim=True)
    
    # Selected experts: each token selects top_k unique experts
    selected_experts = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(batch_seq_len)
    ], dim=0).to(torch.int64)
    
    # Expert weights
    expert_gate_proj_weights = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    expert_up_proj_weights = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    expert_down_proj_weights = torch.randn(
        num_experts, hidden_size, moe_intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    
    # Shared expert weights
    shared_expert_gate_proj_weight = torch.randn(
        shared_expert_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    shared_expert_up_proj_weight = torch.randn(
        shared_expert_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    shared_expert_down_proj_weight = torch.randn(
        hidden_size, shared_expert_intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    shared_expert_gate_weight = torch.randn(
        1, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "selected_experts": selected_experts,
        "expert_gate_proj_weights": expert_gate_proj_weights,
        "expert_up_proj_weights": expert_up_proj_weights,
        "expert_down_proj_weights": expert_down_proj_weights,
        "shared_expert_gate_proj_weight": shared_expert_gate_proj_weight,
        "shared_expert_up_proj_weight": shared_expert_up_proj_weight,
        "shared_expert_down_proj_weight": shared_expert_down_proj_weight,
        "shared_expert_gate_weight": shared_expert_gate_weight,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_gate_proj_weights: torch.Tensor,
    expert_up_proj_weights: torch.Tensor,
    expert_down_proj_weights: torch.Tensor,
    shared_expert_gate_proj_weight: torch.Tensor,
    shared_expert_up_proj_weight: torch.Tensor,
    shared_expert_down_proj_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
):
    """
    Expert computation and weighted aggregation with shared expert.
    
    Args:
        hidden_states: [batch_seq_len, hidden_size]
        routing_weights: [batch_seq_len, num_experts_per_tok]
        selected_experts: [batch_seq_len, num_experts_per_tok]
        expert_gate_proj_weights: [num_experts, moe_intermediate_size, hidden_size]
        expert_up_proj_weights: [num_experts, moe_intermediate_size, hidden_size]
        expert_down_proj_weights: [num_experts, hidden_size, moe_intermediate_size]
        shared_expert_gate_proj_weight: [shared_expert_intermediate_size, hidden_size]
        shared_expert_up_proj_weight: [shared_expert_intermediate_size, hidden_size]
        shared_expert_down_proj_weight: [hidden_size, shared_expert_intermediate_size]
        shared_expert_gate_weight: [1, hidden_size]
    
    Returns:
        output: [batch_seq_len, hidden_size]
    """
    batch_seq_len, hidden_dim = hidden_states.shape
    num_experts = expert_gate_proj_weights.shape[0]
    num_experts_per_tok = selected_experts.shape[1]
    
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # Initialize output accumulator
    final_hidden_states = torch.zeros(
        batch_seq_len, hidden_dim, dtype=dtype, device=device
    )
    
    # Create expert mask for efficient routing
    # Shape: [num_experts, num_experts_per_tok, batch_seq_len]
    expert_mask = F.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)
    
    # Process each expert
    for expert_idx in range(num_experts):
        # Find which tokens are routed to this expert
        # expert_mask[expert_idx] has shape [num_experts_per_tok, batch_seq_len]
        idx, top_x = torch.where(expert_mask[expert_idx])
        
        if top_x.numel() == 0:
            continue
        
        # Gather tokens for this expert
        current_state = hidden_states[top_x]  # [num_tokens_for_expert, hidden_size]
        
        # Get expert weights
        gate_proj_w = expert_gate_proj_weights[expert_idx]  # [intermediate, hidden]
        up_proj_w = expert_up_proj_weights[expert_idx]  # [intermediate, hidden]
        down_proj_w = expert_down_proj_weights[expert_idx]  # [hidden, intermediate]
        
        # Expert computation: SwiGLU MLP
        # gate_proj: [num_tokens, hidden] @ [hidden, intermediate] -> [num_tokens, intermediate]
        gate_out = torch.matmul(current_state, gate_proj_w.t())
        # up_proj: [num_tokens, hidden] @ [hidden, intermediate] -> [num_tokens, intermediate]
        up_out = torch.matmul(current_state, up_proj_w.t())
        # SwiGLU activation
        silu_gate = gate_out / (1.0 + torch.exp(-gate_out.float())).to(dtype)
        intermediate = silu_gate * up_out
        # down_proj: [num_tokens, intermediate] @ [intermediate, hidden] -> [num_tokens, hidden]
        expert_output = torch.matmul(intermediate, down_proj_w.t())
        
        # Weight by routing weights
        # routing_weights[top_x, idx] gives the weight for each token-expert pair
        weighted_output = expert_output * routing_weights[top_x, idx].unsqueeze(-1)
        
        # Scatter-add to final output
        final_hidden_states.index_add_(0, top_x, weighted_output)
    
    # Shared expert computation (applied to all tokens)
    # gate_proj: [batch_seq_len, hidden] @ [hidden, shared_intermediate]
    shared_gate = torch.matmul(hidden_states, shared_expert_gate_proj_weight.t())
    # SiLU activation
    shared_gate_silu = shared_gate / (1.0 + torch.exp(-shared_gate.float())).to(dtype)
    # up_proj: [batch_seq_len, hidden] @ [hidden, shared_intermediate]
    shared_up = torch.matmul(hidden_states, shared_expert_up_proj_weight.t())
    # Element-wise multiply
    shared_intermediate = shared_gate_silu * shared_up
    # down_proj: [batch_seq_len, shared_intermediate] @ [shared_intermediate, hidden]
    shared_expert_output = torch.matmul(shared_intermediate, shared_expert_down_proj_weight.t())
    
    # Gate the shared expert output
    # shared_expert_gate_weight: [1, hidden_size]
    # hidden_states: [batch_seq_len, hidden_size]
    # Result: [batch_seq_len, 1]
    gate_logits = torch.matmul(hidden_states, shared_expert_gate_weight.t())
    shared_gate_weight = torch.sigmoid(gate_logits.float()).to(dtype)
    shared_expert_output = shared_gate_weight * shared_expert_output
    
    # Combine routed experts and shared expert
    output = final_hidden_states + shared_expert_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.425422 ms |
| - | Scoring Baseline | 0.500000 | 1.735311 ms |
| - | Reference Implementation | 0.017716 | 73.512975 ms |
