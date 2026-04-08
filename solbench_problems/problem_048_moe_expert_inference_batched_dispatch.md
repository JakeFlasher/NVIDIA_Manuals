## Description

Optimized MoE expert inference dispatch that batches tokens by expert assignment, processes them through experts sequentially with SwiGLU activation, and reconstructs the output with weighted combinations. This is the critical inference path for a 256-expert MoE system that processes 8 experts per token.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [num_tokens, hidden_size] | bfloat16 |
| topk_idx | [num_tokens, num_experts_per_tok] | int64 |
| topk_weight | [num_tokens, num_experts_per_tok] | bfloat16 |
| gate_weights | [num_experts, intermediate_size, hidden_size] | bfloat16 |
| up_weights | [num_experts, intermediate_size, hidden_size] | bfloat16 |
| down_weights | [num_experts, hidden_size, intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [num_tokens, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | num_experts | num_experts_per_tok | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 1024 | 256 | 8 | 2080 | 6.2756 | 0.8451 |
| 2 | 4096 | 1024 | 256 | 8 | 2112 | 6.3026 | 0.8452 |
| 3 | 4096 | 1024 | 256 | 8 | 2144 | 6.3168 | 0.8452 |
| 4 | 4096 | 1024 | 256 | 8 | 2176 | 6.3600 | 0.8453 |
| 5 | 4096 | 1024 | 256 | 8 | 2048 | 6.2673 | 0.8450 |
| 6 | 4096 | 1024 | 256 | 8 | 2208 | 6.3893 | 0.8454 |
| 7 | 4096 | 1024 | 256 | 8 | 2240 | 6.3927 | 0.8454 |
| 8 | 4096 | 1024 | 256 | 8 | 4096 | 7.6884 | 0.8494 |
| 9 | 4096 | 1024 | 256 | 8 | 2272 | 6.4335 | 0.8455 |
| 10 | 4096 | 1024 | 256 | 8 | 2304 | 6.4253 | 0.8456 |
| 11 | 4096 | 1024 | 256 | 8 | 2336 | 6.4586 | 0.8456 |
| 12 | 4096 | 1024 | 256 | 8 | 2368 | 6.5294 | 0.8457 |
| 13 | 4096 | 1024 | 256 | 8 | 2400 | 6.5448 | 0.8458 |
| 14 | 4096 | 1024 | 256 | 8 | 2432 | 6.5619 | 0.8458 |
| 15 | 4096 | 1024 | 256 | 8 | 2464 | 6.5726 | 0.8459 |
| 16 | 4096 | 1024 | 256 | 8 | 2053 | 6.2725 | 0.8450 |

```python
import torch
import torch.nn.functional as F

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs including non-random topk indices."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Random hidden states
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    
    # Generate valid topk indices - each token selects num_experts_per_tok unique experts
    topk_idx = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(num_tokens)
    ]).to(torch.int64)
    
    # Random weights that sum to 1 per token (softmax-like)
    topk_weight_raw = torch.rand(num_tokens, num_experts_per_tok, device=device)
    topk_weight = (topk_weight_raw / topk_weight_raw.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    
    # Expert weights
    gate_weights = torch.randn(
        num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    up_weights = torch.randn(
        num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    down_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "topk_idx": topk_idx,
        "topk_weight": topk_weight,
        "gate_weights": gate_weights,
        "up_weights": up_weights,
        "down_weights": down_weights,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    """
    MoE expert inference with batched token dispatch.
    
    Args:
        hidden_states: Input tokens [num_tokens, hidden_size]
        topk_idx: Selected expert indices [num_tokens, num_experts_per_tok]
        topk_weight: Expert weights [num_tokens, num_experts_per_tok]
        gate_weights: Gate projection weights [num_experts, intermediate_size, hidden_size]
        up_weights: Up projection weights [num_experts, intermediate_size, hidden_size]
        down_weights: Down projection weights [num_experts, hidden_size, intermediate_size]
        
    Returns:
        output: Weighted expert outputs [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = gate_weights.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    
    # Convert to float32 for numerical stability
    hidden_states_f32 = hidden_states.to(torch.float32)
    topk_weight_f32 = topk_weight.to(torch.float32)
    gate_weights_f32 = gate_weights.to(torch.float32)
    up_weights_f32 = up_weights.to(torch.float32)
    down_weights_f32 = down_weights.to(torch.float32)
    
    # Step 1: Count tokens per expert using scatter
    cnts = torch.zeros(
        (num_tokens, num_experts),
        dtype=torch.int32,
        device=topk_idx.device
    )
    cnts.scatter_(1, topk_idx, 1)
    tokens_per_expert = cnts.sum(dim=0)  # [num_experts]
    
    # Step 2: Sort tokens by expert assignment
    flat_topk_idx = topk_idx.view(-1)  # [num_tokens * num_experts_per_tok]
    idxs = flat_topk_idx.argsort()  # Sort by expert ID
    
    # Gather tokens in sorted order
    token_indices = idxs // num_experts_per_tok
    sorted_tokens = hidden_states_f32[token_indices]  # [num_tokens * num_experts_per_tok, hidden_size]
    
    # Step 3: Process tokens through experts in batches
    outputs = []
    start_idx = 0
    tokens_per_expert_cpu = tokens_per_expert.cpu().numpy()
    
    for expert_id in range(num_experts):
        num_tokens_for_expert = int(tokens_per_expert_cpu[expert_id])
        
        if num_tokens_for_expert == 0:
            continue
            
        end_idx = start_idx + num_tokens_for_expert
        
        # Get tokens for this expert
        expert_input = sorted_tokens[start_idx:end_idx]  # [num_tokens_for_expert, hidden_size]
        
        # Expert computation: SwiGLU MLP
        # gate_proj: [num_tokens_for_expert, hidden_size] @ [hidden_size, intermediate_size]
        gate_out = torch.matmul(expert_input, gate_weights_f32[expert_id].t())
        
        # up_proj
        up_out = torch.matmul(expert_input, up_weights_f32[expert_id].t())
        
        # SwiGLU activation: silu(gate) * up
        # silu(x) = x * sigmoid(x)
        activated = (gate_out * torch.sigmoid(gate_out)) * up_out
        
        # down_proj
        expert_output = torch.matmul(activated, down_weights_f32[expert_id].t())
        
        outputs.append(expert_output)
        start_idx = end_idx
    
    # Step 4: Concatenate all expert outputs
    if len(outputs) == 0:
        return torch.zeros_like(hidden_states)
        
    all_outputs = torch.cat(outputs, dim=0)  # [num_tokens * num_experts_per_tok, hidden_size]
    
    # Step 5: Scatter back to original token positions
    new_x = torch.zeros_like(all_outputs)
    new_x[idxs] = all_outputs
    
    # Step 6: Reshape and apply expert weights
    new_x = new_x.view(num_tokens, num_experts_per_tok, hidden_size)
    
    # Apply weights and sum across experts
    weighted_output = new_x * topk_weight_f32.unsqueeze(-1)
    final_output = weighted_output.sum(dim=1)  # [num_tokens, hidden_size]
    
    return final_output.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.845678 ms |
| - | Scoring Baseline | 0.500000 | 6.479478 ms |
| - | Reference Implementation | 0.076137 | 69.204828 ms |
