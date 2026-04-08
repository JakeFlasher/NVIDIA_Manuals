## Description

Mixture-of-Experts sparse routing and expert dispatch system. Routes each token to top-8 experts out of 128 total experts using gating network, performs expert computation with dynamic batching, and aggregates results weighted by routing probabilities.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| gate_weight | [num_experts, hidden_size] | bfloat16 |
| expert_gate_proj | [num_experts, intermediate_size, hidden_size] | bfloat16 |
| expert_up_proj | [num_experts, intermediate_size, hidden_size] | bfloat16 |
| expert_down_proj | [num_experts, hidden_size, intermediate_size] | bfloat16 |
| norm_topk_prob | scalar | bool |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_experts | num_experts_per_tok | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 128 | 8 | 768 | 1 | 1024 | 0.6104 | 0.1591 |
| 2 | 2048 | 128 | 8 | 768 | 4 | 1024 | 1.2433 | 0.1723 |
| 3 | 2048 | 128 | 8 | 768 | 32 | 128 | 1.2510 | 0.1723 |
| 4 | 2048 | 128 | 8 | 768 | 2 | 512 | 0.6035 | 0.1591 |
| 5 | 2048 | 128 | 8 | 768 | 1 | 2048 | 0.8289 | 0.1602 |
| 6 | 2048 | 128 | 8 | 768 | 16 | 256 | 1.2381 | 0.1723 |
| 7 | 2048 | 128 | 8 | 768 | 4 | 2048 | 2.1893 | 0.3442 |
| 8 | 2048 | 128 | 8 | 768 | 1 | 1056 | 0.6001 | 0.1591 |
| 9 | 2048 | 128 | 8 | 768 | 8 | 613 | 1.4013 | 0.2062 |
| 10 | 2048 | 128 | 8 | 768 | 8 | 256 | 0.8309 | 0.1602 |
| 11 | 2048 | 128 | 8 | 768 | 1 | 1088 | 0.6316 | 0.1592 |
| 12 | 2048 | 128 | 8 | 768 | 4 | 288 | 0.6295 | 0.1592 |
| 13 | 2048 | 128 | 8 | 768 | 1 | 1120 | 0.6258 | 0.1592 |
| 14 | 2048 | 128 | 8 | 768 | 4 | 256 | 0.5959 | 0.1591 |
| 15 | 2048 | 128 | 8 | 768 | 16 | 512 | 2.1150 | 0.3442 |
| 16 | 2048 | 128 | 8 | 768 | 1 | 1152 | 0.6400 | 0.1592 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    norm_topk_prob: bool,
):
    """
    Sparse MoE routing and expert dispatch with top-k gating.
    
    1. Gate network computes routing logits for all 128 experts
    2. Top-8 experts are selected per token via softmax + topk
    3. Routing weights are normalized across selected experts
    4. Tokens are dynamically dispatched to experts
    5. Expert outputs are weighted and aggregated
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_experts = gate_weight.shape[0]
    top_k = 8
    
    # Flatten batch and sequence dimensions for routing
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    num_tokens = hidden_states_flat.shape[0]
    
    # Step 1: Compute routing logits via gating network
    # (num_tokens, hidden_size) @ (hidden_size, num_experts) -> (num_tokens, num_experts)
    router_logits = torch.matmul(hidden_states_flat, gate_weight.t())
    
    # Step 2: Compute routing weights via softmax
    routing_weights = F.softmax(router_logits.float(), dim=1).to(hidden_states.dtype)
    
    # Step 3: Select top-k experts per token
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # Step 4: Normalize routing weights across selected experts (if enabled)
    if norm_topk_prob:
        routing_weights_topk = routing_weights_topk / (routing_weights_topk.sum(dim=-1, keepdim=True) + 1e-9)
    
    # Step 5: Initialize output accumulator
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Step 6: Create expert mask for efficient dispatching
    # One-hot encode selected experts: (num_tokens, top_k, num_experts)
    # Then permute to: (num_experts, top_k, num_tokens)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Step 7: Loop over experts and compute outputs for assigned tokens
    for expert_idx in range(num_experts):
        # Find which tokens are routed to this expert and at which top-k position
        idx, top_x = torch.where(expert_mask[expert_idx])
        
        if top_x.numel() == 0:
            continue
        
        # Gather hidden states for tokens assigned to this expert
        current_state = hidden_states_flat[top_x]
        
        # Expert MLP forward: SiLU(gate_proj(x)) * up_proj(x), then down_proj
        gate_out = torch.matmul(current_state, expert_gate_proj[expert_idx].t())
        up_out = torch.matmul(current_state, expert_up_proj[expert_idx].t())
        
        # SiLU activation: x * sigmoid(x)
        silu_gate = gate_out * torch.sigmoid(gate_out)
        intermediate = silu_gate * up_out
        
        expert_out = torch.matmul(intermediate, expert_down_proj[expert_idx].t())
        
        # Weight by routing probability
        current_hidden_states = expert_out * routing_weights_topk[top_x, idx, None]
        
        # Scatter-add expert outputs back to final output
        final_hidden_states.index_add_(0, top_x, current_hidden_states)
    
    # Step 8: Reshape output back to original dimensions
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    
    return final_hidden_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.180959 ms |
| - | Scoring Baseline | 0.500000 | 0.898527 ms |
| - | Reference Implementation | 0.048010 | 15.777195 ms |
