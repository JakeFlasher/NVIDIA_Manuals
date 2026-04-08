## Description

Sparse Mixture-of-Experts block with top-10 routing across 512 experts. Includes router network, top-k expert selection with normalized weights, parallel expert computation with weighted aggregation, and shared expert with learned gating. This is a complete MoE forward pass with dynamic token-to-expert routing.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| router_weight | [num_experts, hidden_size] | bfloat16 |
| expert_gate_proj | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_up_proj | [num_experts, moe_intermediate_size, hidden_size] | bfloat16 |
| expert_down_proj | [num_experts, hidden_size, moe_intermediate_size] | bfloat16 |
| shared_gate_proj | [shared_expert_intermediate_size, hidden_size] | bfloat16 |
| shared_up_proj | [shared_expert_intermediate_size, hidden_size] | bfloat16 |
| shared_down_proj | [hidden_size, shared_expert_intermediate_size] | bfloat16 |
| shared_expert_gate_weight | [1, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |
| router_logits | [batch_seq, num_experts] | bfloat16 |

| # | hidden_size | num_experts | num_experts_per_tok | moe_intermediate_size | shared_expert_intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 512 | 10 | 512 | 512 | 2 | 1664 | 1.7185 | 0.4256 |
| 2 | 2048 | 512 | 10 | 512 | 512 | 16 | 224 | 1.8102 | 0.4259 |
| 3 | 2048 | 512 | 10 | 512 | 512 | 64 | 128 | 2.6144 | 0.4315 |
| 4 | 2048 | 512 | 10 | 512 | 512 | 32 | 256 | 2.6190 | 0.4315 |
| 5 | 2048 | 512 | 10 | 512 | 512 | 64 | 613 | 9.0660 | 1.5449 |
| 6 | 2048 | 512 | 10 | 512 | 512 | 32 | 128 | 1.8845 | 0.4265 |
| 7 | 2048 | 512 | 10 | 512 | 512 | 1 | 3296 | 1.7313 | 0.4256 |
| 8 | 2048 | 512 | 10 | 512 | 512 | 8 | 416 | 1.7269 | 0.4256 |
| 9 | 2048 | 512 | 10 | 512 | 512 | 1 | 3328 | 1.7337 | 0.4256 |
| 10 | 2048 | 512 | 10 | 512 | 512 | 1 | 3360 | 1.7325 | 0.4257 |
| 11 | 2048 | 512 | 10 | 512 | 512 | 16 | 449 | 2.4615 | 0.4303 |
| 12 | 2048 | 512 | 10 | 512 | 512 | 4 | 832 | 1.7075 | 0.4256 |
| 13 | 2048 | 512 | 10 | 512 | 512 | 2 | 1696 | 1.7286 | 0.4257 |
| 14 | 2048 | 512 | 10 | 512 | 512 | 2 | 1728 | 1.7751 | 0.4258 |
| 15 | 2048 | 512 | 10 | 512 | 512 | 16 | 256 | 1.8854 | 0.4265 |
| 16 | 2048 | 512 | 10 | 512 | 512 | 8 | 448 | 1.7912 | 0.4259 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
):
    """
    Sparse MoE block with top-10 routing across 512 experts.
    
    Steps:
    1. Compute router logits via linear projection
    2. Apply softmax and select top-10 experts per token
    3. Normalize routing weights to sum to 1
    4. Process tokens through selected experts with SwiGLU activation
    5. Aggregate weighted expert outputs
    6. Add shared expert output with sigmoid gating
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_experts = router_weight.shape[0]
    top_k = 10
    
    # Flatten batch and sequence dimensions for routing
    hidden_states_flat = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden_size)
    
    # Compute router logits: (batch*seq, hidden_size) @ (hidden_size, num_experts)
    router_logits = torch.matmul(hidden_states_flat, router_weight.t())  # (batch*seq, 512)
    
    # Softmax over experts and select top-k
    routing_weights = F.softmax(router_logits.float(), dim=1)  # (batch*seq, 512) in float32
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    # routing_weights_topk: (batch*seq, 10), selected_experts: (batch*seq, 10)
    
    # Normalize top-k weights to sum to 1
    routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
    routing_weights_topk = routing_weights_topk.to(torch.bfloat16)
    
    # Initialize output accumulator
    num_tokens = batch_size * sequence_length
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )
    
    # Create expert mask: (num_experts, top_k, num_tokens)
    # expert_mask[i, j, k] = 1 if token k selected expert i in position j
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Find which experts have at least one token assigned
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
    
    # Process each expert that has tokens assigned
    for expert_idx in expert_hit:
        expert_idx_int = expert_idx.item()
        
        # Find which tokens selected this expert and at which top-k position
        idx, top_x = torch.where(expert_mask[expert_idx_int])
        
        if top_x.numel() == 0:
            continue
        
        # Gather tokens for this expert
        current_state = hidden_states_flat[top_x]  # (num_tokens_for_expert, hidden_size)
        
        # Get expert weights
        gate_w = expert_gate_proj[expert_idx_int]  # (intermediate_size, hidden_size)
        up_w = expert_up_proj[expert_idx_int]      # (intermediate_size, hidden_size)
        down_w = expert_down_proj[expert_idx_int]  # (hidden_size, intermediate_size)
        
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate_out = torch.matmul(current_state, gate_w.t())  # (n, intermediate_size)
        up_out = torch.matmul(current_state, up_w.t())      # (n, intermediate_size)
        
        # SiLU activation: x * sigmoid(x)
        silu_out = gate_out * torch.sigmoid(gate_out)
        
        # Element-wise multiply and down projection
        intermediate = silu_out * up_out
        expert_output = torch.matmul(intermediate, down_w.t())  # (n, hidden_size)
        
        # Weight by routing weights
        weighted_output = expert_output * routing_weights_topk[top_x, idx, None]
        
        # Scatter-add back to output
        final_hidden_states.index_add_(0, top_x, weighted_output)
    
    # Shared expert processes all tokens
    # SwiGLU for shared expert
    shared_gate_out = torch.matmul(hidden_states_flat, shared_gate_proj.t())  # (batch*seq, shared_intermediate)
    shared_up_out = torch.matmul(hidden_states_flat, shared_up_proj.t())      # (batch*seq, shared_intermediate)
    shared_silu = shared_gate_out * torch.sigmoid(shared_gate_out)
    shared_intermediate = shared_silu * shared_up_out
    shared_expert_output = torch.matmul(shared_intermediate, shared_down_proj.t())  # (batch*seq, hidden_size)
    
    # Shared expert gating: sigmoid(x @ gate_weight.T) * shared_output
    shared_gate = torch.sigmoid(torch.matmul(hidden_states_flat, shared_expert_gate_weight.t()))  # (batch*seq, 1)
    shared_expert_output = shared_gate * shared_expert_output
    
    # Combine sparse MoE and shared expert
    final_hidden_states = final_hidden_states + shared_expert_output
    
    # Reshape back to (batch, seq, hidden)
    output = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
    
    return output, router_logits.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.462622 ms |
| - | Scoring Baseline | 0.500000 | 2.099327 ms |
| - | Reference Implementation | 0.024423 | 72.086798 ms |
