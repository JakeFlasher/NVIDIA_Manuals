## Description

Backward pass for MoE expert parallel execution. Computes gradients for hidden_states, topk_weights, gate_weights, up_weights, and down_weights through the SwiGLU MLP expert computation with token dispatch and weighted aggregation.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [num_tokens, hidden_size] | float32 |
| hidden_states | [num_tokens, hidden_size] | float32 |
| topk_indices | [num_tokens, num_experts_per_tok] | int64 |
| topk_weights | [num_tokens, num_experts_per_tok] | float32 |
| gate_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | float32 |
| up_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | float32 |
| down_weights | [n_routed_experts, hidden_size, moe_intermediate_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [num_tokens, hidden_size] | float32 |
| grad_topk_weights | [num_tokens, num_experts_per_tok] | float32 |
| grad_gate_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | float32 |
| grad_up_weights | [n_routed_experts, moe_intermediate_size, hidden_size] | float32 |
| grad_down_weights | [n_routed_experts, hidden_size, moe_intermediate_size] | float32 |

| # | hidden_size | moe_intermediate_size | n_routed_experts | num_experts_per_tok | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 2048 | 256 | 8 | 2080 | 14.4559 | 1.7119 |
| 2 | 4096 | 2048 | 256 | 8 | 2112 | 14.5840 | 1.7123 |
| 3 | 4096 | 2048 | 256 | 8 | 4096 | 20.0656 | 2.7321 |
| 4 | 4096 | 2048 | 256 | 8 | 2048 | 14.5438 | 1.7115 |
| 5 | 4096 | 2048 | 256 | 8 | 2144 | 14.6118 | 1.7126 |
| 6 | 4096 | 2048 | 256 | 8 | 2176 | 14.5550 | 1.7130 |
| 7 | 4096 | 2048 | 256 | 8 | 2208 | 14.6191 | 1.7134 |
| 8 | 4096 | 2048 | 256 | 8 | 2560 | 15.2859 | 1.7175 |
| 9 | 4096 | 2048 | 256 | 8 | 6144 | 23.3591 | 4.0980 |
| 10 | 4096 | 2048 | 256 | 8 | 2240 | 14.8042 | 1.7138 |
| 11 | 4096 | 2048 | 256 | 8 | 2272 | 14.9819 | 1.7142 |
| 12 | 4096 | 2048 | 256 | 8 | 2304 | 15.1060 | 1.7145 |
| 13 | 4096 | 2048 | 256 | 8 | 2336 | 14.9343 | 1.7149 |
| 14 | 4096 | 2048 | 256 | 8 | 2368 | 15.1678 | 1.7153 |
| 15 | 4096 | 2048 | 256 | 8 | 2400 | 15.3027 | 1.7157 |
| 16 | 4096 | 2048 | 256 | 8 | 2432 | 15.1965 | 1.7160 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    n_routed_experts = axes_and_scalars["n_routed_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Gradient from downstream
    grad_output = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32)
    
    # Original forward pass inputs
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32)
    
    # Generate valid topk_indices - each token selects num_experts_per_tok unique experts
    topk_indices = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(n_routed_experts, device=device)[:num_experts_per_tok]
        topk_indices[i] = perm
    
    # Routing weights (softmax normalized)
    topk_weights = torch.randn(num_tokens, num_experts_per_tok, device=device, dtype=torch.float32)
    topk_weights = F.softmax(topk_weights, dim=-1)
    
    # Expert weights
    gate_weights = torch.randn(n_routed_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    up_weights = torch.randn(n_routed_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    down_weights = torch.randn(n_routed_experts, hidden_size, moe_intermediate_size, device=device, dtype=torch.float32) * 0.02
    
    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "gate_weights": gate_weights,
        "up_weights": up_weights,
        "down_weights": down_weights,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
):
    """
    Backward pass for MoE expert parallel execution.
    
    Computes gradients through:
    1. Scatter-add aggregation (transpose = gather)
    2. Weight multiplication
    3. Down projection
    4. SwiGLU activation (gate_output * up_output)
    5. SiLU activation
    6. Gate and up projections
    7. Token gathering (transpose = scatter-add)
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = gate_weights.shape[0]
    moe_intermediate_size = gate_weights.shape[1]
    num_experts_per_tok = topk_indices.shape[1]
    device = hidden_states.device
    
    # Initialize output gradients
    grad_hidden_states = torch.zeros_like(hidden_states)
    grad_topk_weights = torch.zeros_like(topk_weights)
    grad_gate_weights = torch.zeros_like(gate_weights)
    grad_up_weights = torch.zeros_like(up_weights)
    grad_down_weights = torch.zeros_like(down_weights)
    
    # Create expert mask: [n_experts, num_tokens, num_experts_per_tok]
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)  # [n_experts, num_tokens, k]
    
    # Process each expert
    for expert_idx in range(n_routed_experts):
        mask = expert_mask[expert_idx]  # [num_tokens, k]
        token_indices, weight_indices = torch.where(mask)
        
        if token_indices.numel() == 0:
            continue
        
        # Get data for this expert's tokens
        expert_input = hidden_states[token_indices]  # [n_e, hidden_size]
        expert_weights_vals = topk_weights[token_indices, weight_indices]  # [n_e]
        
        # Recompute forward pass activations for this expert
        gate_pre_act = F.linear(expert_input, gate_weights[expert_idx])  # [n_e, intermediate]
        gate_output = F.silu(gate_pre_act)  # [n_e, intermediate]
        up_output = F.linear(expert_input, up_weights[expert_idx])  # [n_e, intermediate]
        intermediate = gate_output * up_output  # [n_e, intermediate]
        expert_output = F.linear(intermediate, down_weights[expert_idx])  # [n_e, hidden]
        
        # Gather gradient from output (reverse of scatter-add)
        grad_weighted_output = grad_output[token_indices]  # [n_e, hidden]
        
        # Gradient through weight multiplication:
        # weighted_output = expert_output * expert_weights.unsqueeze(-1)
        grad_expert_output = grad_weighted_output * expert_weights_vals.unsqueeze(-1)  # [n_e, hidden]
        grad_expert_weights = (grad_weighted_output * expert_output).sum(dim=-1)  # [n_e]
        
        # Accumulate gradient for topk_weights
        grad_topk_weights[token_indices, weight_indices] += grad_expert_weights
        
        # Gradient through down projection: expert_output = intermediate @ down_weights.T
        # grad_intermediate = grad_expert_output @ down_weights
        # grad_down_weights += grad_expert_output.T @ intermediate
        grad_intermediate = F.linear(grad_expert_output, down_weights[expert_idx].t())  # [n_e, intermediate]
        grad_down_weights[expert_idx] += grad_expert_output.t() @ intermediate  # [hidden, intermediate]
        
        # Gradient through element-wise multiplication: intermediate = gate_output * up_output
        grad_gate_output = grad_intermediate * up_output  # [n_e, intermediate]
        grad_up_output = grad_intermediate * gate_output  # [n_e, intermediate]
        
        # Gradient through SiLU: gate_output = silu(gate_pre_act)
        # silu(x) = x * sigmoid(x)
        # d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sigmoid_gate = torch.sigmoid(gate_pre_act)
        silu_grad = sigmoid_gate * (1.0 + gate_pre_act * (1.0 - sigmoid_gate))
        grad_gate_pre_act = grad_gate_output * silu_grad  # [n_e, intermediate]
        
        # Gradient through gate projection: gate_pre_act = expert_input @ gate_weights.T
        grad_expert_input_gate = F.linear(grad_gate_pre_act, gate_weights[expert_idx].t())  # [n_e, hidden]
        grad_gate_weights[expert_idx] += grad_gate_pre_act.t() @ expert_input  # [intermediate, hidden]
        
        # Gradient through up projection: up_output = expert_input @ up_weights.T
        grad_expert_input_up = F.linear(grad_up_output, up_weights[expert_idx].t())  # [n_e, hidden]
        grad_up_weights[expert_idx] += grad_up_output.t() @ expert_input  # [intermediate, hidden]
        
        # Combine gradients for expert input
        grad_expert_input = grad_expert_input_gate + grad_expert_input_up  # [n_e, hidden]
        
        # Scatter gradient back to original token positions (reverse of gather)
        grad_hidden_states.index_add_(0, token_indices, grad_expert_input)
    
    return grad_hidden_states, grad_topk_weights, grad_gate_weights, grad_up_weights, grad_down_weights
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 1.863529 ms |
| - | Scoring Baseline | 0.500000 | 15.580161 ms |
| - | Reference Implementation | 0.123849 | 99.207735 ms |
