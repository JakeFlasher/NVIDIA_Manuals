## Description

Batched MoE expert forward pass for inference that processes all experts in parallel. Includes gate_up projection, custom gated GLU activation with clamping, down projection, and routing weight application.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| routing_weights | [tokens, num_experts] | float32 |
| gate_up_proj | [num_experts, hidden_size, gate_up_dim] | float32 |
| gate_up_proj_bias | [num_experts, gate_up_dim] | float32 |
| down_proj | [num_experts, expert_dim, hidden_size] | float32 |
| down_proj_bias | [num_experts, hidden_size] | float32 |
| alpha | scalar | float32 |
| limit | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | float32 |

| # | hidden_size | intermediate_size | num_experts | expert_dim | gate_up_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2880 | 2880 | 128 | 2880 | 5760 | 4 | 293 | 19.4087 | 4.1223 |
| 2 | 2880 | 2880 | 128 | 2880 | 5760 | 4 | 128 | 9.5183 | 1.8011 |
| 3 | 2880 | 2880 | 128 | 2880 | 5760 | 16 | 256 | 75.2635 | 14.4059 |
| 4 | 2880 | 2880 | 128 | 2880 | 5760 | 2 | 512 | 16.6912 | 3.6018 |
| 5 | 2880 | 2880 | 128 | 2880 | 5760 | 1 | 128 | 3.2788 | 0.4506 |
| 6 | 2880 | 2880 | 128 | 2880 | 5760 | 1 | 613 | 12.9344 | 2.1563 |
| 7 | 2880 | 2880 | 128 | 2880 | 5760 | 2 | 128 | 5.0857 | 0.9007 |
| 8 | 2880 | 2880 | 128 | 2880 | 5760 | 2 | 211 | 8.6389 | 1.4846 |
| 9 | 2880 | 2880 | 128 | 2880 | 5760 | 4 | 512 | 29.6217 | 7.2031 |
| 10 | 2880 | 2880 | 128 | 2880 | 5760 | 1 | 2048 | 29.7149 | 7.2031 |
| 11 | 2880 | 2880 | 128 | 2880 | 5760 | 4 | 1024 | 75.6766 | 14.4059 |
| 12 | 2880 | 2880 | 128 | 2880 | 5760 | 1 | 256 | 5.1071 | 0.9007 |
| 13 | 2880 | 2880 | 128 | 2880 | 5760 | 2 | 256 | 9.6203 | 1.8011 |
| 14 | 2880 | 2880 | 128 | 2880 | 5760 | 1 | 4096 | 75.8158 | 14.4059 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """
    Batched MoE expert forward pass.
    
    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        routing_weights: (batch_size * seq_len, num_experts)
        gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
        gate_up_proj_bias: (num_experts, 2 * expert_dim)
        down_proj: (num_experts, expert_dim, hidden_size)
        down_proj_bias: (num_experts, hidden_size)
        alpha: scalar for custom activation
        limit: clamping limit
    
    Returns:
        output: (batch_size, seq_len, hidden_size)
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    num_experts = gate_up_proj.shape[0]
    expert_dim = down_proj.shape[1]
    
    # Flatten batch and sequence dimensions
    # Shape: (batch_size * seq_len, hidden_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    
    # Repeat inputs for all experts
    # Shape: (num_experts * batch_size * seq_len, hidden_size)
    hidden_repeated = hidden_flat.repeat(num_experts, 1)
    
    # Reshape for batched matrix multiplication
    # Shape: (num_experts, batch_size * seq_len, hidden_size)
    hidden_batched = hidden_repeated.view(num_experts, -1, hidden_size)
    
    # Batched gate_up projection
    # bmm: (num_experts, tokens, hidden_size) x (num_experts, hidden_size, 2*expert_dim)
    # Output shape: (num_experts, tokens, 2*expert_dim)
    gate_up = torch.bmm(hidden_batched, gate_up_proj)
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)
    
    # Split into gate and up projections (interleaved)
    # Each shape: (num_experts, tokens, expert_dim)
    gate = gate_up[..., ::2]  # Even indices
    up = gate_up[..., 1::2]   # Odd indices
    
    # Apply clamping to prevent overflow
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    
    # Custom gated GLU activation: gate * sigmoid(gate * alpha) * (up + 1)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    
    # Batched down projection
    # bmm: (num_experts, tokens, expert_dim) x (num_experts, expert_dim, hidden_size)
    # Output shape: (num_experts, tokens, hidden_size)
    expert_outputs = torch.bmm(gated_output, down_proj)
    expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1)
    
    # Reshape for routing weight application
    # Shape: (num_experts, batch_size, seq_len, hidden_size)
    expert_outputs = expert_outputs.view(num_experts, batch_size, seq_len, hidden_size)
    
    # Apply routing weights and sum across experts
    # routing_weights shape: (batch_size * seq_len, num_experts)
    # Reshape to: (num_experts, batch_size, seq_len, 1)
    routing_weights_reshaped = routing_weights.transpose(0, 1).view(
        num_experts, batch_size, seq_len, 1
    )
    
    # Weighted sum across experts
    # Shape: (batch_size, seq_len, hidden_size)
    output = (expert_outputs * routing_weights_reshaped).sum(dim=0)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 3.132083 ms |
| - | Scoring Baseline | 0.500000 | 16.461496 ms |
| - | Reference Implementation | 0.464453 | 18.476117 ms |
