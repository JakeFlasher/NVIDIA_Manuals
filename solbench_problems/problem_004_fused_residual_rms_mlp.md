## Description

Fused operation combining residual connection, RMSNorm, and complete SwiGLU MLP block. This pattern appears at every decoder layer after the attention block where the attention output residual is added, then immediately normalized, and fed into the MLP (gate_proj, up_proj, SiLU activation, element-wise multiply, down_proj).
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| norm_weight | [hidden_size] | bfloat16 |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16384 | 53248 | 2 | 4096 | 33.1216 | 23.6752 |
| 2 | 16384 | 53248 | 1 | 256 | 1.2292 | 0.7402 |
| 3 | 16384 | 53248 | 1 | 4096 | 16.0503 | 11.8378 |
| 4 | 16384 | 53248 | 1 | 1024 | 3.9699 | 2.9597 |
| 5 | 16384 | 53248 | 64 | 128 | 33.5112 | 23.6752 |
| 6 | 16384 | 53248 | 8 | 512 | 16.2164 | 11.8378 |
| 7 | 16384 | 53248 | 16 | 128 | 8.5234 | 5.9191 |
| 8 | 16384 | 53248 | 1 | 2048 | 8.4308 | 5.9191 |
| 9 | 16384 | 53248 | 1 | 1571 | 6.5376 | 4.5406 |
| 10 | 16384 | 53248 | 4 | 373 | 5.9623 | 4.3123 |
| 11 | 16384 | 53248 | 1 | 2053 | 8.7640 | 5.9335 |
| 12 | 16384 | 53248 | 8 | 256 | 8.4248 | 5.9191 |
| 13 | 16384 | 53248 | 1 | 997 | 4.0287 | 2.8817 |
| 14 | 16384 | 53248 | 2 | 2048 | 16.3761 | 11.8378 |
| 15 | 16384 | 53248 | 16 | 256 | 16.3166 | 11.8378 |
| 16 | 16384 | 53248 | 4 | 1024 | 16.3156 | 11.8378 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
):
    """
    Fused residual + RMSNorm + SwiGLU MLP.
    
    Args:
        hidden_states: Output from attention block [batch_size, seq_len, hidden_size]
        residual: Residual connection from before attention [batch_size, seq_len, hidden_size]
        norm_weight: RMSNorm weight [hidden_size]
        gate_proj_weight: Gate projection weight [intermediate_size, hidden_size]
        up_proj_weight: Up projection weight [intermediate_size, hidden_size]
        down_proj_weight: Down projection weight [hidden_size, intermediate_size]
        eps: Epsilon for numerical stability
        
    Returns:
        Output tensor after MLP [batch_size, seq_len, hidden_size]
    """
    # Step 1: Residual connection
    # Shape: [batch_size, seq_len, hidden_size]
    hidden_states = residual + hidden_states
    
    # Step 2: RMSNorm
    # Convert to float32 for numerical stability
    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    
    # Compute variance: mean of squared values
    # Shape: [batch_size, seq_len, 1]
    variance = hidden_states_f32.pow(2).mean(dim=-1, keepdim=True)
    
    # Normalize and scale
    # Shape: [batch_size, seq_len, hidden_size]
    hidden_states_f32 = hidden_states_f32 * torch.rsqrt(variance + eps)
    hidden_states = (norm_weight * hidden_states_f32).to(input_dtype)
    
    # Step 3: SwiGLU MLP
    # Gate projection: [batch_size, seq_len, hidden_size] @ [hidden_size, intermediate_size]
    # Shape: [batch_size, seq_len, intermediate_size]
    gate_output = F.linear(hidden_states, gate_proj_weight)
    
    # Up projection
    # Shape: [batch_size, seq_len, intermediate_size]
    up_output = F.linear(hidden_states, up_proj_weight)
    
    # SwiGLU activation: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    # Shape: [batch_size, seq_len, intermediate_size]
    intermediate = F.silu(gate_output) * up_output
    
    # Down projection
    # Shape: [batch_size, seq_len, hidden_size]
    output = F.linear(intermediate, down_proj_weight)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 6.777476 ms |
| - | Scoring Baseline | 0.500000 | 9.590333 ms |
| - | Reference Implementation | 0.461599 | 10.048956 ms |
