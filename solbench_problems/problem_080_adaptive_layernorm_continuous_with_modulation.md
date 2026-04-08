## Description

AdaLayerNormContinuous normalization with adaptive scale and shift modulation conditioned on timestep embeddings. Computes layer normalization, projects temb through a linear layer to get scale/shift, then modulates: normalized * (1 + scale) + shift.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, inner_dim] | float32 |
| temb | [batch_size, inner_dim] | float32 |
| linear_weight | [inner_dim_x2, inner_dim] | float32 |
| linear_bias | [inner_dim_x2] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, inner_dim] | float32 |

| # | inner_dim | inner_dim_x2 | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2432 | 4864 | 4 | 449 | 0.1145 | 0.0038 |
| 2 | 2432 | 4864 | 2 | 2048 | 0.1241 | 0.0082 |
| 3 | 2432 | 4864 | 1 | 4096 | 0.0995 | 0.0082 |
| 4 | 2432 | 4864 | 1 | 256 | 0.0480 | 0.0009 |
| 5 | 2432 | 4864 | 1 | 1024 | 0.0531 | 0.0023 |
| 6 | 2432 | 4864 | 4 | 256 | 0.0979 | 0.0024 |
| 7 | 2432 | 4864 | 1 | 128 | 0.0474 | 0.0006 |
| 8 | 2432 | 4864 | 2 | 128 | 0.0670 | 0.0009 |
| 9 | 2432 | 4864 | 16 | 256 | 0.1544 | 0.0082 |
| 10 | 2432 | 4864 | 2 | 1321 | 0.0992 | 0.0054 |
| 11 | 2432 | 4864 | 2 | 293 | 0.0704 | 0.0015 |
| 12 | 2432 | 4864 | 1 | 2048 | 0.0703 | 0.0043 |
| 13 | 2432 | 4864 | 1 | 997 | 0.0524 | 0.0023 |
| 14 | 2432 | 4864 | 8 | 1024 | 0.1991 | 0.0160 |
| 15 | 2432 | 4864 | 2 | 512 | 0.0752 | 0.0024 |
| 16 | 2432 | 4864 | 32 | 128 | 0.1705 | 0.0082 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    eps: float,
):
    """
    Adaptive Layer Normalization with continuous modulation.
    
    1. Compute mean and variance across the feature dimension
    2. Normalize: (x - mean) / sqrt(variance + eps)
    3. Apply learned linear projection to temb to get scale and shift
    4. Modulate: normalized * (1 + scale) + shift
    
    Args:
        hidden_states: (batch, seq_len, inner_dim) - Input features to normalize
        temb: (batch, inner_dim) - Timestep conditioning embeddings
        linear_weight: (inner_dim*2, inner_dim) - Weight for temb projection
        linear_bias: (inner_dim*2,) - Bias for temb projection
        eps: Epsilon for numerical stability
    
    Returns:
        (batch, seq_len, inner_dim) - Normalized and modulated features
    """
    # Layer normalization: compute mean and variance
    # Shape: (batch, seq_len, inner_dim)
    mean = hidden_states.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
    variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, 1)
    
    # Normalize
    normalized = (hidden_states - mean) / torch.sqrt(variance + eps)
    
    # Generate scale and shift from temb via linear projection
    # temb: (batch, inner_dim) @ linear_weight.T: (inner_dim, inner_dim*2) -> (batch, inner_dim*2)
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    
    # Split into scale and shift
    # Each is (batch, inner_dim)
    inner_dim = hidden_states.shape[-1]
    scale = modulation[:, :inner_dim]
    shift = modulation[:, inner_dim:]
    
    # Apply modulation: normalized * (1 + scale) + shift
    # Unsqueeze scale and shift to broadcast over seq_len dimension
    # (batch, inner_dim) -> (batch, 1, inner_dim)
    scale = scale.unsqueeze(1)
    shift = shift.unsqueeze(1)
    
    # Final modulated output
    output = normalized * (1.0 + scale) + shift
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003232 ms |
| - | Scoring Baseline | 0.500000 | 0.087290 ms |
| - | Reference Implementation | 0.333953 | 0.170791 ms |
