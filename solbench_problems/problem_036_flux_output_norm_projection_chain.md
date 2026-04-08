## Description

Flux's final output processing chain combining AdaLayerNormContinuous normalization with output projection. Performs: (1) LayerNorm without affine parameters, (2) SiLU activation on temb followed by linear projection to extract shift/scale modulation parameters, (3) element-wise modulation (shift and scale), (4) large linear projection to output space.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, inner_dim] | float32 |
| temb | [batch_size, inner_dim] | float32 |
| linear_weight | [modulation_dim, inner_dim] | float32 |
| linear_bias | [modulation_dim] | float32 |
| proj_out_weight | [output_dim, inner_dim] | float32 |
| proj_out_bias | [output_dim] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, output_dim] | float32 |

| # | inner_dim | output_dim | modulation_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 64 | 6144 | 2 | 293 | 0.0896 | 0.0023 |
| 2 | 3072 | 64 | 6144 | 32 | 256 | 0.1745 | 0.0269 |
| 3 | 3072 | 64 | 6144 | 1 | 2053 | 0.0978 | 0.0070 |
| 4 | 3072 | 64 | 6144 | 1 | 128 | 0.0956 | 0.0008 |
| 5 | 3072 | 64 | 6144 | 32 | 128 | 0.1380 | 0.0137 |
| 6 | 3072 | 64 | 6144 | 8 | 256 | 0.1207 | 0.0070 |
| 7 | 3072 | 64 | 6144 | 4 | 512 | 0.1189 | 0.0070 |
| 8 | 3072 | 64 | 6144 | 2 | 1571 | 0.1289 | 0.0105 |
| 9 | 3072 | 64 | 6144 | 1 | 2048 | 0.0958 | 0.0070 |
| 10 | 3072 | 64 | 6144 | 2 | 4096 | 0.1754 | 0.0268 |
| 11 | 3072 | 64 | 6144 | 16 | 128 | 0.1173 | 0.0070 |
| 12 | 3072 | 64 | 6144 | 2 | 1024 | 0.1177 | 0.0070 |
| 13 | 3072 | 64 | 6144 | 4 | 256 | 0.1140 | 0.0037 |
| 14 | 3072 | 64 | 6144 | 16 | 512 | 0.1734 | 0.0268 |
| 15 | 3072 | 64 | 6144 | 4 | 1024 | 0.1358 | 0.0136 |
| 16 | 3072 | 64 | 6144 | 1 | 1024 | 0.0947 | 0.0037 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    """
    Flux output normalization and projection chain.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, inner_dim)
        temb: Timestep embeddings of shape (batch_size, inner_dim)
        linear_weight: Weight for modulation projection (2*inner_dim, inner_dim)
        linear_bias: Bias for modulation projection (2*inner_dim,)
        proj_out_weight: Weight for output projection (output_dim, inner_dim)
        proj_out_bias: Bias for output projection (output_dim,)
        eps: Epsilon for LayerNorm numerical stability
        
    Returns:
        Output tensor of shape (batch_size, seq_len, output_dim)
    """
    # Step 1: Layer normalization without affine parameters
    # Compute mean and variance along the last dimension
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states_norm = (hidden_states - mean) / torch.sqrt(var + eps)
    
    # Step 2: SiLU activation on temb followed by linear projection
    # SiLU: x * sigmoid(x)
    temb_silu = temb * torch.sigmoid(temb)
    # Linear projection: (batch_size, inner_dim) @ (inner_dim, 2*inner_dim) + bias
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
    
    # Split into shift and scale
    inner_dim = temb.shape[-1]
    shift = modulation[:, :inner_dim]  # (batch_size, inner_dim)
    scale = modulation[:, inner_dim:]  # (batch_size, inner_dim)
    
    # Step 3: Apply adaptive modulation
    # Expand shift and scale to match hidden_states shape
    shift = shift.unsqueeze(1)  # (batch_size, 1, inner_dim)
    scale = scale.unsqueeze(1)  # (batch_size, 1, inner_dim)
    hidden_states_mod = hidden_states_norm * (1.0 + scale) + shift
    
    # Step 4: Large output projection
    # (batch_size, seq_len, inner_dim) @ (inner_dim, output_dim) + bias
    output = F.linear(hidden_states_mod, proj_out_weight, proj_out_bias)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.007552 ms |
| - | Scoring Baseline | 0.500000 | 0.121311 ms |
| - | Reference Implementation | 0.324579 | 0.244408 ms |
