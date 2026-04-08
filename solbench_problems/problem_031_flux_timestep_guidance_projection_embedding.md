## Description

FLUX's combined timestep and text projection embeddings that process timestep through sinusoidal embeddings, then concatenate with pooled text projections and project to inner dimension. This generates the conditioning signal (temb) that modulates all transformer blocks through adaptive layer normalization. The operation involves: (1) timestep scaling by 1000, (2) sinusoidal position embedding generation with frequencies, (3) MLP projection of timestep embeddings, (4) linear projection of pooled text embeddings, (5) addition to produce final conditioning vector.
| Name | Shape | Dtype |
| --- | --- | --- |
| timestep | [batch_size] | float32 |
| pooled_projections | [batch_size, pooled_projection_dim] | float32 |
| freqs | [half_time_embed_dim] | float32 |
| timestep_linear1_weight | [inner_dim, time_embed_dim] | float32 |
| timestep_linear1_bias | [inner_dim] | float32 |
| timestep_linear2_weight | [inner_dim, inner_dim] | float32 |
| timestep_linear2_bias | [inner_dim] | float32 |
| text_embedder_weight | [inner_dim, pooled_projection_dim] | float32 |
| text_embedder_bias | [inner_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| conditioning | [batch_size, inner_dim] | float32 |

| # | inner_dim | pooled_projection_dim | time_embed_dim | half_time_embed_dim | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 768 | 768 | 384 | 20 | 0.1857 | 0.0007 |
| 2 | 3072 | 768 | 768 | 384 | 2 | 0.1282 | 0.0004 |
| 3 | 3072 | 768 | 768 | 384 | 15 | 0.1824 | 0.0006 |
| 4 | 3072 | 768 | 768 | 384 | 9 | 0.2514 | 0.0005 |
| 5 | 3072 | 768 | 768 | 384 | 1 | 0.0754 | 0.0004 |
| 6 | 3072 | 768 | 768 | 384 | 11 | 0.1722 | 0.0006 |
| 7 | 3072 | 768 | 768 | 384 | 47 | 0.2073 | 0.0011 |
| 8 | 3072 | 768 | 768 | 384 | 5 | 0.1472 | 0.0005 |
| 9 | 3072 | 768 | 768 | 384 | 17 | 0.1845 | 0.0007 |
| 10 | 3072 | 768 | 768 | 384 | 29 | 0.2019 | 0.0009 |
| 11 | 3072 | 768 | 768 | 384 | 4 | 0.1254 | 0.0005 |
| 12 | 3072 | 768 | 768 | 384 | 12 | 0.1792 | 0.0006 |
| 13 | 3072 | 768 | 768 | 384 | 37 | 0.2057 | 0.0010 |
| 14 | 3072 | 768 | 768 | 384 | 41 | 0.2071 | 0.0010 |
| 15 | 3072 | 768 | 768 | 384 | 59 | 0.2086 | 0.0013 |
| 16 | 3072 | 768 | 768 | 384 | 24 | 0.1859 | 0.0008 |

```python
import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    inner_dim = axes_and_scalars["inner_dim"]
    pooled_projection_dim = axes_and_scalars["pooled_projection_dim"]
    time_embed_dim = axes_and_scalars["time_embed_dim"]
    half_time_embed_dim = axes_and_scalars["half_time_embed_dim"]
    
    # Timestep values in [0, 1] range
    timestep = torch.rand(batch_size, dtype=torch.float32, device=device)
    
    # Pooled text projections
    pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=torch.float32, device=device)
    
    # Precomputed frequency tensor for sinusoidal embeddings
    # Frequencies: 10000^(-2i/d) for i in [0, d/2)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, time_embed_dim, 2, dtype=torch.float32, device=device) / time_embed_dim
    )
    
    # Timestep MLP weights
    timestep_linear1_weight = torch.randn(inner_dim, time_embed_dim, dtype=torch.float32, device=device) * 0.02
    timestep_linear1_bias = torch.zeros(inner_dim, dtype=torch.float32, device=device)
    timestep_linear2_weight = torch.randn(inner_dim, inner_dim, dtype=torch.float32, device=device) * 0.02
    timestep_linear2_bias = torch.zeros(inner_dim, dtype=torch.float32, device=device)
    
    # Text embedder weights
    text_embedder_weight = torch.randn(inner_dim, pooled_projection_dim, dtype=torch.float32, device=device) * 0.02
    text_embedder_bias = torch.zeros(inner_dim, dtype=torch.float32, device=device)
    
    return {
        "timestep": timestep,
        "pooled_projections": pooled_projections,
        "freqs": freqs,
        "timestep_linear1_weight": timestep_linear1_weight,
        "timestep_linear1_bias": timestep_linear1_bias,
        "timestep_linear2_weight": timestep_linear2_weight,
        "timestep_linear2_bias": timestep_linear2_bias,
        "text_embedder_weight": text_embedder_weight,
        "text_embedder_bias": text_embedder_bias,
    }

@torch.no_grad()
def run(
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    freqs: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
):
    # Scale timesteps by 1000 (FLUX convention)
    timestep_scaled = timestep * 1000.0
    
    # Generate sinusoidal embeddings
    # timestep_scaled shape: (batch_size,)
    # freqs shape: (half_time_embed_dim,)
    # Compute arguments: timestep_scaled[:, None] * freqs[None, :]
    args = timestep_scaled[:, None] * freqs[None, :]  # (batch_size, half_time_embed_dim)
    
    # Compute sin and cos
    sin_embed = torch.sin(args)
    cos_embed = torch.cos(args)
    
    # Concatenate cos and sin: [cos, sin]
    timestep_embed = torch.cat([cos_embed, sin_embed], dim=-1)  # (batch_size, time_embed_dim)
    
    # Timestep MLP: Linear -> SiLU -> Linear
    # First linear layer
    x = torch.nn.functional.linear(timestep_embed, timestep_linear1_weight, timestep_linear1_bias)
    
    # SiLU activation: x * sigmoid(x)
    x = x * torch.sigmoid(x)
    
    # Second linear layer
    timestep_embed = torch.nn.functional.linear(x, timestep_linear2_weight, timestep_linear2_bias)
    
    # Project pooled text embeddings
    text_embed = torch.nn.functional.linear(pooled_projections, text_embedder_weight, text_embedder_bias)
    
    # Combine timestep and text embeddings
    conditioning = timestep_embed + text_embed
    
    return conditioning
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000682 ms |
| - | Scoring Baseline | 0.500000 | 0.172203 ms |
| - | Reference Implementation | 0.490968 | 0.178482 ms |
