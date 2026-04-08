## Description

RoPE (Rotary Position Embedding) computation with llama3 scaling that computes cos/sin embeddings from inverse frequencies and position IDs. Uses frequency-based scaling with high_freq_factor=4.0, low_freq_factor=1.0, factor=8.0 for extended context support up to 131072 positions. Returns concatenated cos and sin embeddings.
| Name | Shape | Dtype |
| --- | --- | --- |
| position_ids | [batch_size, seq_len] | int64 |
| inv_freq | [half_head_dim] | float32 |
| attention_scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| cos_sin | [batch_size, seq_len, head_dim, 2] | bfloat16 |

| # | head_dim | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 64 | 16 | 256 | 0.0599 | 0.0007 |
| 2 | 128 | 64 | 1 | 2048 | 0.0620 | 0.0005 |
| 3 | 128 | 64 | 1 | 613 | 0.0579 | 0.0004 |
| 4 | 128 | 64 | 1 | 1024 | 0.0578 | 0.0005 |
| 5 | 128 | 64 | 16 | 373 | 0.0571 | 0.0008 |
| 6 | 128 | 64 | 16 | 919 | 0.0578 | 0.0014 |
| 7 | 128 | 64 | 2 | 131 | 0.0568 | 0.0004 |
| 8 | 128 | 64 | 1 | 256 | 0.0568 | 0.0004 |
| 9 | 128 | 64 | 4 | 256 | 0.0565 | 0.0005 |
| 10 | 128 | 64 | 4 | 2048 | 0.0581 | 0.0009 |
| 11 | 128 | 64 | 4 | 1024 | 0.0569 | 0.0007 |
| 12 | 128 | 64 | 4 | 211 | 0.0566 | 0.0005 |
| 13 | 128 | 64 | 1 | 512 | 0.0579 | 0.0004 |
| 14 | 128 | 64 | 64 | 256 | 0.0579 | 0.0015 |
| 15 | 128 | 64 | 64 | 541 | 0.0706 | 0.0027 |
| 16 | 128 | 64 | 32 | 128 | 0.0568 | 0.0007 |

```python
import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs including precomputed llama3-scaled inverse frequencies."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    head_dim = axes_and_scalars["head_dim"]
    
    # Llama3 RoPE constants
    rope_theta = 500000.0
    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    original_max_position_embeddings = 8192
    
    # Compute inverse frequencies with llama3 scaling
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (rope_theta ** (dim_indices / head_dim))
    
    # Compute wavelengths and scaling factors
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    
    # Wavelength for each frequency component
    wavelens = 2 * math.pi / inv_freq
    
    # Smooth interpolation factor
    smooth_factor = (original_max_position_embeddings / wavelens - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
    
    # Apply scaled frequencies
    scaled_inv_freq = inv_freq / factor
    inv_freq = (1 - smooth_factor) * inv_freq + smooth_factor * scaled_inv_freq
    
    # Generate position_ids (sequential positions for each batch)
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    
    return {
        "position_ids": position_ids,
        "inv_freq": inv_freq,
        "attention_scaling": 1.0
    }

@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float
) -> torch.Tensor:
    """
    Compute rotary position embeddings.
    
    Args:
        position_ids: Position indices (batch_size, seq_len)
        inv_freq: Precomputed inverse frequencies with llama3 scaling (head_dim/2,)
        attention_scaling: Attention scaling factor
    
    Returns:
        cos_sin: Concatenated cos and sin embeddings (batch_size, seq_len, head_dim, 2)
    """
    batch_size = position_ids.shape[0]
    
    # Expand inv_freq for batch dimension: (batch_size, head_dim/2, 1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        batch_size, -1, 1
    )
    
    # Expand position_ids: (batch_size, 1, seq_len)
    position_ids_expanded = position_ids[:, None, :].float()
    
    # Compute frequencies: (batch_size, head_dim/2, seq_len) -> (batch_size, seq_len, head_dim/2)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    
    # Duplicate frequencies for complex number representation
    # (batch_size, seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Compute cos and sin with attention scaling
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    
    # Stack cos and sin along last dimension: (batch_size, seq_len, head_dim, 2)
    cos_sin = torch.stack([cos, sin], dim=-1)
    
    # Convert to bfloat16
    return cos_sin.to(dtype=torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000683 ms |
| 1st place | Clever Scorpion | 0.692693 | 0.026224 ms |
| - | Scoring Baseline | 0.500000 | 0.058502 ms |
| - | Reference Implementation | 0.407541 | 0.084793 ms |
