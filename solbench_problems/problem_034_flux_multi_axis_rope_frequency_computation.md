## Description

Multi-axis rotary position embedding frequency computation that generates cos/sin frequency tensors for 3D position encoding (time, height, width). Computes 1D RoPE for each axis with dimensions [16, 56, 56] and concatenates them. Involves frequency band calculation, outer products, sin/cos evaluation, repeat_interleave operations, and concatenation across axes.
| Name | Shape | Dtype |
| --- | --- | --- |
| ids | [seq_len, n_axes] | float32 |
| theta | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| freqs_cos | [seq_len, total_dim] | float32 |
| freqs_sin | [seq_len, total_dim] | float32 |

| # | n_axes | axis0_dim | axis1_dim | axis2_dim | total_dim | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 56 | 56 | 128 | 997 | 0.0596 | 0.0005 |
| 2 | 3 | 16 | 56 | 56 | 128 | 373 | 0.0573 | 0.0004 |
| 3 | 3 | 16 | 56 | 56 | 128 | 1163 | 0.0575 | 0.0005 |
| 4 | 3 | 16 | 56 | 56 | 128 | 3967 | 0.0570 | 0.0007 |
| 5 | 3 | 16 | 56 | 56 | 128 | 3319 | 0.0611 | 0.0006 |
| 6 | 3 | 16 | 56 | 56 | 128 | 211 | 0.0566 | 0.0004 |
| 7 | 3 | 16 | 56 | 56 | 128 | 1024 | 0.0574 | 0.0005 |
| 8 | 3 | 16 | 56 | 56 | 128 | 4096 | 0.0567 | 0.0007 |
| 9 | 3 | 16 | 56 | 56 | 128 | 1321 | 0.0588 | 0.0005 |
| 10 | 3 | 16 | 56 | 56 | 128 | 293 | 0.0569 | 0.0004 |
| 11 | 3 | 16 | 56 | 56 | 128 | 691 | 0.0567 | 0.0004 |
| 12 | 3 | 16 | 56 | 56 | 128 | 853 | 0.0567 | 0.0005 |
| 13 | 3 | 16 | 56 | 56 | 128 | 131 | 0.1100 | 0.0004 |
| 14 | 3 | 16 | 56 | 56 | 128 | 1087 | 0.0572 | 0.0005 |
| 15 | 3 | 16 | 56 | 56 | 128 | 4093 | 0.0566 | 0.0007 |
| 16 | 3 | 16 | 56 | 56 | 128 | 3011 | 0.0568 | 0.0006 |

```python
import torch

@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    """
    Multi-axis RoPE frequency computation.
    
    Args:
        ids: Position IDs of shape [seq_len, 3] containing [time, height, width] positions
        theta: Base frequency for RoPE computation
    
    Returns:
        Tuple of (freqs_cos, freqs_sin) each of shape [seq_len, total_dim]
        where total_dim = 16 + 56 + 56 = 128
    """
    # Fixed axis dimensions for Flux
    axes_dim = [16, 56, 56]
    n_axes = 3
    
    seq_len = ids.shape[0]
    device = ids.device
    
    freqs_dtype = torch.float32
    
    cos_list = []
    sin_list = []
    
    # Convert ids to float for computation
    pos = ids.float()
    
    # Process each axis independently
    for axis_idx in range(n_axes):
        dim = axes_dim[axis_idx]
        
        # Get positions for this axis: [seq_len]
        axis_pos = pos[:, axis_idx]
        
        # Compute frequency bands for this dimension
        # freq_bands shape: [dim // 2]
        half_dim = dim // 2
        freq_exponents = torch.arange(half_dim, dtype=freqs_dtype, device=device)
        freq_bands = 1.0 / (theta ** (freq_exponents / half_dim))
        
        # Compute angles: [seq_len, dim // 2]
        # Outer product of positions and frequency bands
        angles = axis_pos.unsqueeze(-1).to(freqs_dtype) * freq_bands.unsqueeze(0)
        
        # Compute cos and sin
        cos_angles = torch.cos(angles)  # [seq_len, dim // 2]
        sin_angles = torch.sin(angles)  # [seq_len, dim // 2]
        
        # Repeat interleave to match real RoPE format
        # Each frequency is repeated twice: [f1, f1, f2, f2, ...]
        cos_interleaved = torch.repeat_interleave(cos_angles, 2, dim=-1)  # [seq_len, dim]
        sin_interleaved = torch.repeat_interleave(sin_angles, 2, dim=-1)  # [seq_len, dim]
        
        cos_list.append(cos_interleaved)
        sin_list.append(sin_interleaved)
    
    # Concatenate all axes: [seq_len, total_dim]
    freqs_cos = torch.cat(cos_list, dim=-1)
    freqs_sin = torch.cat(sin_list, dim=-1)
    
    return freqs_cos, freqs_sin
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000508 ms |
| - | Scoring Baseline | 0.500000 | 0.059885 ms |
| - | Reference Implementation | 0.188754 | 0.258287 ms |
