## Description

Computes the inverse frequency tensor for RoPE embeddings. Generates frequencies as: inv_freq = 1.0 / (rope_theta^(arange(0, head_dim, 2) / head_dim)). This is a one-time initialization operation that determines the frequency spectrum for rotary position embeddings.
| Name | Shape | Dtype |
| --- | --- | --- |
| rope_theta | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| inv_freq | [half_head_dim] | float32 |

| # | half_head_dim | head_dim | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 64 | 128 | 0.0108 | 0.0004 |
| 2 | 64 | 128 | 0.0104 | 0.0004 |
| 3 | 64 | 128 | 0.0107 | 0.0004 |
| 4 | 64 | 128 | 0.0104 | 0.0004 |
| 5 | 64 | 128 | 0.0104 | 0.0004 |
| 6 | 64 | 128 | 0.0103 | 0.0004 |
| 7 | 64 | 128 | 0.0104 | 0.0004 |
| 8 | 64 | 128 | 0.0105 | 0.0004 |
| 9 | 64 | 128 | 0.0107 | 0.0004 |
| 10 | 64 | 128 | 0.0104 | 0.0004 |
| 11 | 64 | 128 | 0.0103 | 0.0004 |
| 12 | 64 | 128 | 0.0105 | 0.0004 |
| 13 | 64 | 128 | 0.0104 | 0.0004 |
| 14 | 64 | 128 | 0.0104 | 0.0004 |
| 15 | 64 | 128 | 0.0107 | 0.0004 |
| 16 | 64 | 128 | 0.0103 | 0.0004 |

```python
import torch

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    """
    Computes inverse frequency tensor for Rotary Position Embeddings.
    
    inv_freq[i] = 1.0 / (rope_theta^(2*i / head_dim))
    for i in [0, 1, 2, ..., head_dim//2 - 1]
    
    This is equivalent to:
    inv_freq = 1.0 / (rope_theta^(arange(0, head_dim, 2) / head_dim))
    
    Args:
        rope_theta: Base frequency for RoPE computation (e.g., 1000000.0)
        
    Returns:
        Inverse frequency tensor of shape (head_dim // 2,) = (64,)
    """
    head_dim = 128
    
    # Create range [0, 2, 4, ..., head_dim-2] = [0, 2, 4, ..., 126]
    # Shape: (head_dim // 2,) = (64,)
    indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device='cuda')
    
    # Compute exponents: indices / head_dim
    # Shape: (64,)
    exponents = indices / float(head_dim)
    
    # Compute theta^exponents
    # Shape: (64,)
    theta_powers = torch.pow(float(rope_theta), exponents)
    
    # Compute inverse: 1.0 / theta^exponents
    # Shape: (64,)
    inv_freq = 1.0 / theta_powers
    
    return inv_freq
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000400 ms |
| - | Scoring Baseline | 0.500000 | 0.010475 ms |
| 1st place | Clever Scorpion | 0.352045 | 0.018946 ms |
| - | Reference Implementation | 0.206226 | 0.039184 ms |
