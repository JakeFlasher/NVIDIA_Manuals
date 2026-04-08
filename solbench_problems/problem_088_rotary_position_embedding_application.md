## Description

Rotary Position Embedding (RoPE) application that rotates query and key vectors in attention mechanisms based on their positions. Splits the feature dimension into pairs, applies rotation matrices derived from position indices and frequency bands. Critical for diffusion transformers as it provides position information without learned parameters.
| Name | Shape | Dtype |
| --- | --- | --- |
| query | [batch_size, num_heads, seq_len, head_dim] | float32 |
| key | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |
| cos | [seq_len, head_dim] | float32 |
| sin | [seq_len, head_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_rotated | [batch_size, num_heads, seq_len, head_dim] | float32 |
| key_rotated | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |

| # | head_dim | half_head_dim | batch_size | num_heads | num_kv_heads | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 64 | 4 | 24 | 6 | 293 | 0.1046 | 0.0028 |
| 2 | 128 | 64 | 32 | 16 | 4 | 4096 | 0.6299 | 0.1757 |
| 3 | 128 | 64 | 2 | 40 | 8 | 4096 | 0.1398 | 0.0269 |
| 4 | 128 | 64 | 8 | 24 | 8 | 1024 | 0.1032 | 0.0180 |
| 5 | 128 | 64 | 8 | 16 | 4 | 373 | 0.0983 | 0.0044 |
| 6 | 128 | 64 | 2 | 48 | 6 | 919 | 0.0996 | 0.0071 |
| 7 | 128 | 64 | 64 | 8 | 8 | 128 | 0.0988 | 0.0092 |
| 8 | 128 | 64 | 8 | 32 | 4 | 1087 | 0.1118 | 0.0214 |
| 9 | 128 | 64 | 1 | 40 | 8 | 131 | 0.0994 | 0.0008 |
| 10 | 128 | 64 | 4 | 48 | 12 | 1321 | 0.1116 | 0.0217 |
| 11 | 128 | 64 | 1 | 64 | 8 | 8192 | 0.1788 | 0.0403 |
| 12 | 128 | 64 | 4 | 36 | 6 | 2048 | 0.1450 | 0.0235 |
| 13 | 128 | 64 | 2 | 32 | 4 | 211 | 0.0993 | 0.0014 |
| 14 | 128 | 64 | 1 | 56 | 8 | 853 | 0.0993 | 0.0041 |
| 15 | 128 | 64 | 1 | 8 | 8 | 128 | 0.0983 | 0.0005 |
| 16 | 128 | 64 | 8 | 28 | 4 | 1024 | 0.1055 | 0.0180 |

```python
import torch

@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Apply rotary position embeddings to query and key tensors.
    
    RoPE rotation formula: x_rotated = x * cos + rotate_half(x) * sin
    where rotate_half([x0, x1, x2, x3, ...]) = [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim)
    
    Returns:
        Tuple of (rotated_query, rotated_key) with same shapes as inputs
    """
    head_dim = query.shape[-1]
    half_dim = head_dim // 2
    
    # Reshape cos and sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Rotate half function: for input [x0, x1, ..., x_{d-1}]
    # returns [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    query_rotated = query * cos + rotate_half(query) * sin
    key_rotated = key * cos + rotate_half(key) * sin
    
    return query_rotated, key_rotated
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.008779 ms |
| 1st place | Mystic Orca | 0.767525 | 0.039999 ms |
| - | Scoring Baseline | 0.500000 | 0.123756 ms |
| - | Reference Implementation | 0.263366 | 0.373107 ms |
