## Description

Fused KV cache update with RoPE application. Applies rotary position embeddings to incoming key states and updates both key and value caches in a single operation. Critical for autoregressive generation performance.
| Name | Shape | Dtype |
| --- | --- | --- |
| key_states | [batch_size, num_kv_heads, new_seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_kv_heads, new_seq_len, head_dim] | bfloat16 |
| cos | [batch_size, 1, new_seq_len, head_dim] | bfloat16 |
| sin | [batch_size, 1, new_seq_len, head_dim] | bfloat16 |
| key_cache | [batch_size, num_kv_heads, current_seq_len, head_dim] | bfloat16 |
| value_cache | [batch_size, num_kv_heads, current_seq_len, head_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| updated_key_cache | [batch_size, num_kv_heads, updated_seq_len, head_dim] | bfloat16 |
| updated_value_cache | [batch_size, num_kv_heads, updated_seq_len, head_dim] | bfloat16 |

| # | num_kv_heads | head_dim | batch_size | new_seq_len | current_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 128 | 2 | 512 | 0 | 0.0499 | 0.0021 |
| 2 | 10 | 128 | 64 | 1 | 255 | 0.1580 | 0.0169 |
| 3 | 10 | 128 | 1 | 1 | 0 | 0.0452 | 0.0004 |
| 4 | 10 | 128 | 1 | 1 | 127 | 0.0465 | 0.0005 |
| 5 | 10 | 128 | 4 | 256 | 0 | 0.0483 | 0.0021 |
| 6 | 10 | 128 | 2 | 1 | 510 | 0.0505 | 0.0014 |
| 7 | 10 | 128 | 4 | 1 | 767 | 0.0669 | 0.0035 |
| 8 | 10 | 128 | 32 | 1 | 255 | 0.1001 | 0.0086 |
| 9 | 10 | 128 | 2 | 1024 | 0 | 0.0590 | 0.0039 |
| 10 | 10 | 128 | 2 | 1 | 130 | 0.0463 | 0.0007 |
| 11 | 10 | 128 | 16 | 1 | 996 | 0.1535 | 0.0164 |
| 12 | 10 | 128 | 1 | 128 | 0 | 0.0447 | 0.0006 |
| 13 | 10 | 128 | 1 | 512 | 0 | 0.0475 | 0.0013 |
| 14 | 10 | 128 | 4 | 1 | 372 | 0.0544 | 0.0019 |
| 15 | 10 | 128 | 1 | 1 | 2047 | 0.0620 | 0.0025 |
| 16 | 10 | 128 | 1 | 1 | 511 | 0.0464 | 0.0009 |

```python
import torch

@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Fused KV cache update with RoPE application.
    
    Applies rotary position embeddings to incoming key states and concatenates
    with existing cache to produce updated key and value caches.
    
    Args:
        key_states: New key states (batch, num_kv_heads, new_seq_len, head_dim)
        value_states: New value states (batch, num_kv_heads, new_seq_len, head_dim)
        cos: Cosine position embeddings (batch, 1, new_seq_len, head_dim)
        sin: Sine position embeddings (batch, 1, new_seq_len, head_dim)
        key_cache: Existing key cache (batch, num_kv_heads, current_seq_len, head_dim)
        value_cache: Existing value cache (batch, num_kv_heads, current_seq_len, head_dim)
        
    Returns:
        Tuple of (updated_key_cache, updated_value_cache)
    """
    # Apply RoPE to incoming key states
    # rotate_half: split in half along last dim, negate second half, swap
    head_dim = key_states.shape[-1]
    half_dim = head_dim // 2
    
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    
    # Apply rotation: (k * cos) + (rotate_half(k) * sin)
    key_states_rotated = (key_states * cos) + (k_rotated * sin)
    
    # Concatenate with existing cache along sequence dimension
    updated_key_cache = torch.cat([key_cache, key_states_rotated], dim=2)
    updated_value_cache = torch.cat([value_cache, value_states], dim=2)
    
    return updated_key_cache, updated_value_cache
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002045 ms |
| - | Scoring Baseline | 0.500000 | 0.061112 ms |
| - | Reference Implementation | 0.434863 | 0.079032 ms |
