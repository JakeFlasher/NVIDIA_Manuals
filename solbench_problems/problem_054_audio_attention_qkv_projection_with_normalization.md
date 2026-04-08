## Description

Fused QKV projection with per-head RMS normalization for audio attention. Projects hidden states to Q, K, V representations, reshapes to multi-head format, and applies separate RMS normalization to each head. This is a unique pattern where normalization is applied per-head rather than per-sequence.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| q_proj_weight | [qkv_out_size, hidden_size] | float32 |
| k_proj_weight | [qkv_out_size, hidden_size] | float32 |
| v_proj_weight | [qkv_out_size, hidden_size] | float32 |
| q_norm_weight | [head_dim] | float32 |
| k_norm_weight | [head_dim] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, seq_len, num_heads, head_dim] | float32 |
| key_states | [batch_size, seq_len, num_heads, head_dim] | float32 |
| value_states | [batch_size, seq_len, num_heads, head_dim] | float32 |

| # | hidden_size | num_heads | head_dim | qkv_out_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 128 | 1024 | 2 | 256 | 0.1632 | 0.0022 |
| 2 | 1024 | 8 | 128 | 1024 | 2 | 128 | 0.1267 | 0.0014 |
| 3 | 1024 | 8 | 128 | 1024 | 1 | 128 | 0.1701 | 0.0013 |
| 4 | 1024 | 8 | 128 | 1024 | 8 | 512 | 0.8083 | 0.0146 |
| 5 | 1024 | 8 | 128 | 1024 | 8 | 773 | 1.0565 | 0.0219 |
| 6 | 1024 | 8 | 128 | 1024 | 64 | 128 | 1.5214 | 0.0289 |
| 7 | 1024 | 8 | 128 | 1024 | 1 | 131 | 0.0991 | 0.0013 |
| 8 | 1024 | 8 | 128 | 1024 | 4 | 256 | 0.2321 | 0.0040 |
| 9 | 1024 | 8 | 128 | 1024 | 1 | 4096 | 0.7554 | 0.0146 |
| 10 | 1024 | 8 | 128 | 1024 | 16 | 512 | 1.5231 | 0.0289 |
| 11 | 1024 | 8 | 128 | 1024 | 2 | 293 | 0.1858 | 0.0024 |
| 12 | 1024 | 8 | 128 | 1024 | 4 | 512 | 0.4075 | 0.0075 |
| 13 | 1024 | 8 | 128 | 1024 | 2 | 1024 | 0.4075 | 0.0075 |
| 14 | 1024 | 8 | 128 | 1024 | 4 | 128 | 0.1614 | 0.0022 |
| 15 | 1024 | 8 | 128 | 1024 | 4 | 541 | 0.4109 | 0.0079 |
| 16 | 1024 | 8 | 128 | 1024 | 2 | 2048 | 0.7560 | 0.0146 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
):
    """
    Fused QKV projection with per-head RMS normalization.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size=1024]
        q_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        k_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        v_proj_weight: [qkv_out_size=1024, hidden_size=1024]
        q_norm_weight: [head_dim=128]
        k_norm_weight: [head_dim=128]
        eps: epsilon for RMS norm
    
    Returns:
        query_states: [batch_size, seq_len, num_heads=8, head_dim=128]
        key_states: [batch_size, seq_len, num_heads=8, head_dim=128]
        value_states: [batch_size, seq_len, num_heads=8, head_dim=128]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 8
    head_dim = 128
    
    # Linear projections: [batch, seq, hidden] @ [hidden, qkv_out].T -> [batch, seq, qkv_out]
    # Using matmul with transposed weights
    query = torch.matmul(hidden_states, q_proj_weight.t())
    key = torch.matmul(hidden_states, k_proj_weight.t())
    value = torch.matmul(hidden_states, v_proj_weight.t())
    
    # Reshape to multi-head format: [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)
    
    # Per-head RMS normalization for query
    # Compute variance over head_dim (last dimension)
    q_variance = query.pow(2).mean(dim=-1, keepdim=True)
    q_normed = query / torch.sqrt(q_variance + eps)
    query_states = q_normed * q_norm_weight
    
    # Per-head RMS normalization for key
    k_variance = key.pow(2).mean(dim=-1, keepdim=True)
    k_normed = key / torch.sqrt(k_variance + eps)
    key_states = k_normed * k_norm_weight
    
    # No RMS normalization for value in audio attention
    value_states = value
    
    return query_states, key_states, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.005995 ms |
| - | Scoring Baseline | 0.500000 | 0.379753 ms |
| - | Reference Implementation | 0.441912 | 0.481319 ms |
