## Description

Per-head RMSNorm applied to query and key tensors after projection in Flux attention. Applies RMSNorm independently to each attention head's Q and K vectors, normalizing across head_dim (128) for each of the 48 heads. This dual variant processes both Q and K tensors together to enable kernel fusion opportunities.
| Name | Shape | Dtype |
| --- | --- | --- |
| query | [batch_size, seq_len, num_heads, head_dim] | float32 |
| key | [batch_size, seq_len, num_heads, head_dim] | float32 |
| weight_q | [num_heads, head_dim] | float32 |
| weight_k | [num_heads, head_dim] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_norm | [batch_size, seq_len, num_heads, head_dim] | float32 |
| key_norm | [batch_size, seq_len, num_heads, head_dim] | float32 |

| # | num_heads | head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 48 | 128 | 2 | 128 | 0.0532 | 0.0020 |
| 2 | 48 | 128 | 4 | 1657 | 0.1379 | 0.0429 |
| 3 | 48 | 128 | 4 | 1024 | 0.0956 | 0.0267 |
| 4 | 48 | 128 | 8 | 773 | 0.1306 | 0.0400 |
| 5 | 48 | 128 | 8 | 128 | 0.0525 | 0.0070 |
| 6 | 48 | 128 | 2 | 293 | 0.0529 | 0.0042 |
| 7 | 48 | 128 | 16 | 256 | 0.0965 | 0.0267 |
| 8 | 48 | 128 | 1 | 1024 | 0.0520 | 0.0070 |
| 9 | 48 | 128 | 4 | 256 | 0.0552 | 0.0070 |
| 10 | 48 | 128 | 1 | 128 | 0.0521 | 0.0012 |
| 11 | 48 | 128 | 32 | 128 | 0.0950 | 0.0267 |
| 12 | 48 | 128 | 4 | 512 | 0.0699 | 0.0135 |
| 13 | 48 | 128 | 1 | 131 | 0.0522 | 0.0012 |
| 14 | 48 | 128 | 8 | 256 | 0.0669 | 0.0135 |
| 15 | 48 | 128 | 1 | 8192 | 0.1780 | 0.0529 |
| 16 | 48 | 128 | 1 | 256 | 0.0517 | 0.0020 |

```python
import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor, eps: float):
    """
    Per-head RMSNorm for both query and key tensors.
    
    Args:
        query: Query tensor [batch_size, seq_len, num_heads, head_dim]
        key: Key tensor [batch_size, seq_len, num_heads, head_dim]
        weight_q: Per-head scale for query [num_heads, head_dim]
        weight_k: Per-head scale for key [num_heads, head_dim]
        eps: Epsilon for numerical stability
    
    Returns:
        Tuple of (normalized_query, normalized_key)
    """
    input_dtype = query.dtype
    
    # Compute RMS for query per head (normalize over head_dim dimension)
    # variance shape: [batch, seq_len, num_heads, 1]
    q_variance = query.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    query_norm = query * torch.rsqrt(q_variance + eps)
    
    # Compute RMS for key per head
    k_variance = key.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    key_norm = key * torch.rsqrt(k_variance + eps)
    
    # Apply learnable scales
    # weight shape: [num_heads, head_dim]
    # Broadcast to [batch, seq_len, num_heads, head_dim]
    query_norm = query_norm * weight_q.unsqueeze(0).unsqueeze(0)
    key_norm = key_norm * weight_k.unsqueeze(0).unsqueeze(0)
    
    return query_norm.to(input_dtype), key_norm.to(input_dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.009195 ms |
| - | Scoring Baseline | 0.500000 | 0.073867 ms |
| - | Reference Implementation | 0.213499 | 0.257908 ms |
