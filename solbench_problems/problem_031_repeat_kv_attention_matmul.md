## Description

Fused Grouped Query Attention (GQA) key-value repetition and attention score computation. This operation repeats key tensors from num_key_value_heads to num_attention_heads (4x expansion), then computes scaled attention scores via Q@K^T. The fusion avoids materializing the expanded key tensor, saving 75% of intermediate memory bandwidth.
| Name | Shape | Dtype |
| --- | --- | --- |
| query | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| key | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_weights | [batch_size, num_attention_heads, seq_len, seq_len] | bfloat16 |

| # | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 32 | 128 | 1 | 32 | 128 | 0.0350 | 0.0135 |
| 2 | 32 | 32 | 128 | 1 | 1 | 128 | 0.0258 | 0.0008 |
| 3 | 32 | 32 | 128 | 1 | 1 | 512 | 0.0264 | 0.0037 |
| 4 | 32 | 32 | 128 | 1 | 4 | 1024 | 0.0898 | 0.0442 |
| 5 | 32 | 32 | 128 | 1 | 1 | 1024 | 0.0380 | 0.0113 |
| 6 | 32 | 32 | 128 | 1 | 16 | 128 | 0.0272 | 0.0070 |
| 7 | 32 | 32 | 128 | 1 | 1 | 1321 | 0.0985 | 0.0178 |
| 8 | 32 | 32 | 128 | 1 | 1 | 4096 | 0.2786 | 0.1492 |
| 9 | 32 | 32 | 128 | 1 | 2 | 512 | 0.0319 | 0.0070 |
| 10 | 32 | 32 | 128 | 1 | 2 | 449 | 0.0594 | 0.0057 |
| 11 | 32 | 32 | 128 | 1 | 2 | 128 | 0.0271 | 0.0012 |
| 12 | 32 | 32 | 128 | 1 | 2 | 2048 | 0.1546 | 0.0792 |
| 13 | 32 | 32 | 128 | 1 | 1 | 256 | 0.0259 | 0.0015 |
| 14 | 32 | 32 | 128 | 1 | 1 | 131 | 0.0305 | 0.0008 |
| 15 | 32 | 32 | 128 | 1 | 2 | 4096 | 0.5286 | 0.2980 |
| 16 | 32 | 32 | 128 | 1 | 2 | 1571 | 0.2142 | 0.0483 |

```python
import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Fused GQA key-value repetition and attention score computation.
    
    Args:
        query: [batch_size, num_attention_heads, seq_len, head_dim]
        key: [batch_size, num_key_value_heads, seq_len, head_dim]
    
    Returns:
        attn_weights: [batch_size, num_attention_heads, seq_len, seq_len]
    """
    # Constants
    head_dim = 128
    num_key_value_groups = 1
    scaling = head_dim ** -0.5
    
    batch_size, num_key_value_heads, slen, _ = key.shape
    
    # Repeat KV heads to match query heads
    # [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]
    key_expanded = key[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, slen, head_dim
    )
    key_states = key_expanded.reshape(batch_size, num_key_value_heads * num_key_value_groups, slen, head_dim)
    
    # Compute attention scores: Q @ K^T with scaling
    # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # -> [batch, num_heads, seq_len, seq_len]
    attn_weights = torch.matmul(query.to(torch.float32), key_states.transpose(2, 3).to(torch.float32)) * scaling
    
    return attn_weights.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.010571 ms |
| - | Scoring Baseline | 0.500000 | 0.061609 ms |
| - | Reference Implementation | 0.127735 | 0.428247 ms |
