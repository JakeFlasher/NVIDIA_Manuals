## Description

Fused attention Q@K^T computation with GQA key repetition and scaling. Takes query (batch, num_heads=4, seq_len, head_dim=256) and key (batch, num_kv_heads=1, seq_len, head_dim=256), repeats KV heads 4x to match query heads, computes Q@K^T, and applies scaling (1/sqrt(256)=0.0625).
| Name | Shape | Dtype |
| --- | --- | --- |
| query | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| key | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_scores | [batch_size, num_attention_heads, seq_len, seq_len] | bfloat16 |

| # | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1 | 256 | 4 | 2 | 773 | 0.0738 | 0.0022 |
| 2 | 4 | 1 | 256 | 4 | 1 | 293 | 0.0300 | 0.0006 |
| 3 | 4 | 1 | 256 | 4 | 8 | 1024 | 0.0419 | 0.0119 |
| 4 | 4 | 1 | 256 | 4 | 1 | 8192 | 0.1830 | 0.0763 |
| 5 | 4 | 1 | 256 | 4 | 1 | 1024 | 0.0236 | 0.0018 |
| 6 | 4 | 1 | 256 | 4 | 1 | 128 | 0.0232 | 0.0005 |
| 7 | 4 | 1 | 256 | 4 | 2 | 128 | 0.0231 | 0.0005 |
| 8 | 4 | 1 | 256 | 4 | 1 | 1657 | 0.1007 | 0.0038 |
| 9 | 4 | 1 | 256 | 4 | 1 | 2048 | 0.0324 | 0.0055 |
| 10 | 4 | 1 | 256 | 4 | 32 | 256 | 0.0270 | 0.0053 |
| 11 | 4 | 1 | 256 | 4 | 1 | 4096 | 0.0617 | 0.0194 |
| 12 | 4 | 1 | 256 | 4 | 1 | 512 | 0.0231 | 0.0008 |
| 13 | 4 | 1 | 256 | 4 | 1 | 256 | 0.0234 | 0.0006 |
| 14 | 4 | 1 | 256 | 4 | 2 | 1024 | 0.0267 | 0.0033 |
| 15 | 4 | 1 | 256 | 4 | 4 | 2048 | 0.0622 | 0.0206 |
| 16 | 4 | 1 | 256 | 4 | 8 | 256 | 0.0234 | 0.0016 |

```python
import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    """
    Fused attention Q@K^T computation with GQA key repetition and scaling.
    
    Args:
        query: Query tensor of shape (batch, num_attention_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_key_value_heads, seq_len, head_dim)
        scaling: Scaling factor (1/sqrt(query_pre_attn_scalar))
    
    Returns:
        attn_scores: Attention scores of shape (batch, num_attention_heads, seq_len, seq_len)
    """
    batch, num_key_value_heads, slen, head_dim = key.shape
    num_attention_heads = query.shape[1]
    n_rep = num_attention_heads // num_key_value_heads  # 8
    
    # Step 1: Repeat key heads to match query heads (GQA)
    # Expand: (batch, num_kv_heads, 1, seq_len, head_dim) -> (batch, num_kv_heads, n_rep, seq_len, head_dim)
    key_expanded = key[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    # Reshape: (batch, num_kv_heads * n_rep, seq_len, head_dim)
    key_states = key_expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    # Step 2: Compute attention scores Q@K^T
    # query: (batch, num_attention_heads, seq_len, head_dim)
    # key_states.transpose(2, 3): (batch, num_attention_heads, head_dim, seq_len)
    # Result: (batch, num_attention_heads, seq_len, seq_len)
    attn_weights = torch.matmul(query.to(torch.float32), key_states.transpose(2, 3).to(torch.float32))
    
    # Step 3: Apply scaling
    attn_weights = attn_weights * scaling
    
    return attn_weights.to(query.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003065 ms |
| - | Scoring Baseline | 0.500000 | 0.038623 ms |
| - | Reference Implementation | 0.179078 | 0.203860 ms |
