## Description

Scaled dot-product attention with QK LayerNorm normalization. Query and key tensors are normalized using LayerNorm before computing attention scores. This is used in SD3.5 transformer blocks with 24 heads and 64-dim per head.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, dim] | float32 |
| qkv_weight | [dim_times_3, dim] | float32 |
| qkv_bias | [dim_times_3] | float32 |
| q_norm_weight | [head_dim] | float32 |
| q_norm_bias | [head_dim] | float32 |
| k_norm_weight | [head_dim] | float32 |
| k_norm_bias | [head_dim] | float32 |
| out_proj_weight | [dim, dim] | float32 |
| out_proj_bias | [dim] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, dim] | float32 |

| # | dim | num_heads | head_dim | dim_times_3 | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1536 | 24 | 64 | 4608 | 4 | 293 | 0.2726 | 0.0138 |
| 2 | 1536 | 24 | 64 | 4608 | 16 | 691 | 1.7478 | 0.1415 |
| 3 | 1536 | 24 | 64 | 4608 | 8 | 613 | 0.8645 | 0.0617 |
| 4 | 1536 | 24 | 64 | 4608 | 1 | 2048 | 0.8378 | 0.0360 |
| 5 | 1536 | 24 | 64 | 4608 | 2 | 512 | 0.2744 | 0.0128 |
| 6 | 1536 | 24 | 64 | 4608 | 4 | 128 | 0.1970 | 0.0060 |
| 7 | 1536 | 24 | 64 | 4608 | 4 | 256 | 0.2275 | 0.0120 |
| 8 | 1536 | 24 | 64 | 4608 | 1 | 1024 | 0.3657 | 0.0146 |
| 9 | 1536 | 24 | 64 | 4608 | 4 | 211 | 0.2193 | 0.0098 |
| 10 | 1536 | 24 | 64 | 4608 | 4 | 1024 | 0.9641 | 0.0573 |
| 11 | 1536 | 24 | 64 | 4608 | 16 | 256 | 0.4600 | 0.0466 |
| 12 | 1536 | 24 | 64 | 4608 | 16 | 449 | 1.0283 | 0.0862 |
| 13 | 1536 | 24 | 64 | 4608 | 8 | 256 | 0.3086 | 0.0235 |
| 14 | 1536 | 24 | 64 | 4608 | 4 | 512 | 0.4004 | 0.0253 |
| 15 | 1536 | 24 | 64 | 4608 | 1 | 128 | 0.1482 | 0.0029 |
| 16 | 1536 | 24 | 64 | 4608 | 1 | 256 | 0.1748 | 0.0033 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, dim = hidden_states.shape
    num_heads = 24
    head_dim = 64
    scale = head_dim ** -0.5
    
    # QKV projection: (batch, seq_len, dim) -> (batch, seq_len, 3 * dim)
    qkv = torch.matmul(hidden_states, qkv_weight.t()) + qkv_bias
    
    # Reshape and split: (batch, seq_len, 3 * dim) -> 3 x (batch, num_heads, seq_len, head_dim)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply LayerNorm to Q: normalize across head_dim
    # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    q_mean = q.mean(dim=-1, keepdim=True)
    q_var = q.var(dim=-1, unbiased=False, keepdim=True)
    q_normalized = (q - q_mean) / torch.sqrt(q_var + eps)
    q = q_normalized * q_norm_weight + q_norm_bias
    
    # Apply LayerNorm to K: normalize across head_dim
    k_mean = k.mean(dim=-1, keepdim=True)
    k_var = k.var(dim=-1, unbiased=False, keepdim=True)
    k_normalized = (k - k_mean) / torch.sqrt(k_var + eps)
    k = k_normalized * k_norm_weight + k_norm_bias
    
    # Scale query
    q = q * scale
    
    # Compute attention scores: (batch, num_heads, seq_len, seq_len)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    
    # Softmax over last dimension
    attn_probs = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values: (batch, num_heads, seq_len, head_dim)
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, dim)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    
    # Output projection
    output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.020168 ms |
| - | Scoring Baseline | 0.500000 | 0.403680 ms |
| - | Reference Implementation | 0.382175 | 0.689678 ms |
