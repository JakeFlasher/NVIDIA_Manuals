## Description

Fused encoder final RMSNorm with cross-attention K/V projection. Applies RMSNorm to encoder hidden states then projects to cross-attention keys and values, avoiding intermediate materialization of normalized states.
| Name | Shape | Dtype |
| --- | --- | --- |
| encoder_hidden_states | [batch_size, encoder_seq_len, hidden_size] | float16 |
| norm_weight | [hidden_size] | float16 |
| k_proj_weight | [kv_hidden_size, hidden_size] | float16 |
| v_proj_weight | [kv_hidden_size, hidden_size] | float16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| keys | [batch_size, num_kv_heads, encoder_seq_len, head_dim] | float16 |
| values | [batch_size, num_kv_heads, encoder_seq_len, head_dim] | float16 |

| # | hidden_size | num_kv_heads | head_dim | kv_hidden_size | batch_size | encoder_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 2 | 64 | 128 | 64 | 128 | 0.0611 | 0.0097 |
| 2 | 1024 | 2 | 64 | 128 | 32 | 128 | 0.0520 | 0.0050 |
| 3 | 1024 | 2 | 64 | 128 | 1 | 853 | 0.0527 | 0.0014 |
| 4 | 1024 | 2 | 64 | 128 | 1 | 256 | 0.0589 | 0.0007 |
| 5 | 1024 | 2 | 64 | 128 | 4 | 256 | 0.0572 | 0.0016 |
| 6 | 1024 | 2 | 64 | 128 | 2 | 1024 | 0.0528 | 0.0027 |
| 7 | 1024 | 2 | 64 | 128 | 8 | 256 | 0.0580 | 0.0027 |
| 8 | 1024 | 2 | 64 | 128 | 1 | 541 | 0.0661 | 0.0010 |
| 9 | 1024 | 2 | 64 | 128 | 4 | 512 | 0.0729 | 0.0027 |
| 10 | 1024 | 2 | 64 | 128 | 16 | 449 | 0.1087 | 0.0086 |
| 11 | 1024 | 2 | 64 | 128 | 4 | 691 | 0.1112 | 0.0035 |
| 12 | 1024 | 2 | 64 | 128 | 4 | 1024 | 0.0574 | 0.0050 |
| 13 | 1024 | 2 | 64 | 128 | 8 | 512 | 0.0568 | 0.0050 |
| 14 | 1024 | 2 | 64 | 128 | 1 | 131 | 0.0563 | 0.0005 |
| 15 | 1024 | 2 | 64 | 128 | 16 | 512 | 0.0608 | 0.0097 |
| 16 | 1024 | 2 | 64 | 128 | 8 | 613 | 0.0573 | 0.0060 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    """
    Fused encoder final RMSNorm with cross-attention K/V projection.
    
    Args:
        encoder_hidden_states: (batch_size, encoder_seq_len, 1024)
        norm_weight: (1024,) RMSNorm weight
        k_proj_weight: (128, 1024) Key projection weight
        v_proj_weight: (128, 1024) Value projection weight
        eps: RMSNorm epsilon
    
    Returns:
        keys: (batch_size, 2, encoder_seq_len, 64)
        values: (batch_size, 2, encoder_seq_len, 64)
    """
    batch_size, seq_len, hidden_size = encoder_hidden_states.shape
    num_kv_heads = 2
    head_dim = 64
    
    # RMSNorm computation
    input_dtype = encoder_hidden_states.dtype
    hidden_states = encoder_hidden_states.to(torch.float32)
    
    # Compute variance and normalize
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    normalized = (norm_weight * hidden_states).to(input_dtype)
    
    # K/V projections
    # normalized: (batch, seq_len, 1024)
    # weights: (128, 1024)
    # output: (batch, seq_len, 128)
    keys_flat = F.linear(normalized, k_proj_weight, bias=None)
    values_flat = F.linear(normalized, v_proj_weight, bias=None)
    
    # Reshape to multi-head format
    # (batch, seq_len, 128) -> (batch, seq_len, 2, 64) -> (batch, 2, seq_len, 64)
    keys = keys_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    
    values = values_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    values = values.transpose(1, 2).contiguous()
    
    return keys, values
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002949 ms |
| - | Scoring Baseline | 0.500000 | 0.063175 ms |
| - | Reference Implementation | 0.299979 | 0.143270 ms |
