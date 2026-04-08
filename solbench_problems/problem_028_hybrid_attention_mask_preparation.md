## Description

Hybrid causal mask preparation system that creates separate attention masks for full attention and sliding window attention layers. Creates two distinct 4D causal masks: standard causal masking for full attention layers and sliding window causal masking for SWA layers.
| Name | Shape | Dtype |
| --- | --- | --- |
| batch_size_scalar | scalar | int64 |
| seq_length_scalar | scalar | int64 |
| past_key_values_length_scalar | scalar | int64 |

| Name | Shape | Dtype |
| --- | --- | --- |
| full_attention_mask | [batch_size, num_attention_heads, seq_length, total_length] | bool |
| sliding_window_attention_mask | [batch_size, swa_num_attention_heads, seq_length, total_length] | bool |

| # | num_attention_heads | swa_num_attention_heads | sliding_window | batch_size | seq_length | past_key_values_length | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 64 | 128 | 4 | 16 | 240 | 0.0192 | 0.0004 |
| 2 | 64 | 64 | 128 | 1 | 128 | 0 | 0.0190 | 0.0004 |
| 3 | 64 | 64 | 128 | 8 | 1024 | 1024 | 0.0196 | 0.0004 |
| 4 | 64 | 64 | 128 | 2 | 32 | 96 | 0.0188 | 0.0004 |
| 5 | 64 | 64 | 128 | 2 | 4096 | 4096 | 0.0724 | 0.0004 |
| 6 | 64 | 64 | 128 | 8 | 997 | 0 | 0.0192 | 0.0004 |
| 7 | 64 | 64 | 128 | 1 | 131 | 0 | 0.0186 | 0.0004 |
| 8 | 64 | 64 | 128 | 32 | 256 | 128 | 0.0187 | 0.0004 |
| 9 | 64 | 64 | 128 | 16 | 2048 | 0 | 0.0206 | 0.0004 |
| 10 | 64 | 64 | 128 | 4 | 1571 | 1024 | 0.0224 | 0.0004 |
| 11 | 64 | 64 | 128 | 64 | 128 | 0 | 0.0186 | 0.0004 |
| 12 | 64 | 64 | 128 | 8 | 3089 | 0 | 0.0296 | 0.0004 |

```python
import torch

@torch.no_grad()
def run(
    batch_size_scalar: int,
    seq_length_scalar: int,
    past_key_values_length_scalar: int,
):
    """
    Hybrid attention mask preparation for models with mixed full and sliding window attention.
    
    Creates two types of causal masks:
    1. Full causal mask: Standard lower-triangular mask for full attention layers
    2. Sliding window causal mask: Banded diagonal mask for sliding window attention layers
    """
    # Constants
    num_attention_heads = 64
    swa_num_attention_heads = 64
    sliding_window = 128
    dtype = torch.bool
    device = torch.device('cuda')
    
    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past_key_values_length = int(past_key_values_length_scalar)
    
    target_length = seq_length
    source_length = seq_length + past_key_values_length
    
    # Create full causal mask
    full_mask = torch.ones(
        (target_length, source_length),
        dtype=dtype,
        device=device,
    )
    
    # Make it causal (lower triangular)
    target_indices = torch.arange(target_length, device=device)[:, None]
    source_indices = torch.arange(source_length, device=device)[None, :]
    causal_cond = target_indices >= (source_indices - past_key_values_length)
    full_mask = full_mask.masked_fill(causal_cond, False)
    
    # Expand to [batch_size, num_heads, seq_length, source_length]
    full_attention_mask = full_mask[None, None, :, :].expand(
        batch_size, num_attention_heads, target_length, source_length
    ).contiguous()
    
    # Create sliding window causal mask
    swa_mask = torch.zeros(
        (target_length, source_length),
        dtype=dtype,
        device=device,
    )
    
    # Sliding window condition: within window size
    window_cond = (source_indices - past_key_values_length) >= (target_indices - sliding_window)
    
    # Combine conditions: must be both causal and within window
    valid_positions = causal_cond & window_cond
    swa_mask = swa_mask.masked_fill(valid_positions, False)
    
    # Expand to [batch_size, num_heads, seq_length, source_length]
    sliding_window_attention_mask = swa_mask[None, None, :, :].expand(
        batch_size, swa_num_attention_heads, target_length, source_length
    ).contiguous()
    
    return full_attention_mask, sliding_window_attention_mask
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000400 ms |
| - | Scoring Baseline | 0.500000 | 0.022456 ms |
| - | Reference Implementation | 0.059210 | 1.105142 ms |
