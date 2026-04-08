## Description

Fused cos/sin embedding generation from frequencies for RoPE. Takes frequency tensor (result of inv_freq @ position_ids) and generates cos/sin embeddings with attention scaling. Concatenates frequencies with themselves, computes cos and sin with scaling in a single pass.
| Name | Shape | Dtype |
| --- | --- | --- |
| freqs | [batch_size, seq_len, half_head_dim] | float32 |
| attention_scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |

| # | head_dim | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 64 | 4 | 512 | 0.0311 | 0.0006 |
| 2 | 128 | 64 | 2 | 691 | 0.0302 | 0.0005 |
| 3 | 128 | 64 | 1 | 4096 | 0.0293 | 0.0008 |
| 4 | 128 | 64 | 16 | 373 | 0.0300 | 0.0010 |
| 5 | 128 | 64 | 64 | 541 | 0.0505 | 0.0039 |
| 6 | 128 | 64 | 8 | 512 | 0.0294 | 0.0008 |
| 7 | 128 | 64 | 8 | 256 | 0.0315 | 0.0006 |
| 8 | 128 | 64 | 2 | 512 | 0.0295 | 0.0005 |
| 9 | 128 | 64 | 1 | 512 | 0.0290 | 0.0005 |
| 10 | 128 | 64 | 16 | 128 | 0.0289 | 0.0006 |
| 11 | 128 | 64 | 4 | 1024 | 0.0294 | 0.0008 |
| 12 | 128 | 64 | 16 | 512 | 0.0315 | 0.0012 |
| 13 | 128 | 64 | 1 | 2048 | 0.0288 | 0.0006 |
| 14 | 128 | 64 | 8 | 853 | 0.0318 | 0.0011 |
| 15 | 128 | 64 | 8 | 128 | 0.0289 | 0.0005 |
| 16 | 128 | 64 | 1 | 1024 | 0.0288 | 0.0005 |

```python
import torch

@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    """
    Fused cos/sin embedding generation from frequencies for RoPE.
    
    Args:
        freqs: Frequency tensor of shape [batch_size, seq_len, head_dim // 2]
               Result of (inv_freq @ position_ids).transpose(1, 2)
        attention_scaling: Scaling factor for attention
    
    Returns:
        Tuple of (cos, sin) embeddings, each of shape [batch_size, seq_len, head_dim]
    """
    # Concatenate frequencies with themselves along last dimension
    # This doubles the head_dim dimension: [batch, seq, head_dim//2] -> [batch, seq, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # Compute cos and sin with scaling
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    
    # Cast to target dtype (bfloat16)
    cos = cos.to(torch.bfloat16)
    sin = sin.to(torch.bfloat16)
    
    return cos, sin
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000753 ms |
| 1st place | Mystic Orca | 0.714653 | 0.012692 ms |
| 2nd place | Clever Scorpion | 0.541246 | 0.026272 ms |
| - | Scoring Baseline | 0.500000 | 0.030853 ms |
| - | Reference Implementation | 0.370801 | 0.052041 ms |
