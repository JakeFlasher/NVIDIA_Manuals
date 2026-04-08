## Description

Fused Add + RMSNorm with hidden_size=7168 for DeepSeek-V3/R1. Epsilon is fixed at 1e-6.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, hidden_size] | bfloat16 |
| residual | [batch_size, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 7168 | 7 | 0.0658 | 0.0004 |
| 2 | 7168 | 1 | 0.0601 | 0.0004 |
| 3 | 7168 | 32 | 0.0639 | 0.0006 |
| 4 | 7168 | 18 | 0.0647 | 0.0005 |
| 5 | 7168 | 539 | 0.0641 | 0.0034 |
| 6 | 7168 | 14521 | 0.1723 | 0.0819 |
| 7 | 7168 | 11949 | 0.1413 | 0.0674 |
| 8 | 7168 | 64 | 0.0640 | 0.0008 |

```python
import torch

@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 7168

    EPS = 1e-6

    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002276 ms |
| - | Scoring Baseline | 0.500000 | 0.079721 ms |
| - | Reference Implementation | 0.282719 | 0.233816 ms |
