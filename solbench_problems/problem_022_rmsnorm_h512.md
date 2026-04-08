## Description

Root Mean Square Normalization with hidden_size=512. Captured from DeepSeek-V3/R1. Epsilon is fixed at 1e-6.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 512 | 7 | 0.0241 | 0.0004 |
| 2 | 512 | 1 | 0.0239 | 0.0004 |
| 3 | 512 | 32 | 0.0235 | 0.0004 |
| 4 | 512 | 18 | 0.0237 | 0.0004 |
| 5 | 512 | 539 | 0.0236 | 0.0005 |
| 6 | 512 | 14521 | 0.0490 | 0.0043 |
| 7 | 512 | 11949 | 0.0435 | 0.0036 |
| 8 | 512 | 64 | 0.0279 | 0.0004 |

```python
import torch

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 512

    EPS = 1e-6

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000728 ms |
| - | Scoring Baseline | 0.500000 | 0.028614 ms |
| - | Reference Implementation | 0.229400 | 0.127893 ms |
