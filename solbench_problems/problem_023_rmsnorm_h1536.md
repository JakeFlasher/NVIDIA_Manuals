## Description

Root Mean Square Normalization with hidden_size=1536. Captured from DeepSeek-V3/R1. Epsilon is fixed at 1e-6.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 1536 | 7 | 0.0213 | 0.0004 |
| 2 | 1536 | 1 | 0.0202 | 0.0004 |
| 3 | 1536 | 32 | 0.0200 | 0.0004 |
| 4 | 1536 | 18 | 0.0203 | 0.0004 |
| 5 | 1536 | 539 | 0.0217 | 0.0008 |
| 6 | 1536 | 14521 | 0.0672 | 0.0120 |
| 7 | 1536 | 11949 | 0.0579 | 0.0100 |
| 8 | 1536 | 64 | 0.0205 | 0.0005 |

```python
import torch

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 1536

    EPS = 1e-6

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001026 ms |
| - | Scoring Baseline | 0.500000 | 0.027219 ms |
| - | Reference Implementation | 0.228022 | 0.089269 ms |
