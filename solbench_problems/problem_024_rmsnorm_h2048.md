## Description

Root Mean Square Normalization with hidden_size=2048. Captured from Qwen3-30B-A3B. Epsilon is fixed at 1e-6.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 2048 | 6 | 0.0161 | 0.0004 |
| 2 | 2048 | 1 | 0.0156 | 0.0004 |
| 3 | 2048 | 34 | 0.0154 | 0.0004 |
| 4 | 2048 | 79 | 0.0154 | 0.0005 |
| 5 | 2048 | 16254 | 0.0477 | 0.0178 |
| 6 | 2048 | 12383 | 0.0391 | 0.0136 |
| 7 | 2048 | 64 | 0.0880 | 0.0005 |

```python
import torch

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 2048

    EPS = 1e-6

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001213 ms |
| 1st place | Stellar Flamingo | 0.671991 | 0.013841 ms |
| - | Scoring Baseline | 0.500000 | 0.026715 ms |
| - | Reference Implementation | 0.215132 | 0.107082 ms |
