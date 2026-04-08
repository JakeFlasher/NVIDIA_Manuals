## Description

Root Mean Square Normalization with hidden_size=128. Captured from Qwen3-30B-A3B. Epsilon is fixed at 1e-6.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 128 | 192 | 0.0140 | 0.0004 |
| 2 | 128 | 24 | 0.0138 | 0.0004 |
| 3 | 128 | 32 | 0.0137 | 0.0004 |
| 4 | 128 | 4 | 0.0136 | 0.0004 |
| 5 | 128 | 1088 | 0.0139 | 0.0005 |
| 6 | 128 | 136 | 0.0138 | 0.0004 |
| 7 | 128 | 2528 | 0.0141 | 0.0006 |
| 8 | 128 | 316 | 0.0137 | 0.0004 |
| 9 | 128 | 520128 | 0.0962 | 0.0351 |
| 10 | 128 | 65016 | 0.0276 | 0.0047 |
| 11 | 128 | 396256 | 0.0681 | 0.0269 |
| 12 | 128 | 49532 | 0.0273 | 0.0037 |
| 13 | 128 | 2048 | 0.0137 | 0.0005 |
| 14 | 128 | 256 | 0.0138 | 0.0004 |

```python
import torch

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 128

    EPS = 1e-6

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001105 ms |
| - | Scoring Baseline | 0.500000 | 0.019615 ms |
| - | Reference Implementation | 0.158273 | 0.100213 ms |
