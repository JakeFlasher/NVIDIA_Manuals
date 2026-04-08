## Description

Root Mean Square Normalization with hidden_size=4096. Captured from Llama-3.1-8B. Epsilon is fixed at 1e-5.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- |
| 1 | 4096 | 7 | 0.0160 | 0.0004 |
| 2 | 4096 | 1 | 0.0157 | 0.0004 |
| 3 | 4096 | 34 | 0.0155 | 0.0005 |
| 4 | 4096 | 170 | 0.0161 | 0.0008 |
| 5 | 4096 | 14418 | 0.0640 | 0.0312 |
| 6 | 4096 | 11832 | 0.0551 | 0.0257 |
| 7 | 4096 | 64 | 0.0154 | 0.0005 |
| 8 | 4096 | 16 | 0.0157 | 0.0004 |
| 9 | 4096 | 10827 | 0.0520 | 0.0235 |
| 10 | 4096 | 8804 | 0.0444 | 0.0192 |
| 11 | 4096 | 63 | 0.0157 | 0.0005 |
| 12 | 4096 | 79 | 0.0157 | 0.0006 |
| 13 | 4096 | 14509 | 0.0641 | 0.0314 |
| 14 | 4096 | 15 | 0.0154 | 0.0004 |

```python
import torch

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 4096

    EPS = 1e-5

    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002009 ms |
| 1st place | Stellar Flamingo | 0.648557 | 0.015051 ms |
| 2nd place | AKO4ALL_L2 | 0.593301 | 0.018001 ms |
| - | Scoring Baseline | 0.500000 | 0.024625 ms |
| 3rd place | Jasper Otter | 0.467955 | 0.029355 ms |
| - | Reference Implementation | 0.137815 | 0.150646 ms |
