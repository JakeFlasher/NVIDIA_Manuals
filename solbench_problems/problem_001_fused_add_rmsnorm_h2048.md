## Description

Fused Add + RMSNorm with hidden_size=2048 for Qwen3-30B-A3B. Epsilon is fixed at 1e-6.
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
| 1 | 2048 | 6 | 0.0612 | 0.0004 |
| 2 | 2048 | 1 | 0.0630 | 0.0004 |
| 3 | 2048 | 34 | 0.0610 | 0.0005 |
| 4 | 2048 | 79 | 0.0596 | 0.0005 |
| 5 | 2048 | 16254 | 0.0728 | 0.0264 |
| 6 | 2048 | 12383 | 0.0651 | 0.0202 |
| 7 | 2048 | 64 | 0.0584 | 0.0005 |

```python
import torch

@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 2048

    EPS = 1e-6

    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001402 ms |
| 1st place | Stellar Flamingo | 0.845603 | 0.013211 ms |
| - | Scoring Baseline | 0.500000 | 0.062862 ms |
| - | Reference Implementation | 0.328276 | 0.140447 ms |
