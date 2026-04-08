## Description

Fused Add + RMSNorm with hidden_size=4096 for Llama-3.1-8B. Epsilon is fixed at 1e-5.
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
| 1 | 4096 | 7 | 0.0323 | 0.0004 |
| 2 | 4096 | 1 | 0.0316 | 0.0004 |
| 3 | 4096 | 34 | 0.0322 | 0.0005 |
| 4 | 4096 | 170 | 0.0320 | 0.0009 |
| 5 | 4096 | 14418 | 0.1065 | 0.0466 |
| 6 | 4096 | 11832 | 0.0970 | 0.0383 |
| 7 | 4096 | 64 | 0.0321 | 0.0006 |
| 8 | 4096 | 16 | 0.0320 | 0.0005 |
| 9 | 4096 | 10827 | 0.0925 | 0.0351 |
| 10 | 4096 | 8804 | 0.0846 | 0.0286 |
| 11 | 4096 | 63 | 0.0318 | 0.0006 |
| 12 | 4096 | 79 | 0.0320 | 0.0007 |
| 13 | 4096 | 14509 | 0.1069 | 0.0469 |
| 14 | 4096 | 15 | 0.0317 | 0.0004 |

```python
import torch

@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 4096

    EPS = 1e-5

    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002464 ms |
| - | Scoring Baseline | 0.500000 | 0.047544 ms |
| - | Reference Implementation | 0.187346 | 0.209431 ms |
