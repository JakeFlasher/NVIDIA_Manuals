## Description

GELU activation function using tanh approximation applied element-wise to video latent tensors. Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). This is executed in transformer feedforward networks on intermediate tensors.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, intermediate_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, intermediate_size] | float32 |

| # | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 2 | 512 | 0.0151 | 0.0048 |
| 2 | 4096 | 8 | 997 | 0.0515 | 0.0345 |
| 3 | 4096 | 4 | 541 | 0.0200 | 0.0096 |
| 4 | 4096 | 4 | 128 | 0.0131 | 0.0026 |
| 5 | 4096 | 32 | 1024 | 0.1757 | 0.1404 |
| 6 | 4096 | 8 | 256 | 0.0193 | 0.0092 |
| 7 | 4096 | 8 | 4096 | 0.1716 | 0.1404 |
| 8 | 4096 | 16 | 512 | 0.0525 | 0.0354 |
| 9 | 4096 | 4 | 8192 | 0.1718 | 0.1404 |
| 10 | 4096 | 16 | 1571 | 0.1349 | 0.1078 |
| 11 | 4096 | 2 | 293 | 0.0135 | 0.0029 |
| 12 | 4096 | 1 | 256 | 0.0128 | 0.0015 |
| 13 | 4096 | 2 | 4096 | 0.0524 | 0.0354 |
| 14 | 4096 | 4 | 256 | 0.0146 | 0.0048 |
| 15 | 4096 | 2 | 256 | 0.0130 | 0.0026 |
| 16 | 4096 | 1 | 512 | 0.0131 | 0.0026 |

```python
import torch
import math

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation using tanh approximation.
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute inner term: sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Apply tanh and compute final result
    output = 0.5 * x * (1.0 + torch.tanh(inner))
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.013929 ms |
| - | Scoring Baseline | 0.500000 | 0.034058 ms |
| - | Reference Implementation | 0.084966 | 0.232099 ms |
