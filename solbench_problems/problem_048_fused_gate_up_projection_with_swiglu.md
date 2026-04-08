## Description

Gemma3's fused gate and up projection followed by GELU-tanh activation and element-wise multiplication. Computes gelu_tanh(x @ gate_proj.T) * (x @ up_proj.T) where gate_proj and up_proj are weight matrices. This fuses two parallel linear projections from hidden_size=3072 to intermediate_size=24576, followed by GELU activation on gate output and element-wise multiplication with up output.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, hidden_size] | bfloat16 |
| gate_proj | [intermediate_size, hidden_size] | bfloat16 |
| up_proj | [intermediate_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, intermediate_size] | bfloat16 |

| # | hidden_size | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 24576 | 2 | 1321 | 0.6924 | 0.4409 |
| 2 | 3072 | 24576 | 2 | 128 | 0.1475 | 0.0431 |
| 3 | 3072 | 24576 | 64 | 128 | 1.8951 | 1.3663 |
| 4 | 3072 | 24576 | 1 | 8192 | 1.9777 | 1.3663 |
| 5 | 3072 | 24576 | 4 | 256 | 0.3401 | 0.1711 |
| 6 | 3072 | 24576 | 4 | 449 | 0.4979 | 0.2998 |
| 7 | 3072 | 24576 | 1 | 131 | 0.1420 | 0.0406 |
| 8 | 3072 | 24576 | 2 | 1024 | 0.5363 | 0.3419 |
| 9 | 3072 | 24576 | 16 | 512 | 1.9499 | 1.3663 |
| 10 | 3072 | 24576 | 1 | 256 | 0.1488 | 0.0431 |
| 11 | 3072 | 24576 | 4 | 128 | 0.2125 | 0.0858 |
| 12 | 3072 | 24576 | 8 | 128 | 0.3314 | 0.1711 |
| 13 | 3072 | 24576 | 4 | 512 | 0.5352 | 0.3419 |
| 14 | 3072 | 24576 | 1 | 128 | 0.1258 | 0.0406 |
| 15 | 3072 | 24576 | 2 | 2048 | 0.9712 | 0.6833 |
| 16 | 3072 | 24576 | 4 | 2048 | 1.9319 | 1.3663 |

```python
import torch
import math

@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    """
    Fused gate and up projection with GELU-tanh activation.
    
    Computes: gelu_tanh(x @ gate_proj.T) * (x @ up_proj.T)
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        gate_proj: Gate projection weights of shape (intermediate_size, hidden_size)
        up_proj: Up projection weights of shape (intermediate_size, hidden_size)
    
    Returns:
        Output tensor of shape (batch_size, seq_len, intermediate_size)
    """
    # Compute gate projection: x @ gate_proj.T
    # x: (batch_size, seq_len, hidden_size)
    # gate_proj: (intermediate_size, hidden_size)
    # gate_output: (batch_size, seq_len, intermediate_size)
    gate_output = torch.matmul(x, gate_proj.t())
    
    # Compute up projection: x @ up_proj.T
    # up_output: (batch_size, seq_len, intermediate_size)
    up_output = torch.matmul(x, up_proj.t())
    
    # Apply GELU with tanh approximation to gate output
    # GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    gate_float = gate_output.to(torch.float32)
    inner = sqrt_2_over_pi * (gate_float + 0.044715 * gate_float.pow(3))
    activated_gate = 0.5 * gate_float * (1.0 + torch.tanh(inner))
    activated_gate = activated_gate.to(gate_output.dtype)
    
    # Element-wise multiplication: activated_gate * up_output
    output = activated_gate * up_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.253067 ms |
| - | Scoring Baseline | 0.500000 | 0.494416 ms |
| - | Reference Implementation | 0.234954 | 1.003246 ms |
