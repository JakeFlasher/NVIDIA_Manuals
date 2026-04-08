## Description

GEGLU (Gated GLU with GELU) activation function. Splits input along last dimension into two halves, applies approximate GELU (tanh-based) to the first half, and multiplies element-wise with the second half. Used in transformer feed-forward networks.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, input_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, inner_dim] | float32 |

| # | inner_dim | input_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 10240 | 1 | 2048 | 0.0633 | 0.0086 |
| 2 | 5120 | 10240 | 4 | 541 | 0.0597 | 0.0091 |
| 3 | 5120 | 10240 | 8 | 256 | 0.0580 | 0.0086 |
| 4 | 5120 | 10240 | 16 | 256 | 0.0774 | 0.0168 |
| 5 | 5120 | 10240 | 4 | 1024 | 0.0779 | 0.0168 |
| 6 | 5120 | 10240 | 1 | 128 | 0.0230 | 0.0009 |
| 7 | 5120 | 10240 | 8 | 1024 | 0.1153 | 0.0332 |
| 8 | 5120 | 10240 | 1 | 512 | 0.0320 | 0.0025 |
| 9 | 5120 | 10240 | 32 | 128 | 0.0776 | 0.0168 |
| 10 | 5120 | 10240 | 2 | 211 | 0.0292 | 0.0021 |
| 11 | 5120 | 10240 | 8 | 613 | 0.0840 | 0.0200 |
| 12 | 5120 | 10240 | 1 | 1024 | 0.0488 | 0.0045 |
| 13 | 5120 | 10240 | 1 | 131 | 0.0232 | 0.0009 |
| 14 | 5120 | 10240 | 16 | 449 | 0.1035 | 0.0292 |
| 15 | 5120 | 10240 | 4 | 128 | 0.0320 | 0.0025 |
| 16 | 5120 | 10240 | 2 | 512 | 0.0486 | 0.0045 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """
    GEGLU activation: GELU(x_gate) * x_linear
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, inner_dim * 2)
        
    Returns:
        Output tensor of shape (batch_size, seq_len, inner_dim)
    """
    # Split input into two halves along last dimension
    # x_gate and x_linear each have shape (batch_size, seq_len, inner_dim)
    x_gate, x_linear = x.chunk(2, dim=-1)
    
    # Apply approximate GELU (tanh-based) to gate and multiply with linear part
    # GELU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    output = F.gelu(x_gate, approximate='tanh') * x_linear
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.006611 ms |
| 1st place | gnehiz | 0.669614 | 0.029204 ms |
| - | Scoring Baseline | 0.500000 | 0.053019 ms |
| - | Reference Implementation | 0.398736 | 0.076300 ms |
