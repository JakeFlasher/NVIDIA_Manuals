## Description

Backward pass for tanh-gated residual addition. Computes gradients for residual, hidden_states, and gate parameter. Forward: output = residual + tanh(gate) * hidden_states * mask. Backward computes: grad_residual = grad_output, grad_hidden_states = grad_output * tanh(gate) * mask, grad_gate = sum(grad_output * hidden_states * mask * sech^2(gate)).
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| gate | [] | bfloat16 |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| mask | [batch_size, seq_len, 1] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_gate | [] | bfloat16 |

| # | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 1 | 131 | 0.1173 | 0.0012 |
| 2 | 4096 | 1 | 512 | 0.1205 | 0.0037 |
| 3 | 4096 | 64 | 128 | 0.2662 | 0.0529 |
| 4 | 4096 | 8 | 512 | 0.1585 | 0.0267 |
| 5 | 4096 | 32 | 128 | 0.1598 | 0.0267 |
| 6 | 4096 | 4 | 1024 | 0.1583 | 0.0267 |
| 7 | 4096 | 4 | 128 | 0.1259 | 0.0037 |
| 8 | 4096 | 2 | 512 | 0.1271 | 0.0070 |
| 9 | 4096 | 1 | 256 | 0.1218 | 0.0020 |
| 10 | 4096 | 4 | 256 | 0.1246 | 0.0070 |
| 11 | 4096 | 1 | 997 | 0.1206 | 0.0068 |
| 12 | 4096 | 2 | 128 | 0.1244 | 0.0020 |
| 13 | 4096 | 2 | 256 | 0.1256 | 0.0037 |
| 14 | 4096 | 8 | 691 | 0.1947 | 0.0358 |
| 15 | 4096 | 16 | 256 | 0.1579 | 0.0267 |
| 16 | 4096 | 2 | 1571 | 0.1480 | 0.0205 |

```python
import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Backward pass for tanh-gated residual addition.
    
    Forward was: output = residual + tanh(gate) * hidden_states * mask
    
    Gradients:
    - grad_residual = grad_output (identity)
    - grad_hidden_states = grad_output * tanh(gate) * mask
    - grad_gate = sum(grad_output * hidden_states * mask * sech^2(gate))
              = sum(grad_output * hidden_states * mask * (1 - tanh^2(gate)))
    """
    # Compute tanh(gate) for gradient computation
    gate_float = gate.to(torch.float32)
    gate_value = torch.tanh(gate_float)
    
    # Gradient w.r.t. residual: dy/d(residual) = 1
    grad_residual = grad_output.clone()
    
    # Gradient w.r.t. hidden_states: dy/d(hidden_states) = tanh(gate) * mask
    grad_hidden_states = grad_output * gate_value * mask
    
    # Gradient w.r.t. gate: dy/d(gate) = sum(hidden_states * mask * sech^2(gate) * grad_output)
    # Using identity: sech^2(x) = 1 - tanh^2(x)
    sech_squared = 1.0 - gate_value * gate_value
    
    # Apply mask to hidden_states
    masked_hidden_states = hidden_states * mask
    
    # Compute gradient: sum over all elements
    grad_gate = torch.sum(grad_output.to(torch.float32) * masked_hidden_states.to(torch.float32)) * sech_squared
    
    return grad_residual.to(torch.bfloat16), grad_hidden_states.to(torch.bfloat16), grad_gate.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.008917 ms |
| - | Scoring Baseline | 0.500000 | 0.143247 ms |
| - | Reference Implementation | 0.386509 | 0.231703 ms |
