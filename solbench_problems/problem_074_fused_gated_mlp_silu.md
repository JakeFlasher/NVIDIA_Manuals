## Description

Gated MLP with SiLU activation that fuses gate and up projections into a single linear layer, then splits and applies gated activation before down projection. The pattern is: gate_up_proj -> chunk -> SiLU(gate) * up -> down_proj.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| gate_up_weight | [intermediate_size_x2, hidden_size] | float32 |
| down_weight | [hidden_size, intermediate_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | float32 |

| # | hidden_size | intermediate_size | intermediate_size_x2 | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 4096 | 8192 | 2 | 2048 | 0.2569 | 0.0573 |
| 2 | 1024 | 4096 | 8192 | 32 | 256 | 0.4225 | 0.1142 |
| 3 | 1024 | 4096 | 8192 | 8 | 256 | 0.1710 | 0.0289 |
| 4 | 1024 | 4096 | 8192 | 8 | 512 | 0.2511 | 0.0573 |
| 5 | 1024 | 4096 | 8192 | 8 | 613 | 0.3095 | 0.0685 |
| 6 | 1024 | 4096 | 8192 | 4 | 1024 | 0.2503 | 0.0573 |
| 7 | 1024 | 4096 | 8192 | 4 | 449 | 0.1656 | 0.0254 |
| 8 | 1024 | 4096 | 8192 | 16 | 773 | 0.5874 | 0.1722 |
| 9 | 1024 | 4096 | 8192 | 4 | 512 | 0.1699 | 0.0289 |
| 10 | 1024 | 4096 | 8192 | 64 | 128 | 0.4186 | 0.1142 |
| 11 | 1024 | 4096 | 8192 | 32 | 128 | 0.2498 | 0.0573 |
| 12 | 1024 | 4096 | 8192 | 2 | 128 | 0.0715 | 0.0040 |
| 13 | 1024 | 4096 | 8192 | 2 | 1249 | 0.2063 | 0.0351 |
| 14 | 1024 | 4096 | 8192 | 16 | 512 | 0.4191 | 0.1142 |
| 15 | 1024 | 4096 | 8192 | 2 | 512 | 0.1253 | 0.0146 |
| 16 | 1024 | 4096 | 8192 | 4 | 256 | 0.1246 | 0.0146 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    # hidden_states: (batch_size, seq_len, hidden_size)
    # gate_up_weight: (2 * intermediate_size, hidden_size)
    # down_weight: (hidden_size, intermediate_size)
    
    # Fused gate and up projection: (B, S, H) @ (H, 2*I) -> (B, S, 2*I)
    up_states = F.linear(hidden_states, gate_up_weight)
    
    # Split into gate and up components along last dimension
    # Each has shape (B, S, I)
    gate, up_states = up_states.chunk(2, dim=-1)
    
    # Apply SiLU gating: SiLU(x) = x * sigmoid(x)
    # Then multiply: up_states = up_states * silu(gate)
    silu_gate = gate * torch.sigmoid(gate)
    up_states = up_states * silu_gate
    
    # Down projection: (B, S, I) @ (I, H) -> (B, S, H)
    output = F.linear(up_states, down_weight)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.042658 ms |
| - | Scoring Baseline | 0.500000 | 0.229721 ms |
| - | Reference Implementation | 0.405927 | 0.314419 ms |
