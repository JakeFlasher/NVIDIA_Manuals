## Description

FLUX feedforward network with GELU approximate activation (tanh approximation). Performs Linear projection to 4x hidden dimension, GELU-tanh activation, and Linear projection back to hidden dimension. This appears 76 times per forward pass in FLUX transformer blocks.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_dim] | float32 |
| fc1_weight | [mlp_hidden_dim, hidden_dim] | float32 |
| fc1_bias | [mlp_hidden_dim] | float32 |
| fc2_weight | [hidden_dim, mlp_hidden_dim] | float32 |
| fc2_bias | [hidden_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_dim] | float32 |

| # | hidden_dim | mlp_hidden_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 12288 | 32 | 256 | 1.8068 | 0.6833 |
| 2 | 3072 | 12288 | 1 | 1024 | 0.3376 | 0.0858 |
| 3 | 3072 | 12288 | 4 | 2048 | 1.7056 | 0.6833 |
| 4 | 3072 | 12288 | 1 | 2048 | 0.5880 | 0.1711 |
| 5 | 3072 | 12288 | 4 | 512 | 0.5167 | 0.1711 |
| 6 | 3072 | 12288 | 8 | 1024 | 1.9373 | 0.6833 |
| 7 | 3072 | 12288 | 32 | 512 | 3.9561 | 1.3663 |
| 8 | 3072 | 12288 | 8 | 512 | 1.0099 | 0.3419 |
| 9 | 3072 | 12288 | 4 | 541 | 0.5435 | 0.1808 |
| 10 | 3072 | 12288 | 16 | 1024 | 4.0163 | 1.3663 |
| 11 | 3072 | 12288 | 4 | 256 | 0.3742 | 0.0858 |
| 12 | 3072 | 12288 | 16 | 512 | 1.7603 | 0.6833 |
| 13 | 3072 | 12288 | 1 | 131 | 0.1415 | 0.0113 |
| 14 | 3072 | 12288 | 1 | 512 | 0.1880 | 0.0431 |
| 15 | 3072 | 12288 | 1 | 4096 | 0.8952 | 0.3419 |
| 16 | 3072 | 12288 | 4 | 3089 | 2.8702 | 1.0305 |

```python
import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    """
    FLUX FeedForward with GELU approximate activation.
    
    Architecture:
    - Linear: hidden_dim (3072) -> mlp_hidden_dim (12288)
    - GELU activation with tanh approximation
    - Linear: mlp_hidden_dim (12288) -> hidden_dim (3072)
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    # Step 1: First linear projection [batch, seq, 3072] -> [batch, seq, 12288]
    x = F.linear(hidden_states, fc1_weight, fc1_bias)
    
    # Step 2: GELU activation with tanh approximation
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = F.gelu(x, approximate="tanh")
    
    # Step 3: Second linear projection [batch, seq, 12288] -> [batch, seq, 3072]
    output = F.linear(x, fc2_weight, fc2_bias)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.272608 ms |
| - | Scoring Baseline | 0.500000 | 0.914765 ms |
| - | Reference Implementation | 0.491546 | 0.933859 ms |
