## Description

Whisper decoder output projection layer that maps hidden states (d_model=1280) to vocabulary logits (vocab_size=51866). This is a critical bottleneck in autoregressive generation as it runs for every generated token. The operation is C = A @ B.T where A is hidden_states and B is the weight matrix.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, d_model] | float16 |
| weight | [vocab_size, d_model] | float16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| logits | [batch_size, seq_len, vocab_size] | float16 |

| # | d_model | vocab_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1280 | 51866 | 4 | 293 | 0.2189 | 0.0863 |
| 2 | 1280 | 51866 | 2 | 1024 | 0.2767 | 0.1505 |
| 3 | 1280 | 51866 | 1 | 2048 | 0.2760 | 0.1505 |
| 4 | 1280 | 51866 | 4 | 1 | 0.0748 | 0.0178 |
| 5 | 1280 | 51866 | 2 | 128 | 0.1298 | 0.0212 |
| 6 | 1280 | 51866 | 16 | 1 | 0.0761 | 0.0179 |
| 7 | 1280 | 51866 | 32 | 128 | 0.4508 | 0.3007 |
| 8 | 1280 | 51866 | 32 | 541 | 1.7000 | 1.2695 |
| 9 | 1280 | 51866 | 16 | 449 | 0.7325 | 0.5270 |
| 10 | 1280 | 51866 | 64 | 128 | 0.8020 | 0.6009 |
| 11 | 1280 | 51866 | 1 | 1 | 0.0614 | 0.0177 |
| 12 | 1280 | 51866 | 1 | 128 | 0.1284 | 0.0194 |
| 13 | 1280 | 51866 | 1 | 512 | 0.1615 | 0.0379 |
| 14 | 1280 | 51866 | 2 | 211 | 0.1634 | 0.0313 |
| 15 | 1280 | 51866 | 64 | 613 | 4.2114 | 2.8764 |
| 16 | 1280 | 51866 | 16 | 256 | 0.4468 | 0.3007 |

```python
import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Whisper decoder output projection: projects hidden states to vocabulary logits.
    
    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, d_model=1280)
        weight: Tensor of shape (vocab_size=51866, d_model=1280)
    
    Returns:
        logits: Tensor of shape (batch_size, seq_len, vocab_size=51866)
    """
    # Linear projection without bias: output = input @ weight.T
    # hidden_states: (batch_size, seq_len, d_model)
    # weight: (vocab_size, d_model)
    # logits: (batch_size, seq_len, vocab_size)
    logits = torch.matmul(hidden_states, weight.t())
    return logits
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.111398 ms |
| - | Scoring Baseline | 0.500000 | 0.287491 ms |
| - | Reference Implementation | 0.241636 | 0.745662 ms |
