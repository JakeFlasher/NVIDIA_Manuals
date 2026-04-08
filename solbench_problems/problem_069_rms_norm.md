## Description

Fused residual addition followed by RMSNorm. Computes hidden_states = residual + hidden_states, then applies RMSNorm: output = weight * (x / sqrt(mean(x^2) + eps)). This pattern occurs twice per decoder layer in transformer models.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 1 | 8192 | 0.1288 | 0.0529 |
| 2 | 8192 | 1 | 131 | 0.0249 | 0.0012 |
| 3 | 8192 | 16 | 256 | 0.0782 | 0.0267 |
| 4 | 8192 | 4 | 128 | 0.0247 | 0.0037 |
| 5 | 8192 | 4 | 541 | 0.0455 | 0.0143 |
| 6 | 8192 | 1 | 2048 | 0.0418 | 0.0135 |
| 7 | 8192 | 2 | 293 | 0.0248 | 0.0042 |
| 8 | 8192 | 4 | 2048 | 0.1038 | 0.0529 |
| 9 | 8192 | 8 | 128 | 0.0283 | 0.0070 |
| 10 | 8192 | 8 | 512 | 0.0745 | 0.0267 |
| 11 | 8192 | 4 | 256 | 0.0288 | 0.0070 |
| 12 | 8192 | 2 | 2048 | 0.0747 | 0.0267 |
| 13 | 8192 | 1 | 4096 | 0.0739 | 0.0267 |
| 14 | 8192 | 1 | 256 | 0.0240 | 0.0020 |
| 15 | 8192 | 2 | 1024 | 0.0425 | 0.0135 |
| 16 | 8192 | 1 | 1024 | 0.0285 | 0.0070 |

```python
import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Residual addition
    x = residual + hidden_states
    
    # RMSNorm computation in float32 for numerical stability
    x_fp32 = x.to(torch.float32)
    
    # Compute variance (mean of squares) along last dimension
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    
    # Normalize by RMS (root mean square)
    x_normalized = x_fp32 * torch.rsqrt(variance + eps)
    
    # Apply learned scaling and convert back to bfloat16
    output = weight * x_normalized.to(torch.bfloat16)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.011022 ms |
| 1st place | Mystic Orca | 0.713265 | 0.025635 ms |
| 2nd place | AKO4ALL_L1 | 0.675320 | 0.028318 ms |
| - | Scoring Baseline | 0.500000 | 0.045130 ms |
| - | Reference Implementation | 0.138939 | 0.227287 ms |
| 3rd place | Rapid Lynx | 0.114354 | 0.285881 ms |
