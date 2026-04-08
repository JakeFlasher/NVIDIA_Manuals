## Description

Post-normalization residual connection pattern: output = residual + RMSNorm(sublayer_output). RMSNorm applies learned weight scaling after normalization. Used after both attention and MLP blocks in Olmo3 architecture.
| Name | Shape | Dtype |
| --- | --- | --- |
| sublayer_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| weight | [hidden_size] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 16 | 1024 | 0.1341 | 0.0354 |
| 2 | 4096 | 8 | 2048 | 0.1337 | 0.0354 |
| 3 | 4096 | 32 | 256 | 0.0782 | 0.0179 |
| 4 | 4096 | 8 | 997 | 0.0762 | 0.0174 |
| 5 | 4096 | 16 | 512 | 0.0781 | 0.0179 |
| 6 | 4096 | 4 | 2048 | 0.0782 | 0.0179 |
| 7 | 4096 | 1 | 131 | 0.0199 | 0.0007 |
| 8 | 4096 | 2 | 2053 | 0.0452 | 0.0092 |
| 9 | 4096 | 2 | 4096 | 0.0787 | 0.0179 |
| 10 | 4096 | 8 | 512 | 0.0454 | 0.0092 |
| 11 | 4096 | 4 | 128 | 0.0213 | 0.0015 |
| 12 | 4096 | 1 | 1024 | 0.0234 | 0.0026 |
| 13 | 4096 | 2 | 293 | 0.0220 | 0.0017 |
| 14 | 4096 | 2 | 2048 | 0.0459 | 0.0092 |
| 15 | 4096 | 8 | 256 | 0.0308 | 0.0048 |
| 16 | 4096 | 1 | 128 | 0.0197 | 0.0007 |

```python
import torch

@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Post-normalization residual connection: output = residual + RMSNorm(sublayer_output)
    
    RMSNorm computation:
    1. Compute variance: mean of squared values along hidden dimension
    2. Normalize: x * rsqrt(variance + eps)
    3. Apply learned scale (weight parameter)
    4. Add residual connection
    
    Args:
        sublayer_output: Output from attention or MLP sublayer [batch, seq_len, hidden_size]
        residual: Residual connection input [batch, seq_len, hidden_size]
        weight: Learned scale parameter [hidden_size]
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor with residual added [batch, seq_len, hidden_size]
    """
    # Store input dtype for final conversion
    input_dtype = sublayer_output.dtype
    
    # RMSNorm computation in float32 for numerical stability
    normalized = sublayer_output.to(torch.float32)
    
    # Compute variance: mean of squared values along hidden dimension
    # Shape: [batch, seq_len, 1]
    variance = normalized.pow(2).mean(-1, keepdim=True)
    
    # Normalize: x * rsqrt(variance + eps)
    # rsqrt is more efficient than 1/sqrt
    normalized = normalized * torch.rsqrt(variance + eps)
    
    # Apply learned scale (weight parameter)
    normalized = weight.to(torch.float32) * normalized
    
    # Convert back to input dtype
    normalized = normalized.to(input_dtype)
    
    # Add residual connection
    output = residual + normalized
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.006913 ms |
| - | Scoring Baseline | 0.500000 | 0.047355 ms |
| - | Reference Implementation | 0.135662 | 0.264334 ms |
