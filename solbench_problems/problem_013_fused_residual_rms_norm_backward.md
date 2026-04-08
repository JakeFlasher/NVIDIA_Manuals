## Description

Backward pass for fused residual addition and RMSNorm. Computes gradients for hidden_states, residual, and weight given the gradient of the output.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| x | [batch_size, seq_len, hidden_size] | float32 |
| normalized | [batch_size, seq_len, hidden_size] | float32 |
| rstd | [batch_size, seq_len, 1] | float32 |
| weight | [hidden_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_weight | [hidden_size] | float32 |

| # | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2560 | 4 | 1973 | 0.1668 | 0.0109 |
| 2 | 2560 | 64 | 8192 | 5.7280 | 0.7006 |
| 3 | 2560 | 2 | 773 | 0.0986 | 0.0025 |
| 4 | 2560 | 2 | 256 | 0.1067 | 0.0011 |
| 5 | 2560 | 1 | 1249 | 0.0916 | 0.0021 |
| 6 | 2560 | 4 | 853 | 0.1085 | 0.0050 |
| 7 | 2560 | 16 | 997 | 0.2653 | 0.0217 |
| 8 | 2560 | 2 | 211 | 0.0984 | 0.0010 |
| 9 | 2560 | 1 | 128 | 0.0806 | 0.0006 |
| 10 | 2560 | 32 | 1087 | 0.5013 | 0.0469 |
| 11 | 2560 | 32 | 1657 | 0.6892 | 0.0712 |
| 12 | 2560 | 64 | 1721 | 1.3006 | 0.1475 |
| 13 | 2560 | 8 | 1024 | 0.1733 | 0.0113 |
| 14 | 2560 | 32 | 4096 | 1.5067 | 0.1755 |
| 15 | 2560 | 1 | 1801 | 0.0922 | 0.0028 |
| 16 | 2560 | 1 | 131 | 0.0805 | 0.0006 |

```python
import torch

@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, normalized: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor):
    # Convert grad_output to float32 for computation
    grad_output_f32 = grad_output.to(torch.float32)
    
    # Gradient with respect to weight
    # dy/dweight = normalized (summed over batch and sequence dimensions)
    grad_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))
    
    # Gradient with respect to normalized output
    # dy/dnormalized = weight
    grad_normalized = grad_output_f32 * weight
    
    # Gradient with respect to x (before normalization)
    # For RMSNorm: norm = x * rstd where rstd = 1/sqrt(mean(x^2) + eps)
    # dnorm/dx = rstd * (I - norm * x^T / hidden_size)
    # Therefore: grad_x = rstd * (grad_normalized - mean(grad_normalized * normalized) * normalized)
    
    # Compute mean(grad_normalized * normalized) over hidden dimension
    mean_grad_norm = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
    
    # Gradient through normalization
    grad_x = rstd * (grad_normalized - mean_grad_norm * normalized)
    
    # Convert back to input dtype (bfloat16)
    grad_x_bf16 = grad_x.to(torch.bfloat16)
    
    # Both hidden_states and residual receive the same gradient
    grad_hidden_states = grad_x_bf16
    grad_residual = grad_x_bf16.clone()
    
    return grad_hidden_states, grad_residual, grad_weight
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.009541 ms |
| - | Scoring Baseline | 0.500000 | 0.246253 ms |
| - | Reference Implementation | 0.316223 | 0.549747 ms |
