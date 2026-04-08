## Description

Backward pass for batched 2D RoPE position encoding. Computes gradients through cos/sin operations for mixed text-image sequences. Since positions and theta are non-differentiable, no gradients propagate further.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_cos | [batch_size, seq_len, head_dim] | bfloat16 |
| grad_sin | [batch_size, seq_len, head_dim] | bfloat16 |
| idx_theta | [batch_size, seq_len, head_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_idx_theta | [batch_size, seq_len, head_dim] | float32 |

| # | head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 2 | 773 | 0.0314 | 0.0007 |
| 2 | 128 | 1 | 1801 | 0.0315 | 0.0007 |
| 3 | 128 | 8 | 1024 | 0.0374 | 0.0018 |
| 4 | 128 | 64 | 613 | 0.0634 | 0.0069 |
| 5 | 128 | 1 | 1249 | 0.0301 | 0.0006 |
| 6 | 128 | 32 | 541 | 0.0489 | 0.0033 |
| 7 | 128 | 2 | 1321 | 0.0313 | 0.0008 |
| 8 | 128 | 32 | 1657 | 0.0671 | 0.0093 |
| 9 | 128 | 16 | 449 | 0.0381 | 0.0016 |
| 10 | 128 | 1 | 691 | 0.0303 | 0.0005 |
| 11 | 128 | 2 | 256 | 0.0541 | 0.0005 |
| 12 | 128 | 4 | 853 | 0.0533 | 0.0010 |
| 13 | 128 | 8 | 1489 | 0.0562 | 0.0024 |
| 14 | 128 | 2 | 1879 | 0.0534 | 0.0010 |
| 15 | 128 | 8 | 373 | 0.0557 | 0.0009 |
| 16 | 128 | 64 | 8192 | 0.2916 | 0.0879 |

```python
import torch

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for batched 2D RoPE position encoding.
    
    Computes gradient w.r.t. idx_theta using chain rule:
    - d(cos(x))/dx = -sin(x)
    - d(sin(x))/dx = cos(x)
    
    Therefore:
    grad_idx_theta = -grad_cos * sin(idx_theta) + grad_sin * cos(idx_theta)
    
    Note: In practice, this gradient doesn't propagate further since:
    - idx_theta = position * theta
    - position: discrete indices (non-differentiable)
    - theta: derived from configuration constants (non-differentiable)
    
    Args:
        grad_cos: Gradient w.r.t. cos output [batch_size, seq_len, head_dim]
        grad_sin: Gradient w.r.t. sin output [batch_size, seq_len, head_dim]
        idx_theta: Saved angles from forward pass [batch_size, seq_len, head_dim]
        
    Returns:
        grad_idx_theta: Gradient w.r.t. idx_theta [batch_size, seq_len, head_dim]
    """
    # Compute sin and cos of saved angles
    sin_theta = torch.sin(idx_theta)  # [batch_size, seq_len, head_dim]
    cos_theta = torch.cos(idx_theta)  # [batch_size, seq_len, head_dim]
    
    # Convert gradients to float32 for computation
    grad_cos_f32 = grad_cos.to(torch.float32)
    grad_sin_f32 = grad_sin.to(torch.float32)
    
    # Apply chain rule:
    # d(loss)/d(idx_theta) = d(loss)/d(cos) * d(cos)/d(idx_theta) + d(loss)/d(sin) * d(sin)/d(idx_theta)
    # d(cos)/d(idx_theta) = -sin(idx_theta)
    # d(sin)/d(idx_theta) = cos(idx_theta)
    grad_idx_theta = -grad_cos_f32 * sin_theta + grad_sin_f32 * cos_theta
    
    return grad_idx_theta
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001736 ms |
| - | Scoring Baseline | 0.500000 | 0.049184 ms |
| - | Reference Implementation | 0.333990 | 0.097761 ms |
