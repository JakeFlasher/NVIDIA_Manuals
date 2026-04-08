## Description

Backward pass for Adaptive Instance Normalization 2D. Computes gradients with respect to input, weight (gamma), and bias (beta). The normalization is performed per-channel across spatial dimensions: y = gamma * (x - mean) / sqrt(var + eps) + beta.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [N, C, H, W] | float32 |
| x | [N, C, H, W] | float32 |
| weight | [C] | float32 |
| mean | [N, C, 1, 1] | float32 |
| std | [N, C, 1, 1] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_input | [N, C, H, W] | float32 |
| grad_weight | [C] | float32 |
| grad_bias | [C] | float32 |

| # | C | N | H | W | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 480 | 640 | 1.9581 | 0.0619 |
| 2 | 32 | 2 | 128 | 128 | 0.1407 | 0.0012 |
| 3 | 32 | 1 | 1163 | 1249 | 1.0337 | 0.0368 |
| 4 | 32 | 2 | 691 | 773 | 0.9187 | 0.0272 |
| 5 | 32 | 32 | 120 | 160 | 0.5592 | 0.0158 |
| 6 | 32 | 64 | 60 | 80 | 0.3729 | 0.0081 |
| 7 | 32 | 32 | 64 | 128 | 0.2889 | 0.0070 |
| 8 | 32 | 16 | 240 | 320 | 1.0007 | 0.0312 |
| 9 | 32 | 4 | 720 | 1280 | 2.7214 | 0.0927 |
| 10 | 32 | 8 | 373 | 373 | 0.9364 | 0.0283 |
| 11 | 32 | 2 | 211 | 211 | 0.2406 | 0.0026 |
| 12 | 32 | 8 | 256 | 512 | 0.8773 | 0.0267 |
| 13 | 32 | 4 | 256 | 256 | 0.3191 | 0.0070 |
| 14 | 32 | 1 | 1080 | 1920 | 1.3973 | 0.0523 |
| 15 | 32 | 1 | 2048 | 2048 | 2.4001 | 0.1054 |
| 16 | 32 | 2 | 1321 | 1423 | 2.7378 | 0.0945 |

```python
import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    """
    Backward pass for Adaptive Instance Normalization 2D.
    
    Computes gradients with respect to input, weight (gamma), and bias (beta).
    
    Args:
        grad_output: Gradient of loss w.r.t. output, shape (N, C, H, W)
        x: Original input tensor from forward pass, shape (N, C, H, W)
        weight: Scale parameter (gamma), shape (C,)
        mean: Mean computed in forward pass, shape (N, C, 1, 1)
        std: Standard deviation computed in forward pass, shape (N, C, 1, 1)
        
    Returns:
        grad_input: Gradient w.r.t. input, shape (N, C, H, W)
        grad_weight: Gradient w.r.t. weight (gamma), shape (C,)
        grad_bias: Gradient w.r.t. bias (beta), shape (C,)
    """
    N, C, H, W = x.shape
    spatial_size = H * W
    
    # Compute centered and normalized input
    x_centered = x - mean  # Shape: (N, C, H, W)
    x_normalized = x_centered / std  # Shape: (N, C, H, W)
    
    # Gradient w.r.t. bias (beta)
    # d(loss)/d(beta) = sum(d(loss)/d(output)) over N, H, W
    grad_bias = grad_output.sum(dim=(0, 2, 3))  # Shape: (C,)
    
    # Gradient w.r.t. weight (gamma)
    # d(loss)/d(gamma) = sum(d(loss)/d(output) * x_normalized) over N, H, W
    grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))  # Shape: (C,)
    
    # Gradient w.r.t. input
    # Scale grad_output by weight
    weight_reshaped = weight.view(1, C, 1, 1)
    grad_output_scaled = grad_output * weight_reshaped
    
    # Gradient w.r.t. variance
    # d(loss)/d(var) = sum(d(loss)/d(x_norm) * (x - mean) * -0.5 * (var + eps)^(-3/2))
    grad_var = (grad_output_scaled * x_centered).sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)
    
    # Gradient w.r.t. mean
    # d(loss)/d(mean) = sum(d(loss)/d(x_norm) * -1/std) + d(loss)/d(var) * sum(-2 * (x - mean)) / (H*W)
    grad_mean = (grad_output_scaled / (-std)).sum(dim=(2, 3), keepdim=True)
    grad_mean = grad_mean + grad_var * (-2.0 * x_centered).sum(dim=(2, 3), keepdim=True) / spatial_size
    
    # Gradient w.r.t. input
    # d(loss)/d(x) = d(loss)/d(x_norm) * 1/std + d(loss)/d(var) * 2*(x-mean)/(H*W) + d(loss)/d(mean) * 1/(H*W)
    grad_input = grad_output_scaled / std
    grad_input = grad_input + grad_var * 2.0 * x_centered / spatial_size
    grad_input = grad_input + grad_mean / spatial_size
    
    return grad_input, grad_weight, grad_bias
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.020953 ms |
| - | Scoring Baseline | 0.500000 | 0.793798 ms |
| - | Reference Implementation | 0.389042 | 1.234870 ms |
