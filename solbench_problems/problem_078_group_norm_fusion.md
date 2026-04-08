## Description

GroupNorm operation for Stable Diffusion UNet. Normalizes features across spatial dimensions within channel groups. Computes mean and variance per group, normalizes, and applies affine transform. With 32 groups and channels of 320/640/1280, this operation appears hundreds of times per forward pass.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, num_channels, height, width] | float32 |
| weight | [num_channels] | float32 |
| bias | [num_channels] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, num_channels, height, width] | float32 |

| # | num_groups | batch_size | num_channels | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 16 | 640 | 32 | 32 | 0.0697 | 0.0059 |
| 2 | 32 | 2 | 640 | 32 | 32 | 0.0451 | 0.0011 |
| 3 | 32 | 4 | 1280 | 16 | 16 | 0.0389 | 0.0011 |
| 4 | 32 | 4 | 320 | 64 | 64 | 0.1291 | 0.0031 |
| 5 | 32 | 4 | 1280 | 8 | 8 | 0.0301 | 0.0006 |
| 6 | 32 | 8 | 1280 | 16 | 16 | 0.0444 | 0.0018 |
| 7 | 32 | 2 | 320 | 128 | 128 | 0.0844 | 0.0059 |
| 8 | 32 | 8 | 320 | 64 | 64 | 0.0806 | 0.0059 |
| 9 | 32 | 4 | 640 | 32 | 32 | 0.0501 | 0.0018 |
| 10 | 32 | 2 | 640 | 211 | 53 | 0.1079 | 0.0079 |
| 11 | 32 | 32 | 1280 | 8 | 8 | 0.0475 | 0.0018 |
| 12 | 32 | 2 | 320 | 449 | 7 | 0.0458 | 0.0014 |
| 13 | 32 | 1 | 1280 | 32 | 32 | 0.0453 | 0.0011 |
| 14 | 32 | 2 | 1280 | 16 | 16 | 0.0355 | 0.0007 |
| 15 | 32 | 8 | 640 | 32 | 32 | 0.0592 | 0.0031 |
| 16 | 32 | 16 | 1280 | 16 | 16 | 0.0611 | 0.0031 |

```python
import torch

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    GroupNorm operation.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Scale parameter of shape (C,)
        bias: Shift parameter of shape (C,)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    num_groups = 32
    
    # Reshape to separate groups: (B, num_groups, C//num_groups, H, W)
    x_grouped = x.view(B, num_groups, C // num_groups, H, W)
    
    # Compute mean and variance per group using float32 for numerical stability
    x_grouped_f32 = x_grouped.to(torch.float32)
    mean = x_grouped_f32.mean(dim=[2, 3, 4], keepdim=True)
    var = x_grouped_f32.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
    
    # Normalize: (x - mean) / sqrt(var + eps)
    x_normalized = (x_grouped_f32 - mean) / torch.sqrt(var + eps)
    
    # Reshape back to (B, C, H, W)
    x_normalized = x_normalized.view(B, C, H, W)
    
    # Apply affine transformation
    # Reshape weight and bias for broadcasting: (1, C, 1, 1)
    weight_reshaped = weight.view(1, C, 1, 1)
    bias_reshaped = bias.view(1, C, 1, 1)
    output = x_normalized * weight_reshaped + bias_reshaped
    
    return output.to(x.dtype)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002158 ms |
| - | Scoring Baseline | 0.500000 | 0.056173 ms |
| - | Reference Implementation | 0.334921 | 0.110338 ms |
