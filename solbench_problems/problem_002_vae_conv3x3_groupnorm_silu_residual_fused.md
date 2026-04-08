## Description

Complete fused residual block combining two sequential Conv3x3->GroupNorm->SiLU operations with residual addition. This is the fundamental building block of Sana's VAE encoder/decoder: input -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> add(input). Processes feature maps with 256 channels and 32 groups for GroupNorm.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, channels, height, width] | float32 |
| conv1_weight | [channels, channels, kernel_size, kernel_size] | float32 |
| norm1_weight | [channels] | float32 |
| norm1_bias | [channels] | float32 |
| conv2_weight | [channels, channels, kernel_size, kernel_size] | float32 |
| norm2_weight | [channels] | float32 |
| norm2_bias | [channels] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, channels, height, width] | float32 |

| # | channels | num_groups | kernel_size | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 32 | 3 | 16 | 64 | 64 | 0.6273 | 0.0858 |
| 2 | 256 | 32 | 3 | 1 | 128 | 128 | 0.2534 | 0.0217 |
| 3 | 256 | 32 | 3 | 1 | 131 | 131 | 0.2408 | 0.0228 |
| 4 | 256 | 32 | 3 | 32 | 128 | 128 | 2.6751 | 0.6833 |
| 5 | 256 | 32 | 3 | 2 | 128 | 128 | 0.3516 | 0.0431 |
| 6 | 256 | 32 | 3 | 4 | 128 | 128 | 0.5291 | 0.0858 |
| 7 | 256 | 32 | 3 | 2 | 256 | 256 | 0.9822 | 0.1711 |
| 8 | 256 | 32 | 3 | 4 | 64 | 64 | 0.2464 | 0.0217 |
| 9 | 256 | 32 | 3 | 64 | 64 | 64 | 1.4506 | 0.3419 |
| 10 | 256 | 32 | 3 | 2 | 64 | 64 | 0.2333 | 0.0111 |
| 11 | 256 | 32 | 3 | 1 | 1024 | 1024 | 5.8798 | 1.3663 |
| 12 | 256 | 32 | 3 | 1 | 293 | 293 | 0.6735 | 0.1122 |
| 13 | 256 | 32 | 3 | 4 | 256 | 256 | 1.6709 | 0.3419 |
| 14 | 256 | 32 | 3 | 32 | 64 | 64 | 0.8144 | 0.1711 |
| 15 | 256 | 32 | 3 | 8 | 64 | 64 | 0.3315 | 0.0431 |
| 16 | 256 | 32 | 3 | 1 | 768 | 768 | 3.7849 | 0.7687 |
| 17 | 256 | 32 | 3 | 4 | 128 | 96 | 0.4373 | 0.0644 |
| 18 | 256 | 32 | 3 | 4 | 96 | 128 | 0.4279 | 0.0644 |
| 19 | 256 | 32 | 3 | 8 | 64 | 128 | 0.4967 | 0.0858 |
| 20 | 256 | 32 | 3 | 2 | 256 | 192 | 0.7516 | 0.1284 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    """
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        conv1_weight: First conv weights (C, C, 3, 3)
        norm1_weight: First GroupNorm scale (C,)
        norm1_bias: First GroupNorm bias (C,)
        conv2_weight: Second conv weights (C, C, 3, 3)
        norm2_weight: Second GroupNorm scale (C,)
        norm2_bias: Second GroupNorm bias (C,)
        eps: Epsilon for GroupNorm numerical stability
        
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    num_groups = 32
    
    # Save residual
    residual = x
    
    # First path: Conv3x3 -> GroupNorm -> SiLU
    out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
    out = F.silu(out)
    
    # Second path: Conv3x3 -> GroupNorm -> SiLU
    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
    out = F.silu(out)
    
    # Residual connection
    out = out + residual
    
    return out
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.103659 ms |
| 1st place | AKO4ALL_L1 | 0.550478 | 0.588358 ms |
| 2nd place | Jasper Otter | 0.503544 | 0.684845 ms |
| - | Scoring Baseline | 0.500000 | 0.697215 ms |
| 3rd place | Nestor_Qin | 0.426796 | 0.907494 ms |
| #4 | Quick Impala | 0.357346 | 1.168550 ms |
| #5 | Bob Huang | 0.356062 | 1.268258 ms |
| - | Reference Implementation | 0.328380 | 1.392713 ms |
