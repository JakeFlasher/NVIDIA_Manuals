## Description

Convolutional residual block that combines input convolution (3->32 channels), output convolution (32->3 channels), and residual skip connection. This is the atomic building block for image restoration networks like NAFNet.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, in_channels, height, width] | float32 |
| conv_in_weight | [hidden_channels, in_channels, kernel_size, kernel_size] | float32 |
| conv_in_bias | [hidden_channels] | float32 |
| conv_out_weight | [out_channels, hidden_channels, kernel_size, kernel_size] | float32 |
| conv_out_bias | [out_channels] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, out_channels, height, width] | float32 |

| # | in_channels | hidden_channels | out_channels | kernel_size | padding | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 32 | 3 | 3 | 1 | 8 | 512 | 512 | 0.5136 | 0.0044 |
| 2 | 3 | 32 | 3 | 3 | 1 | 32 | 256 | 256 | 0.6549 | 0.0044 |
| 3 | 3 | 32 | 3 | 3 | 1 | 2 | 256 | 256 | 0.1949 | 0.0007 |
| 4 | 3 | 32 | 3 | 3 | 1 | 4 | 512 | 512 | 0.2837 | 0.0024 |
| 5 | 3 | 32 | 3 | 3 | 1 | 1 | 128 | 128 | 0.0624 | 0.0004 |
| 6 | 3 | 32 | 3 | 3 | 1 | 16 | 64 | 64 | 0.1162 | 0.0005 |
| 7 | 3 | 32 | 3 | 3 | 1 | 8 | 373 | 373 | 0.5273 | 0.0025 |
| 8 | 3 | 32 | 3 | 3 | 1 | 64 | 128 | 128 | 0.5128 | 0.0024 |
| 9 | 3 | 32 | 3 | 3 | 1 | 16 | 128 | 128 | 0.1955 | 0.0009 |
| 10 | 3 | 32 | 3 | 3 | 1 | 1 | 64 | 64 | 0.0562 | 0.0004 |
| 11 | 3 | 32 | 3 | 3 | 1 | 1 | 512 | 512 | 0.1185 | 0.0009 |
| 12 | 3 | 32 | 3 | 3 | 1 | 2 | 512 | 512 | 0.1739 | 0.0014 |
| 13 | 3 | 32 | 3 | 3 | 1 | 2 | 64 | 64 | 0.0559 | 0.0004 |
| 14 | 3 | 32 | 3 | 3 | 1 | 1 | 131 | 131 | 0.0623 | 0.0004 |
| 15 | 3 | 32 | 3 | 3 | 1 | 2 | 541 | 541 | 0.1863 | 0.0015 |
| 16 | 3 | 32 | 3 | 3 | 1 | 16 | 256 | 256 | 0.4738 | 0.0024 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    conv_in_weight: torch.Tensor,
    conv_in_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    conv_out_bias: torch.Tensor,
):
    """
    Convolutional residual block:
    1. conv_in: (B, 3, H, W) -> (B, 32, H, W)
    2. conv_out: (B, 32, H, W) -> (B, 3, H, W)
    3. residual add: output + input
    """
    # Store input for residual connection
    identity = x
    
    # Feature extraction convolution: (B, 3, H, W) -> (B, 32, H, W)
    out = F.conv2d(x, conv_in_weight, conv_in_bias, padding=1)
    
    # Feature reconstruction convolution: (B, 32, H, W) -> (B, 3, H, W)
    out = F.conv2d(out, conv_out_weight, conv_out_bias, padding=1)
    
    # Residual connection: element-wise addition
    out = out + identity
    
    return out
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001156 ms |
| - | Scoring Baseline | 0.500000 | 0.188477 ms |
| - | Reference Implementation | 0.370001 | 0.332791 ms |
