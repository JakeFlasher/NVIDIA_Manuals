## Description

Strided 2D convolution for spatial downsampling used in NAFNet's encoder path. Performs convolution with stride > 1 to reduce spatial dimensions while increasing channel capacity. Combines spatial reduction with feature transformation in a single operation.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, in_channels, height, width] | float32 |
| weight | [out_channels, in_channels, kernel_size, kernel_size] | float32 |
| bias | [out_channels] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, out_channels, out_height, out_width] | float32 |

| # | in_channels | out_channels | kernel_size | stride | padding | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 3 | 2 | 1 | 1 | 384 | 384 | 0.0915 | 0.0023 |
| 2 | 32 | 64 | 3 | 2 | 1 | 1 | 512 | 512 | 0.1206 | 0.0037 |
| 3 | 32 | 64 | 3 | 2 | 1 | 16 | 384 | 384 | 0.4310 | 0.0299 |
| 4 | 32 | 64 | 3 | 2 | 1 | 2 | 449 | 449 | 0.1383 | 0.0055 |
| 5 | 32 | 64 | 3 | 2 | 1 | 2 | 211 | 211 | 0.0922 | 0.0015 |
| 6 | 32 | 64 | 3 | 2 | 1 | 2 | 512 | 512 | 0.1495 | 0.0070 |
| 7 | 32 | 64 | 3 | 2 | 1 | 1 | 1024 | 1024 | 0.2311 | 0.0135 |
| 8 | 32 | 64 | 3 | 2 | 1 | 4 | 512 | 512 | 0.2349 | 0.0135 |
| 9 | 32 | 64 | 3 | 2 | 1 | 4 | 256 | 256 | 0.1129 | 0.0037 |
| 10 | 32 | 64 | 3 | 2 | 1 | 16 | 256 | 256 | 0.2342 | 0.0135 |
| 11 | 32 | 64 | 3 | 2 | 1 | 1 | 373 | 373 | 0.0925 | 0.0021 |
| 12 | 32 | 64 | 3 | 2 | 1 | 64 | 128 | 128 | 0.2363 | 0.0135 |
| 13 | 32 | 64 | 3 | 2 | 1 | 2 | 256 | 256 | 0.0962 | 0.0020 |
| 14 | 32 | 64 | 3 | 2 | 1 | 1 | 131 | 131 | 0.0891 | 0.0006 |
| 15 | 32 | 64 | 3 | 2 | 1 | 2 | 1024 | 1024 | 0.4008 | 0.0267 |
| 16 | 32 | 64 | 3 | 2 | 1 | 8 | 256 | 256 | 0.1493 | 0.0070 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Strided 2D convolution for spatial downsampling.
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Convolution weights of shape (out_channels, in_channels, kernel_size, kernel_size)
        bias: Bias tensor of shape (out_channels,)
    
    Returns:
        Output tensor of shape (batch_size, out_channels, out_height, out_width)
        where out_height = (height + 2*padding - kernel_size) // stride + 1
        and out_width = (width + 2*padding - kernel_size) // stride + 1
    """
    # Fixed constants for NAFNet downsampling
    stride = 2
    padding = 1
    
    # Perform strided convolution
    output = F.conv2d(x, weight, bias, stride=stride, padding=padding)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.005567 ms |
| - | Scoring Baseline | 0.500000 | 0.157665 ms |
| - | Reference Implementation | 0.376434 | 0.258794 ms |
