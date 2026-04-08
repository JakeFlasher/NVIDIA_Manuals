## Description

Fused Conv1d projection to 2x channels followed by channel-wise split into mean and log-variance statistics. Performs a single Conv1d with output channels = 2*out_channels, then splits the result along the channel dimension into separate mean and logs tensors.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, in_channels, time] | float32 |
| weight | [out_channels_2, in_channels, kernel_size] | float32 |
| bias | [out_channels_2] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| mean | [batch_size, out_channels, time] | float32 |
| logs | [batch_size, out_channels, time] | float32 |

| # | in_channels | out_channels | kernel_size | out_channels_2 | batch_size | time | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 192 | 192 | 1 | 384 | 8 | 1571 | 0.2058 | 0.0023 |
| 2 | 192 | 192 | 1 | 384 | 4 | 293 | 0.0365 | 0.0006 |
| 3 | 192 | 192 | 1 | 384 | 16 | 2048 | 0.2160 | 0.0053 |
| 4 | 192 | 192 | 1 | 384 | 2 | 211 | 0.0332 | 0.0005 |
| 5 | 192 | 192 | 1 | 384 | 16 | 449 | 0.0503 | 0.0015 |
| 6 | 192 | 192 | 1 | 384 | 16 | 997 | 0.0741 | 0.0028 |
| 7 | 192 | 192 | 1 | 384 | 4 | 512 | 0.0371 | 0.0007 |
| 8 | 192 | 192 | 1 | 384 | 2 | 2447 | 0.0428 | 0.0012 |
| 9 | 192 | 192 | 1 | 384 | 32 | 1879 | 0.3503 | 0.0095 |
| 10 | 192 | 192 | 1 | 384 | 2 | 256 | 0.0337 | 0.0005 |
| 11 | 192 | 192 | 1 | 384 | 32 | 1087 | 0.1284 | 0.0056 |
| 12 | 192 | 192 | 1 | 384 | 8 | 3089 | 0.1036 | 0.0041 |
| 13 | 192 | 192 | 1 | 384 | 4 | 2767 | 0.0573 | 0.0021 |
| 14 | 192 | 192 | 1 | 384 | 8 | 1024 | 0.0411 | 0.0017 |
| 15 | 192 | 192 | 1 | 384 | 64 | 613 | 0.1475 | 0.0063 |
| 16 | 192 | 192 | 1 | 384 | 32 | 541 | 0.0859 | 0.0030 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Fused Conv1d projection and split.
    
    Args:
        x: Input tensor of shape [batch_size, in_channels, time]
        weight: Conv1d weight of shape [out_channels*2, in_channels, kernel_size]
        bias: Conv1d bias of shape [out_channels*2]
    
    Returns:
        mean: Mean statistics of shape [batch_size, out_channels, time]
        logs: Log-variance statistics of shape [batch_size, out_channels, time]
    """
    # Perform Conv1d projection to 2x channels
    # For kernel_size=1, no padding needed
    # Shape: [batch_size, out_channels*2, time]
    stats = F.conv1d(x, weight, bias, padding=0)
    
    # Split along channel dimension into mean and logs
    out_channels = weight.shape[0] // 2
    mean, logs = torch.split(stats, out_channels, dim=1)
    
    return mean, logs
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002047 ms |
| - | Reference Implementation | 0.540281 | 0.065010 ms |
| - | Scoring Baseline | 0.500000 | 0.076888 ms |
