## Description

ResNet block with time embedding injection for Stable Diffusion UNet. Performs: GroupNorm -> SiLU -> Conv2d -> time_emb projection -> addition -> GroupNorm -> SiLU -> Dropout -> Conv2d -> residual addition. This is a fundamental building block that appears in every down/up block of Stable Diffusion.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, in_channels, height, width] | float32 |
| time_emb | [batch_size, time_emb_channels] | float32 |
| norm1_weight | [in_channels] | float32 |
| norm1_bias | [in_channels] | float32 |
| conv1_weight | [out_channels, in_channels, 3, 3] | float32 |
| conv1_bias | [out_channels] | float32 |
| time_emb_proj_weight | [out_channels, time_emb_channels] | float32 |
| time_emb_proj_bias | [out_channels] | float32 |
| norm2_weight | [out_channels] | float32 |
| norm2_bias | [out_channels] | float32 |
| conv2_weight | [out_channels, out_channels, 3, 3] | float32 |
| conv2_bias | [out_channels] | float32 |
| norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, out_channels, height, width] | float32 |

| # | in_channels | out_channels | time_emb_channels | norm_num_groups | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 320 | 320 | 1280 | 32 | 1 | 16 | 16 | 0.1322 | 0.0009 |
| 2 | 320 | 320 | 1280 | 32 | 4 | 64 | 64 | 0.2928 | 0.0337 |
| 3 | 320 | 320 | 1280 | 32 | 4 | 32 | 32 | 0.1795 | 0.0087 |
| 4 | 320 | 320 | 1280 | 32 | 2 | 96 | 96 | 0.3334 | 0.0379 |
| 5 | 320 | 320 | 1280 | 32 | 8 | 37 | 37 | 0.3029 | 0.0227 |
| 6 | 320 | 320 | 1280 | 32 | 1 | 131 | 131 | 0.3100 | 0.0353 |
| 7 | 320 | 320 | 1280 | 32 | 1 | 48 | 48 | 0.1644 | 0.0051 |
| 8 | 320 | 320 | 1280 | 32 | 4 | 48 | 48 | 0.2736 | 0.0192 |
| 9 | 320 | 320 | 1280 | 32 | 2 | 67 | 67 | 0.2885 | 0.0187 |
| 10 | 320 | 320 | 1280 | 32 | 64 | 16 | 16 | 0.2918 | 0.0338 |
| 11 | 320 | 320 | 1280 | 32 | 4 | 96 | 96 | 0.4910 | 0.0754 |
| 12 | 320 | 320 | 1280 | 32 | 1 | 128 | 128 | 0.3428 | 0.0337 |
| 13 | 320 | 320 | 1280 | 32 | 2 | 32 | 32 | 0.1814 | 0.0046 |
| 14 | 320 | 320 | 1280 | 32 | 2 | 64 | 64 | 0.2749 | 0.0171 |
| 15 | 320 | 320 | 1280 | 32 | 4 | 41 | 41 | 0.2576 | 0.0141 |
| 16 | 320 | 320 | 1280 | 32 | 1 | 53 | 53 | 0.2188 | 0.0061 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    time_emb: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    time_emb_proj_weight: torch.Tensor,
    time_emb_proj_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    norm_eps: float,
):
    """
    ResNet block with time embedding injection.
    
    Flow:
    1. GroupNorm + SiLU + Conv2d on input
    2. SiLU on time_emb, project and add to features
    3. GroupNorm + SiLU + Conv2d
    4. Residual connection (in_channels == out_channels, so no shortcut conv needed)
    """
    # Store input for residual connection
    residual = x
    
    # First conv block: GroupNorm -> SiLU -> Conv2d
    # GroupNorm with 32 groups
    h = F.group_norm(x, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=norm_eps)
    # SiLU activation: x * sigmoid(x)
    h = h * torch.sigmoid(h)
    # Conv2d with 3x3 kernel, padding=1
    h = F.conv2d(h, conv1_weight, conv1_bias, stride=1, padding=1)
    
    # Time embedding injection
    # Apply SiLU to time embedding
    t = time_emb * torch.sigmoid(time_emb)
    # Project time embedding: Linear(time_emb_channels -> out_channels)
    t = F.linear(t, time_emb_proj_weight, time_emb_proj_bias)
    # Reshape for broadcasting: (batch, out_channels) -> (batch, out_channels, 1, 1)
    t = t[:, :, None, None]
    # Add time embedding to features (default mode, not scale-shift)
    h = h + t
    
    # Second conv block: GroupNorm -> SiLU -> Conv2d
    h = F.group_norm(h, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=norm_eps)
    h = h * torch.sigmoid(h)
    # Dropout with p=0.0 is identity, so we skip it
    h = F.conv2d(h, conv2_weight, conv2_bias, stride=1, padding=1)
    
    # Residual connection (no shortcut conv needed since in_channels == out_channels)
    # out_scale_factor = 1.0, so no scaling needed
    output = h + residual
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.015247 ms |
| - | Scoring Baseline | 0.500000 | 0.258177 ms |
| - | Reference Implementation | 0.344231 | 0.475970 ms |
