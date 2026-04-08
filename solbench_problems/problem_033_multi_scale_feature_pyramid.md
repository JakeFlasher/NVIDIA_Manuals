## Description

Multi-scale feature pyramid processing for hierarchical image processing with encoder-decoder architecture and skip connections. Implements U-Net style progressive downsampling and upsampling with channel expansion/reduction at each scale level.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, input_channels, height, width] | float32 |
| encoder_conv0_weight | [32, 3, 3, 3] | float32 |
| encoder_conv0_bias | [32] | float32 |
| downsample_conv0_weight | [64, 32, 3, 3] | float32 |
| downsample_conv0_bias | [64] | float32 |
| encoder_conv1_weight | [64, 64, 3, 3] | float32 |
| encoder_conv1_bias | [64] | float32 |
| downsample_conv1_weight | [128, 64, 3, 3] | float32 |
| downsample_conv1_bias | [128] | float32 |
| encoder_conv2_weight | [128, 128, 3, 3] | float32 |
| encoder_conv2_bias | [128] | float32 |
| bottleneck_weight | [128, 128, 3, 3] | float32 |
| bottleneck_bias | [128] | float32 |
| upsample_conv0_weight | [128, 64, 2, 2] | float32 |
| upsample_conv0_bias | [64] | float32 |
| decoder_conv0_weight | [64, 128, 3, 3] | float32 |
| decoder_conv0_bias | [64] | float32 |
| upsample_conv1_weight | [64, 32, 2, 2] | float32 |
| upsample_conv1_bias | [32] | float32 |
| decoder_conv1_weight | [32, 64, 3, 3] | float32 |
| decoder_conv1_bias | [32] | float32 |
| output_conv_weight | [3, 32, 3, 3] | float32 |
| output_conv_bias | [3] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, input_channels, height, width] | float32 |

| # | input_channels | base_channels | num_scales | scale1_channels | scale2_channels | scale3_channels | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 32 | 3 | 32 | 64 | 128 | 16 | 64 | 64 | 0.4193 | 0.0070 |
| 2 | 3 | 32 | 3 | 32 | 64 | 128 | 8 | 256 | 256 | 1.5041 | 0.0536 |
| 3 | 3 | 32 | 3 | 32 | 64 | 128 | 1 | 512 | 512 | 0.8419 | 0.0270 |
| 4 | 3 | 32 | 3 | 32 | 64 | 128 | 1 | 132 | 132 | 0.2748 | 0.0022 |
| 5 | 3 | 32 | 3 | 32 | 64 | 128 | 32 | 128 | 128 | 2.4806 | 0.0536 |
| 6 | 3 | 32 | 3 | 32 | 64 | 128 | 8 | 64 | 64 | 0.3298 | 0.0037 |
| 7 | 3 | 32 | 3 | 32 | 64 | 128 | 1 | 1024 | 1024 | 2.8042 | 0.1067 |
| 8 | 3 | 32 | 3 | 32 | 64 | 128 | 8 | 372 | 372 | 3.0569 | 0.1127 |
| 9 | 3 | 32 | 3 | 32 | 64 | 128 | 4 | 64 | 64 | 0.2719 | 0.0021 |
| 10 | 3 | 32 | 3 | 32 | 64 | 128 | 1 | 128 | 128 | 0.2680 | 0.0021 |
| 11 | 3 | 32 | 3 | 32 | 64 | 128 | 2 | 256 | 256 | 0.5935 | 0.0137 |
| 12 | 3 | 32 | 3 | 32 | 64 | 128 | 1 | 448 | 448 | 0.7114 | 0.0208 |
| 13 | 3 | 32 | 3 | 32 | 64 | 128 | 2 | 512 | 512 | 1.4984 | 0.0536 |
| 14 | 3 | 32 | 3 | 32 | 64 | 128 | 4 | 128 | 128 | 0.4265 | 0.0070 |
| 15 | 3 | 32 | 3 | 32 | 64 | 128 | 64 | 128 | 128 | 2.6409 | 0.1067 |
| 16 | 3 | 32 | 3 | 32 | 64 | 128 | 16 | 256 | 256 | 2.8452 | 0.1067 |

```python
import math

import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    height = axes_and_scalars["height"]
    width = axes_and_scalars["width"]

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)

    return {
        "x": torch.randn(batch_size, 3, height, width, device=device, generator=g),
        # encoder conv0: 32 out, 3 in, 3x3
        "encoder_conv0_weight": kaiming_conv(32, 3, 3, 3),
        "encoder_conv0_bias": torch.randn(32, device=device, generator=g),
        # downsample conv0: 64 out, 32 in, 3x3
        "downsample_conv0_weight": kaiming_conv(64, 32, 3, 3),
        "downsample_conv0_bias": torch.randn(64, device=device, generator=g),
        # encoder conv1: 64 out, 64 in, 3x3
        "encoder_conv1_weight": kaiming_conv(64, 64, 3, 3),
        "encoder_conv1_bias": torch.randn(64, device=device, generator=g),
        # downsample conv1: 128 out, 64 in, 3x3
        "downsample_conv1_weight": kaiming_conv(128, 64, 3, 3),
        "downsample_conv1_bias": torch.randn(128, device=device, generator=g),
        # encoder conv2: 128 out, 128 in, 3x3
        "encoder_conv2_weight": kaiming_conv(128, 128, 3, 3),
        "encoder_conv2_bias": torch.randn(128, device=device, generator=g),
        # bottleneck: 128 out, 128 in, 3x3
        "bottleneck_weight": kaiming_conv(128, 128, 3, 3),
        "bottleneck_bias": torch.randn(128, device=device, generator=g),
        # upsample conv0 (transposed): 128 in, 64 out, 2x2
        "upsample_conv0_weight": kaiming_conv(128, 64, 2, 2),
        "upsample_conv0_bias": torch.randn(64, device=device, generator=g),
        # decoder conv0: 64 out, 128 in, 3x3
        "decoder_conv0_weight": kaiming_conv(64, 128, 3, 3),
        "decoder_conv0_bias": torch.randn(64, device=device, generator=g),
        # upsample conv1 (transposed): 64 in, 32 out, 2x2
        "upsample_conv1_weight": kaiming_conv(64, 32, 2, 2),
        "upsample_conv1_bias": torch.randn(32, device=device, generator=g),
        # decoder conv1: 32 out, 64 in, 3x3
        "decoder_conv1_weight": kaiming_conv(32, 64, 3, 3),
        "decoder_conv1_bias": torch.randn(32, device=device, generator=g),
        # output conv: 3 out, 32 in, 3x3
        "output_conv_weight": kaiming_conv(3, 32, 3, 3),
        "output_conv_bias": torch.randn(3, device=device, generator=g),
    }

@torch.no_grad()
def run(
    x: torch.Tensor,
    encoder_conv0_weight: torch.Tensor,
    encoder_conv0_bias: torch.Tensor,
    downsample_conv0_weight: torch.Tensor,
    downsample_conv0_bias: torch.Tensor,
    encoder_conv1_weight: torch.Tensor,
    encoder_conv1_bias: torch.Tensor,
    downsample_conv1_weight: torch.Tensor,
    downsample_conv1_bias: torch.Tensor,
    encoder_conv2_weight: torch.Tensor,
    encoder_conv2_bias: torch.Tensor,
    bottleneck_weight: torch.Tensor,
    bottleneck_bias: torch.Tensor,
    upsample_conv0_weight: torch.Tensor,
    upsample_conv0_bias: torch.Tensor,
    decoder_conv0_weight: torch.Tensor,
    decoder_conv0_bias: torch.Tensor,
    upsample_conv1_weight: torch.Tensor,
    upsample_conv1_bias: torch.Tensor,
    decoder_conv1_weight: torch.Tensor,
    decoder_conv1_bias: torch.Tensor,
    output_conv_weight: torch.Tensor,
    output_conv_bias: torch.Tensor,
):
    # Encoder scale 0: input_channels -> base_channels (32)
    enc0 = F.conv2d(x, encoder_conv0_weight, encoder_conv0_bias, padding=1)
    
    # Downsample 0: 32 -> 64, stride=2
    down0 = F.conv2d(enc0, downsample_conv0_weight, downsample_conv0_bias, stride=2, padding=1)
    
    # Encoder scale 1: 64 -> 64
    enc1 = F.conv2d(down0, encoder_conv1_weight, encoder_conv1_bias, padding=1)
    
    # Downsample 1: 64 -> 128, stride=2
    down1 = F.conv2d(enc1, downsample_conv1_weight, downsample_conv1_bias, stride=2, padding=1)
    
    # Encoder scale 2: 128 -> 128
    enc2 = F.conv2d(down1, encoder_conv2_weight, encoder_conv2_bias, padding=1)
    
    # Bottleneck: 128 -> 128
    feat = F.conv2d(enc2, bottleneck_weight, bottleneck_bias, padding=1)
    
    # Decoder path with skip connections
    # Upsample 0: 128 -> 64, stride=2
    up0 = F.conv_transpose2d(feat, upsample_conv0_weight, upsample_conv0_bias, stride=2)
    
    # Skip connection with enc1 (64 channels)
    # Concatenate: 64 + 64 = 128 channels
    skip0 = torch.cat([up0, enc1], dim=1)
    
    # Decoder conv 0: 128 -> 64
    dec0 = F.conv2d(skip0, decoder_conv0_weight, decoder_conv0_bias, padding=1)
    
    # Upsample 1: 64 -> 32, stride=2
    up1 = F.conv_transpose2d(dec0, upsample_conv1_weight, upsample_conv1_bias, stride=2)
    
    # Skip connection with enc0 (32 channels)
    # Concatenate: 32 + 32 = 64 channels
    skip1 = torch.cat([up1, enc0], dim=1)
    
    # Decoder conv 1: 64 -> 32
    dec1 = F.conv2d(skip1, decoder_conv1_weight, decoder_conv1_bias, padding=1)
    
    # Output projection: 32 -> 3
    output = F.conv2d(dec1, output_conv_weight, output_conv_bias, padding=1)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.018987 ms |
| - | Scoring Baseline | 0.500000 | 0.888630 ms |
| - | Reference Implementation | 0.455340 | 1.065796 ms |
