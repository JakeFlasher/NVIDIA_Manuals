## Description

VAE encoder downsampling block for video-to-latent compression. Converts raw video frames to compressed latent representations through spatial and temporal downsampling with 3D convolutions, group normalization, SiLU activations, and residual connections.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, in_channels, frames, height, width] | float32 |
| input_conv_weight | [hidden_div4, in_channels, 3, 3, 3] | float32 |
| input_conv_bias | [hidden_div4] | float32 |
| down1_gn_weight | [hidden_div4] | float32 |
| down1_gn_bias | [hidden_div4] | float32 |
| down1_conv_weight | [hidden_div2, hidden_div4, 3, 4, 4] | float32 |
| down1_conv_bias | [hidden_div2] | float32 |
| down1_res_weight | [hidden_div2, hidden_div4, 1, 1, 1] | float32 |
| down1_res_bias | [hidden_div2] | float32 |
| down2_gn_weight | [hidden_div2] | float32 |
| down2_gn_bias | [hidden_div2] | float32 |
| down2_conv_weight | [hidden_size, hidden_div2, 3, 4, 4] | float32 |
| down2_conv_bias | [hidden_size] | float32 |
| down2_res_weight | [hidden_size, hidden_div2, 1, 1, 1] | float32 |
| down2_res_bias | [hidden_size] | float32 |
| down3_gn_weight | [hidden_size] | float32 |
| down3_gn_bias | [hidden_size] | float32 |
| down3_conv_weight | [hidden_size, hidden_size, 3, 4, 4] | float32 |
| down3_conv_bias | [hidden_size] | float32 |
| down3_res_weight | [hidden_size, hidden_size, 1, 1, 1] | float32 |
| down3_res_bias | [hidden_size] | float32 |
| latent_gn_weight | [hidden_size] | float32 |
| latent_gn_bias | [hidden_size] | float32 |
| latent_conv_weight | [latent_channels, hidden_size, 3, 3, 3] | float32 |
| latent_conv_bias | [latent_channels] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, latent_channels, frames_div4, height_div8, width_div8] | float32 |

| # | in_channels | latent_channels | hidden_size | num_groups | hidden_div4 | hidden_div2 | groups_div4 | groups_div2 | batch_size | frames | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 16 | 384 | 384 | 16.7471 | 3.7190 |
| 2 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 32 | 8 | 64 | 64 | 7.9853 | 1.6531 |
| 3 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 4 | 8 | 256 | 256 | 14.9335 | 3.3058 |
| 4 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 16 | 448 | 448 | 23.6947 | 5.0618 |
| 5 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 16 | 192 | 192 | 4.7173 | 0.9300 |
| 6 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 2 | 8 | 256 | 256 | 7.9312 | 1.6531 |
| 7 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 64 | 8 | 64 | 64 | 15.1082 | 3.3058 |
| 8 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 2 | 24 | 128 | 128 | 6.1041 | 1.2399 |
| 9 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 2 | 16 | 192 | 192 | 8.7275 | 1.8597 |
| 10 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 24 | 256 | 256 | 11.5545 | 2.4795 |
| 11 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 8 | 16 | 128 | 128 | 14.9190 | 3.3058 |
| 12 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 48 | 128 | 128 | 6.0816 | 1.2399 |
| 13 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 16 | 256 | 256 | 7.9099 | 1.6531 |
| 14 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 32 | 256 | 256 | 15.2797 | 3.3058 |
| 15 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 4 | 16 | 128 | 128 | 7.9057 | 1.6531 |
| 16 | 3 | 16 | 1024 | 32 | 256 | 512 | 8 | 16 | 1 | 8 | 384 | 384 | 8.5115 | 1.8597 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    input_conv_weight: torch.Tensor,
    input_conv_bias: torch.Tensor,
    down1_gn_weight: torch.Tensor,
    down1_gn_bias: torch.Tensor,
    down1_conv_weight: torch.Tensor,
    down1_conv_bias: torch.Tensor,
    down1_res_weight: torch.Tensor,
    down1_res_bias: torch.Tensor,
    down2_gn_weight: torch.Tensor,
    down2_gn_bias: torch.Tensor,
    down2_conv_weight: torch.Tensor,
    down2_conv_bias: torch.Tensor,
    down2_res_weight: torch.Tensor,
    down2_res_bias: torch.Tensor,
    down3_gn_weight: torch.Tensor,
    down3_gn_bias: torch.Tensor,
    down3_conv_weight: torch.Tensor,
    down3_conv_bias: torch.Tensor,
    down3_res_weight: torch.Tensor,
    down3_res_bias: torch.Tensor,
    latent_gn_weight: torch.Tensor,
    latent_gn_bias: torch.Tensor,
    latent_conv_weight: torch.Tensor,
    latent_conv_bias: torch.Tensor,
    eps: float,
):
    num_groups = 32
    
    # Initial projection: (B, 3, T, H, W) -> (B, 256, T, H, W)
    h = F.conv3d(x, input_conv_weight, input_conv_bias, stride=1, padding=1)
    
    # First downsampling block with residual
    # Group norm + SiLU + Conv3d
    identity1 = F.conv3d(h, down1_res_weight, down1_res_bias, stride=2)
    h = F.group_norm(h, num_groups // 4, down1_gn_weight, down1_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down1_conv_weight, down1_conv_bias, stride=2, padding=1)
    h = h + identity1
    
    # Second downsampling block with residual
    identity2 = F.conv3d(h, down2_res_weight, down2_res_bias, stride=2)
    h = F.group_norm(h, num_groups // 2, down2_gn_weight, down2_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down2_conv_weight, down2_conv_bias, stride=2, padding=1)
    h = h + identity2
    
    # Third downsampling block with residual (spatial only)
    identity3 = F.conv3d(h, down3_res_weight, down3_res_bias, stride=(1, 2, 2))
    h = F.group_norm(h, num_groups, down3_gn_weight, down3_gn_bias, eps)
    h = F.silu(h)
    h = F.conv3d(h, down3_conv_weight, down3_conv_bias, stride=(1, 2, 2), padding=1)
    h = h + identity3
    
    # Latent projection
    h = F.group_norm(h, num_groups, latent_gn_weight, latent_gn_bias, eps)
    h = F.silu(h)
    output = F.conv3d(h, latent_conv_weight, latent_conv_bias, stride=1, padding=1)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 2.148307 ms |
| - | Scoring Baseline | 0.500000 | 10.122410 ms |
| - | Reference Implementation | 0.184651 | 41.693096 ms |
