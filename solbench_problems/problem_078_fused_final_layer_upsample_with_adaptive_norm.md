## Description

Fused UNet upsampling path that combines timestep-conditioned adaptive group normalization with ResBlock upsampling and final convolution to convert transformer hidden states back to VAE latent space. This critical operation runs at every diffusion step for image generation.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, hidden_size] | bfloat16 |
| timestep_emb | [batch_size, emb_channels] | bfloat16 |
| time_emb_mlp_linear_weight | [double_hidden_channels, emb_channels] | bfloat16 |
| time_emb_mlp_linear_bias | [double_hidden_channels] | bfloat16 |
| resblock_in_norm_weight | [hidden_size] | bfloat16 |
| resblock_in_norm_bias | [hidden_size] | bfloat16 |
| resblock_in_conv_weight | [hidden_channels, hidden_size, 3, 3] | bfloat16 |
| resblock_in_conv_bias | [hidden_channels] | bfloat16 |
| resblock_emb_linear_weight | [double_hidden_channels, emb_channels] | bfloat16 |
| resblock_emb_linear_bias | [double_hidden_channels] | bfloat16 |
| resblock_out_norm_weight | [hidden_channels] | bfloat16 |
| resblock_out_norm_bias | [hidden_channels] | bfloat16 |
| resblock_out_conv_weight | [hidden_channels, hidden_channels, 3, 3] | bfloat16 |
| resblock_out_conv_bias | [hidden_channels] | bfloat16 |
| resblock_skip_conv_weight | [hidden_channels, hidden_size, 1, 1] | bfloat16 |
| resblock_skip_conv_bias | [hidden_channels] | bfloat16 |
| final_norm_weight | [hidden_channels] | bfloat16 |
| final_norm_bias | [hidden_channels] | bfloat16 |
| final_conv_weight | [out_channels, hidden_channels, 3, 3] | bfloat16 |
| final_conv_bias | [out_channels] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, out_channels, token_h, token_w] | bfloat16 |

| # | patch_size | hidden_size | emb_channels | hidden_channels | out_channels | double_hidden_channels | batch_size | token_h | token_w | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 1 | 64 | 64 | 4.0531 | 0.9542 |
| 2 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 8 | 16 | 16 | 2.2695 | 0.4777 |
| 3 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 4 | 16 | 16 | 1.8133 | 0.2391 |
| 4 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 1 | 11 | 11 | 1.5188 | 0.0286 |
| 5 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 1 | 53 | 53 | 3.2083 | 0.6545 |
| 6 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 8 | 23 | 23 | 3.2637 | 0.9863 |
| 7 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 16 | 32 | 32 | 8.4571 | 3.8163 |
| 8 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 32 | 32 | 32 | 15.4617 | 7.6322 |
| 9 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 2 | 17 | 17 | 1.7039 | 0.1351 |
| 10 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 8 | 47 | 47 | 9.6448 | 4.1158 |
| 11 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 2 | 48 | 48 | 3.7112 | 1.0735 |
| 12 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 1 | 29 | 29 | 1.9282 | 0.1963 |
| 13 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 2 | 59 | 59 | 5.2233 | 1.6216 |
| 14 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 64 | 16 | 16 | 8.3486 | 3.8190 |
| 15 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 8 | 32 | 32 | 4.9731 | 1.9083 |
| 16 | 1 | 4096 | 4096 | 3072 | 4 | 6144 | 2 | 41 | 41 | 3.2301 | 0.7834 |

```python
import math

import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    token_h = axes_and_scalars["token_h"]
    token_w = axes_and_scalars["token_w"]
    seq_len = token_h * token_w

    hidden_size = 4096
    emb_channels = 4096
    hidden_channels = 3072
    out_channels = 4
    double_hidden_channels = hidden_channels * 2
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    return {
        "x": torch.randn(batch_size, seq_len, hidden_size, device=device, generator=g).to(dtype),
        # Time embedding — small magnitude
        "timestep_emb": (torch.randn(batch_size, emb_channels, device=device, generator=g) * 0.1).to(dtype),
        # Weight matrices — Xavier init
        "time_emb_mlp_linear_weight": xavier(double_hidden_channels, emb_channels),
        "time_emb_mlp_linear_bias": torch.randn(double_hidden_channels, device=device, generator=g).to(dtype),
        # Norm weights: ones; norm biases: zeros
        "resblock_in_norm_weight": torch.ones(hidden_size, device=device, dtype=dtype),
        "resblock_in_norm_bias": torch.zeros(hidden_size, device=device, dtype=dtype),
        # Conv weights — Kaiming init
        "resblock_in_conv_weight": kaiming_conv(hidden_channels, hidden_size, 3, 3),
        "resblock_in_conv_bias": torch.randn(hidden_channels, device=device, generator=g).to(dtype),
        # Embedding linear — Xavier init
        "resblock_emb_linear_weight": xavier(double_hidden_channels, emb_channels),
        "resblock_emb_linear_bias": torch.randn(double_hidden_channels, device=device, generator=g).to(dtype),
        # Norm weights: ones; norm biases: zeros
        "resblock_out_norm_weight": torch.ones(hidden_channels, device=device, dtype=dtype),
        "resblock_out_norm_bias": torch.zeros(hidden_channels, device=device, dtype=dtype),
        # Conv weights — Kaiming init
        "resblock_out_conv_weight": kaiming_conv(hidden_channels, hidden_channels, 3, 3),
        "resblock_out_conv_bias": torch.randn(hidden_channels, device=device, generator=g).to(dtype),
        # Skip conv — Kaiming (1x1)
        "resblock_skip_conv_weight": kaiming_conv(hidden_channels, hidden_size, 1, 1),
        "resblock_skip_conv_bias": torch.randn(hidden_channels, device=device, generator=g).to(dtype),
        # Final norm weights: ones; biases: zeros
        "final_norm_weight": torch.ones(hidden_channels, device=device, dtype=dtype),
        "final_norm_bias": torch.zeros(hidden_channels, device=device, dtype=dtype),
        # Final conv — Kaiming init
        "final_conv_weight": kaiming_conv(out_channels, hidden_channels, 3, 3),
        "final_conv_bias": torch.randn(out_channels, device=device, generator=g).to(dtype),
        # Epsilon
        "eps": 1e-6,
    }

@torch.no_grad()
def run(
    x: torch.Tensor,
    timestep_emb: torch.Tensor,
    time_emb_mlp_linear_weight: torch.Tensor,
    time_emb_mlp_linear_bias: torch.Tensor,
    resblock_in_norm_weight: torch.Tensor,
    resblock_in_norm_bias: torch.Tensor,
    resblock_in_conv_weight: torch.Tensor,
    resblock_in_conv_bias: torch.Tensor,
    resblock_emb_linear_weight: torch.Tensor,
    resblock_emb_linear_bias: torch.Tensor,
    resblock_out_norm_weight: torch.Tensor,
    resblock_out_norm_bias: torch.Tensor,
    resblock_out_conv_weight: torch.Tensor,
    resblock_out_conv_bias: torch.Tensor,
    resblock_skip_conv_weight: torch.Tensor,
    resblock_skip_conv_bias: torch.Tensor,
    final_norm_weight: torch.Tensor,
    final_norm_bias: torch.Tensor,
    final_conv_weight: torch.Tensor,
    final_conv_bias: torch.Tensor,
    eps: float,
):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = x.shape[2]
    hidden_channels = resblock_in_conv_weight.shape[0]
    out_channels = final_conv_weight.shape[0]
    
    # Infer token_h and token_w from seq_len (assuming square or close to square)
    token_h = int(seq_len ** 0.5)
    token_w = seq_len // token_h
    while token_h * token_w != seq_len:
        token_h -= 1
        token_w = seq_len // token_h
    
    # Reshape from sequence to spatial: (B, seq_len, C) -> (B, C, H, W)
    x_spatial = x.reshape(batch_size, token_h, token_w, hidden_size)
    x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
    
    # Process timestep embedding through MLP: SiLU + Linear
    emb = F.silu(timestep_emb)
    emb = F.linear(emb, time_emb_mlp_linear_weight, time_emb_mlp_linear_bias)  # (B, 2 * hidden_channels)
    
    # ResBlock forward pass
    # Input processing: GroupNorm + SiLU + Conv
    h = F.group_norm(x_spatial, 32, resblock_in_norm_weight, resblock_in_norm_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resblock_in_conv_weight, resblock_in_conv_bias, padding=1)
    
    # Get scale and shift from timestep embedding for adaptive normalization
    emb_out = F.silu(timestep_emb)
    emb_out = F.linear(emb_out, resblock_emb_linear_weight, resblock_emb_linear_bias)  # (B, 2 * out_channels)
    emb_out = emb_out.unsqueeze(-1).unsqueeze(-1)  # (B, 2 * out_channels, 1, 1)
    
    # Adaptive Group Normalization
    h = F.group_norm(h, 32, resblock_out_norm_weight, resblock_out_norm_bias, eps)
    scale, shift = torch.chunk(emb_out, 2, dim=1)  # Each (B, out_channels, 1, 1)
    h = h * (1.0 + scale) + shift  # Adaptive affine transformation
    
    # Output processing: SiLU + Conv
    h = F.silu(h)
    h = F.conv2d(h, resblock_out_conv_weight, resblock_out_conv_bias, padding=1)
    
    # Skip connection (with 1x1 conv since in_channels != out_channels)
    skip = F.conv2d(x_spatial, resblock_skip_conv_weight, resblock_skip_conv_bias)
    h = skip + h
    
    # Final normalization and projection to VAE latent space
    h = F.group_norm(h, 32, final_norm_weight, final_norm_bias, eps)
    h = F.silu(h)
    output = F.conv2d(h, final_conv_weight, final_conv_bias, padding=1)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.841796 ms |
| - | Scoring Baseline | 0.500000 | 3.899399 ms |
| - | Reference Implementation | 0.480850 | 4.122805 ms |
