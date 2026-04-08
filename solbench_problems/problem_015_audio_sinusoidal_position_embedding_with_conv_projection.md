## Description

Audio encoder's sinusoidal position embedding generation followed by 3-stage Conv2d downsampling (8x total reduction), GELU activations, and linear projection to model dimension. Processes mel-spectrogram through conv layers, adds cached sinusoidal position embeddings, and returns padded output.
| Name | Shape | Dtype |
| --- | --- | --- |
| input_features | [batch_size, one, num_mel_bins, time_dim] | bfloat16 |
| conv2d1_weight | [downsample_hidden_size, one, kernel_size, kernel_size] | bfloat16 |
| conv2d1_bias | [downsample_hidden_size] | bfloat16 |
| conv2d2_weight | [downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size] | bfloat16 |
| conv2d2_bias | [downsample_hidden_size] | bfloat16 |
| conv2d3_weight | [downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size] | bfloat16 |
| conv2d3_bias | [downsample_hidden_size] | bfloat16 |
| conv_out_weight | [d_model, conv_out_dim] | bfloat16 |
| positional_embedding | [max_source_positions, d_model] | bfloat16 |
| embed_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, time_after_conv, d_model] | bfloat16 |

| # | d_model | num_mel_bins | max_source_positions | downsample_hidden_size | freq_dim_after_conv | conv_out_dim | one | kernel_size | batch_size | time_dim | time_after_conv | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 2 | 1688 | 211 | 0.4590 | 0.0334 |
| 2 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 32 | 4328 | 541 | 13.7427 | 1.3546 |
| 3 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 1 | 1048 | 131 | 0.2279 | 0.0106 |
| 4 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 1 | 1808 | 226 | 0.2954 | 0.0181 |
| 5 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 32 | 920 | 115 | 3.0187 | 0.2883 |
| 6 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 64 | 128 | 16 | 0.9148 | 0.0805 |
| 7 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 16 | 2048 | 256 | 3.3497 | 0.3208 |
| 8 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 16 | 384 | 48 | 0.7187 | 0.0605 |
| 9 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 4 | 2216 | 277 | 0.9987 | 0.0871 |
| 10 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 2 | 256 | 32 | 0.1876 | 0.0054 |
| 11 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 2 | 2528 | 316 | 0.6266 | 0.0498 |
| 12 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 8 | 1976 | 247 | 1.6926 | 0.1550 |
| 13 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 8 | 3256 | 407 | 2.6926 | 0.2551 |
| 14 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 32 | 768 | 96 | 2.5415 | 0.2407 |
| 15 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 64 | 512 | 64 | 3.3463 | 0.3208 |
| 16 | 1024 | 80 | 1500 | 384 | 10 | 3840 | 1 | 3 | 1 | 3000 | 375 | 0.4056 | 0.0297 |

```python
import math

import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time_dim = axes_and_scalars["time_dim"]
    d_model = 1024
    max_source_positions = 1500
    downsample_hidden_size = 384
    conv_out_dim = 3840  # 384 * 10
    kernel_size = 3
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return {
        "input_features": torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype),
        # Conv weights — Kaiming init
        "conv2d1_weight": kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size),
        "conv2d1_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d2_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d2_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d3_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d3_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        # Linear projection weight
        "conv_out_weight": xavier(d_model, conv_out_dim),
        # Sinusoidal positional embedding
        "positional_embedding": pe.to(dtype),
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }

@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
    b, c, f, t = x.size()
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    
    # Linear projection to d_model (no bias)
    x = F.linear(x, conv_out_weight)
    
    # Scale embeddings
    x = x * embed_scale
    
    # Add positional embeddings
    seq_len = x.shape[1]
    pos_embed = positional_embedding[:seq_len, :].unsqueeze(0)
    x = x + pos_embed
    
    return x
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.085597 ms |
| - | Scoring Baseline | 0.500000 | 1.108679 ms |
| - | Reference Implementation | 0.496873 | 1.121755 ms |
