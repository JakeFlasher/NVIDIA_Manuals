## Description

ConvNextV2 block with Global Response Normalization (GRN). Combines depthwise 7x7 convolution, LayerNorm, pointwise expansion, GELU activation, GRN layer, pointwise projection, and residual connection. GRN computes L2 norm across spatial dimensions and normalizes by channel-wise mean.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, dim, height, width] | float32 |
| dwconv_weight | [dim, 1, kernel_size, kernel_size] | float32 |
| dwconv_bias | [dim] | float32 |
| layernorm_weight | [dim] | float32 |
| layernorm_bias | [dim] | float32 |
| pwconv1_weight | [expansion_dim, dim] | float32 |
| pwconv1_bias | [expansion_dim] | float32 |
| grn_weight | [1, 1, 1, expansion_dim] | float32 |
| grn_bias | [1, 1, 1, expansion_dim] | float32 |
| pwconv2_weight | [dim, expansion_dim] | float32 |
| pwconv2_bias | [dim] | float32 |
| eps | scalar | float32 |
| layer_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, dim, height, width] | float32 |

| # | kernel_size | padding | batch_size | dim | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7 | 3 | 2 | 96 | 56 | 56 | 0.1433 | 0.0009 |
| 2 | 7 | 3 | 8 | 320 | 32 | 32 | 0.2721 | 0.0080 |
| 3 | 7 | 3 | 8 | 768 | 7 | 7 | 0.2167 | 0.0025 |
| 4 | 7 | 3 | 64 | 96 | 112 | 112 | 4.1371 | 0.0699 |
| 5 | 7 | 3 | 1 | 160 | 449 | 7 | 0.1477 | 0.0011 |
| 6 | 7 | 3 | 16 | 768 | 7 | 7 | 0.1733 | 0.0045 |
| 7 | 7 | 3 | 16 | 384 | 14 | 14 | 0.1976 | 0.0046 |
| 8 | 7 | 3 | 32 | 128 | 64 | 64 | 1.0134 | 0.0203 |
| 9 | 7 | 3 | 4 | 96 | 56 | 56 | 0.1859 | 0.0015 |
| 10 | 7 | 3 | 1 | 96 | 56 | 56 | 0.1436 | 0.0007 |
| 11 | 7 | 3 | 8 | 384 | 14 | 14 | 0.1555 | 0.0025 |
| 12 | 7 | 3 | 4 | 192 | 28 | 28 | 0.1773 | 0.0015 |
| 13 | 7 | 3 | 1 | 192 | 28 | 28 | 0.2875 | 0.0007 |
| 14 | 7 | 3 | 2 | 768 | 7 | 7 | 0.1290 | 0.0017 |
| 15 | 7 | 3 | 32 | 768 | 7 | 7 | 0.2331 | 0.0086 |
| 16 | 7 | 3 | 8 | 192 | 28 | 28 | 0.1948 | 0.0025 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    residual = x
    B, C, H, W = x.shape
    
    # Depthwise convolution: (B, C, H, W) -> (B, C, H, W)
    # groups=C for depthwise
    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=C)
    
    # Permute to channels_last: (B, C, H, W) -> (B, H, W, C)
    out = out.permute(0, 2, 3, 1)
    
    # LayerNorm: (B, H, W, C) -> (B, H, W, C)
    out = F.layer_norm(out, (C,), layernorm_weight, layernorm_bias, eps=layer_norm_eps)
    
    # Pointwise expansion: (B, H, W, C) -> (B, H, W, 4C)
    # Using matmul: out @ pwconv1_weight.T + pwconv1_bias
    out = torch.matmul(out, pwconv1_weight.T) + pwconv1_bias
    
    # GELU activation: (B, H, W, 4C) -> (B, H, W, 4C)
    out = F.gelu(out)
    
    # Global Response Normalization (GRN)
    # Compute L2 norm across spatial dimensions (H, W)
    # Shape: (B, H, W, 4C) -> (B, 1, 1, 4C)
    global_features = torch.linalg.vector_norm(out, ord=2, dim=(1, 2), keepdim=True)
    
    # Normalize by channel-wise mean
    # Shape: (B, 1, 1, 4C) -> (B, 1, 1, 4C)
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)
    
    # Apply GRN transformation with learnable parameters
    # x_grn = weight * (x * norm_features) + bias + x
    out = grn_weight * (out * norm_features) + grn_bias + out
    
    # Pointwise projection: (B, H, W, 4C) -> (B, H, W, C)
    out = torch.matmul(out, pwconv2_weight.T) + pwconv2_bias
    
    # Permute back to channels_first: (B, H, W, C) -> (B, C, H, W)
    out = out.permute(0, 3, 1, 2)
    
    # Residual connection (no drop path in inference)
    out = residual + out
    
    return out
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003058 ms |
| - | Scoring Baseline | 0.500000 | 0.249265 ms |
| - | Reference Implementation | 0.393168 | 0.385148 ms |
