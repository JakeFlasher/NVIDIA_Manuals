## Description

VAE self-attention block with group normalization. Applies group norm, flattens spatial dimensions (B,C,H,W) to (B,H*W,C), computes single-head self-attention over spatial positions, reshapes back, and adds residual. Critical for capturing long-range spatial dependencies in VAE encoder/decoder.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, channels, spatial_size, spatial_size] | float32 |
| group_norm_weight | [channels] | float32 |
| group_norm_bias | [channels] | float32 |
| query_weight | [channels, channels] | float32 |
| query_bias | [channels] | float32 |
| key_weight | [channels, channels] | float32 |
| key_bias | [channels] | float32 |
| value_weight | [channels, channels] | float32 |
| value_bias | [channels] | float32 |
| proj_out_weight | [channels, channels] | float32 |
| proj_out_bias | [channels] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, channels, spatial_size, spatial_size] | float32 |

| # | channels | num_groups | spatial_size | seq_len | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 32 | 64 | 4096 | 26 | 6.8930 | 0.6169 |
| 2 | 512 | 32 | 64 | 4096 | 20 | 5.2868 | 0.4747 |
| 3 | 512 | 32 | 64 | 4096 | 10 | 2.7730 | 0.2375 |
| 4 | 512 | 32 | 64 | 4096 | 25 | 6.3654 | 0.5932 |
| 5 | 512 | 32 | 64 | 4096 | 24 | 6.0994 | 0.5695 |
| 6 | 512 | 32 | 64 | 4096 | 18 | 4.6854 | 0.4272 |
| 7 | 512 | 32 | 64 | 4096 | 2 | 0.8055 | 0.0478 |
| 8 | 512 | 32 | 64 | 4096 | 13 | 3.5597 | 0.3087 |
| 9 | 512 | 32 | 64 | 4096 | 17 | 4.4263 | 0.4035 |
| 10 | 512 | 32 | 64 | 4096 | 11 | 3.0203 | 0.2612 |
| 11 | 512 | 32 | 64 | 4096 | 3 | 1.0546 | 0.0715 |
| 12 | 512 | 32 | 64 | 4096 | 6 | 1.8895 | 0.1427 |
| 13 | 512 | 32 | 64 | 4096 | 8 | 2.1949 | 0.1901 |
| 14 | 512 | 32 | 64 | 4096 | 14 | 3.8507 | 0.3324 |
| 15 | 512 | 32 | 64 | 4096 | 16 | 4.1422 | 0.3798 |
| 16 | 512 | 32 | 64 | 4096 | 12 | 3.2895 | 0.2850 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = x.shape
    num_groups = 32
    
    # Store residual
    residual = x
    
    # Group normalization
    # Reshape to (B, num_groups, channels_per_group, H, W)
    channels_per_group = channels // num_groups
    x_grouped = x.view(batch, num_groups, channels_per_group, height, width)
    
    # Compute mean and variance per group
    mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    
    # Normalize
    x_norm = (x_grouped - mean) / torch.sqrt(var + eps)
    
    # Reshape back to (B, C, H, W)
    x_norm = x_norm.view(batch, channels, height, width)
    
    # Apply scale and shift
    x_norm = x_norm * group_norm_weight.view(1, channels, 1, 1) + group_norm_bias.view(1, channels, 1, 1)
    
    # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
    seq_len = height * width
    x_seq = x_norm.view(batch, channels, seq_len)
    x_seq = x_seq.permute(0, 2, 1).contiguous()  # (B, H*W, C)
    
    # Compute Q, K, V projections using linear layers
    # q = x_seq @ query_weight.T + query_bias
    q = torch.matmul(x_seq, query_weight.t()) + query_bias  # (B, H*W, C)
    k = torch.matmul(x_seq, key_weight.t()) + key_bias      # (B, H*W, C)
    v = torch.matmul(x_seq, value_weight.t()) + value_bias  # (B, H*W, C)
    
    # Compute attention scores: Q @ K^T with scaling
    scale = channels ** -0.5
    attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, H*W, H*W)
    
    # Softmax over key dimension
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values
    attn_output = torch.bmm(attn_weights, v)  # (B, H*W, C)
    
    # Output projection
    attn_output = torch.matmul(attn_output, proj_out_weight.t()) + proj_out_bias
    
    # Reshape back to spatial format: (B, H*W, C) -> (B, C, H, W)
    attn_output = attn_output.permute(0, 2, 1).contiguous()
    attn_output = attn_output.view(batch, channels, height, width)
    
    # Residual connection
    output = residual + attn_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.275311 ms |
| - | Scoring Baseline | 0.500000 | 3.258725 ms |
| - | Reference Implementation | 0.188003 | 13.190439 ms |
