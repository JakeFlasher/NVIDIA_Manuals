## Description

Fused MLP pathway for ConvNextV2: expansion linear (dim -> 4*dim), GELU activation, Global Response Normalization (GRN) with L2 norm and channel-wise normalization, and projection linear (4*dim -> dim). Input/output in NHWC format.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, height, width, dim] | float32 |
| pwconv1_weight | [hidden_dim, dim] | float32 |
| pwconv1_bias | [hidden_dim] | float32 |
| grn_weight | [1, 1, 1, hidden_dim] | float32 |
| grn_bias | [1, 1, 1, hidden_dim] | float32 |
| pwconv2_weight | [dim, hidden_dim] | float32 |
| pwconv2_bias | [dim] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, height, width, dim] | float32 |

| # | dim | hidden_dim | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 512 | 4 | 28 | 28 | 0.1061 | 0.0014 |
| 2 | 128 | 512 | 16 | 14 | 14 | 0.1174 | 0.0014 |
| 3 | 128 | 512 | 64 | 14 | 14 | 0.1542 | 0.0046 |
| 4 | 128 | 512 | 2 | 56 | 56 | 0.1527 | 0.0025 |
| 5 | 128 | 512 | 64 | 56 | 56 | 0.5889 | 0.0674 |
| 6 | 128 | 512 | 32 | 28 | 28 | 0.1901 | 0.0088 |
| 7 | 128 | 512 | 64 | 28 | 28 | 0.1926 | 0.0172 |
| 8 | 128 | 512 | 4 | 56 | 56 | 0.1720 | 0.0046 |
| 9 | 128 | 512 | 8 | 14 | 14 | 0.1074 | 0.0009 |
| 10 | 128 | 512 | 32 | 14 | 14 | 0.1282 | 0.0025 |
| 11 | 128 | 512 | 16 | 28 | 28 | 0.1507 | 0.0046 |
| 12 | 128 | 512 | 32 | 56 | 56 | 0.3303 | 0.0339 |
| 13 | 128 | 512 | 1 | 28 | 28 | 0.1086 | 0.0007 |
| 14 | 128 | 512 | 16 | 56 | 56 | 0.1985 | 0.0172 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
):
    # Expansion linear: (B, H, W, dim) -> (B, H, W, hidden_dim)
    # F.linear computes x @ weight.T + bias
    x = F.linear(hidden_states, pwconv1_weight, pwconv1_bias)
    
    # GELU activation
    x = F.gelu(x)
    
    # Global Response Normalization (GRN)
    # Compute L2 norm across spatial dimensions (H, W)
    # Shape: (B, H, W, hidden_dim) -> (B, 1, 1, hidden_dim)
    global_features = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    
    # Normalize by channel-wise mean: (B, 1, 1, hidden_dim) -> (B, 1, 1, hidden_dim)
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)
    
    # Apply learnable affine transformation with residual connection
    # weight * (input * norm_features) + bias + input
    x = grn_weight * (x * norm_features) + grn_bias + x
    
    # Projection linear: (B, H, W, hidden_dim) -> (B, H, W, dim)
    output = F.linear(x, pwconv2_weight, pwconv2_bias)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.004890 ms |
| - | Scoring Baseline | 0.500000 | 0.169426 ms |
| - | Reference Implementation | 0.386094 | 0.268297 ms |
