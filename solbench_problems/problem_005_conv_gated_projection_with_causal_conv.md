## Description

Complete fused convolution layer operation that combines triple linear projection, element-wise gating, causal 1D grouped convolution, gating with output channel, and final output projection. This is the dominant operation in non-attention layers in LFM2 architecture.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, seq_len, hidden_size] | bfloat16 |
| in_proj_weight | [triple_hidden, hidden_size] | bfloat16 |
| in_proj_bias | [triple_hidden] | bfloat16 |
| conv_weight | [hidden_size, 1, conv_kernel_size] | bfloat16 |
| conv_bias | [hidden_size] | bfloat16 |
| out_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| out_proj_bias | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | conv_kernel_size | triple_hidden | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 4 | 6144 | 2 | 4096 | 0.5172 | 0.1522 |
| 2 | 2048 | 4 | 6144 | 2 | 1879 | 0.2716 | 0.0701 |
| 3 | 2048 | 4 | 6144 | 16 | 256 | 0.3723 | 0.0763 |
| 4 | 2048 | 4 | 6144 | 2 | 2048 | 0.2956 | 0.0763 |
| 5 | 2048 | 4 | 6144 | 1 | 2048 | 0.1763 | 0.0384 |
| 6 | 2048 | 4 | 6144 | 1 | 256 | 0.0748 | 0.0051 |
| 7 | 2048 | 4 | 6144 | 32 | 256 | 0.6712 | 0.1522 |
| 8 | 2048 | 4 | 6144 | 1 | 4096 | 0.2827 | 0.0763 |
| 9 | 2048 | 4 | 6144 | 4 | 512 | 0.1828 | 0.0384 |
| 10 | 2048 | 4 | 6144 | 2 | 128 | 0.0748 | 0.0051 |
| 11 | 2048 | 4 | 6144 | 8 | 997 | 0.5080 | 0.1482 |
| 12 | 2048 | 4 | 6144 | 1 | 8192 | 0.4939 | 0.1522 |
| 13 | 2048 | 4 | 6144 | 1 | 1024 | 0.1599 | 0.0194 |
| 14 | 2048 | 4 | 6144 | 4 | 541 | 0.1902 | 0.0405 |
| 15 | 2048 | 4 | 6144 | 2 | 1024 | 0.1800 | 0.0384 |
| 16 | 2048 | 4 | 6144 | 4 | 256 | 0.1456 | 0.0194 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    """
    Complete fused convolution layer with gated projections and causal convolution.
    
    Operation flow:
    1. Triple linear projection: x -> (B, C, x_proj) via in_proj
    2. Element-wise gating: Bx = B * x_proj
    3. Grouped causal 1D convolution on Bx with kernel_size=4
    4. Output gating: y = C * conv_out
    5. Final output projection: y -> out_proj(y)
    """
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]
    
    # Step 1: Triple linear projection
    # Shape: (batch_size, seq_len, 3 * hidden_size)
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    
    # Transpose for conv1d: (batch_size, 3 * hidden_size, seq_len)
    BCx = BCx.transpose(-1, -2)
    
    # Split into B, C, x_proj along channel dimension
    # Each has shape: (batch_size, hidden_size, seq_len)
    B, C, x_proj = BCx.chunk(3, dim=1)
    
    # Step 2: Element-wise gating
    # Shape: (batch_size, hidden_size, seq_len)
    Bx = B * x_proj
    
    # Step 3: Grouped causal 1D convolution
    # Apply conv with causal padding
    # Padding is kernel_size - 1 on the left for causal
    Bx_padded = F.pad(Bx, (conv_kernel_size - 1, 0))
    
    # Grouped conv1d with groups=hidden_size (depthwise)
    conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=hidden_size)
    
    # Shape: (batch_size, hidden_size, seq_len)
    
    # Step 4: Output gating with C
    # Shape: (batch_size, hidden_size, seq_len)
    y = C * conv_out
    
    # Transpose back: (batch_size, seq_len, hidden_size)
    y = y.transpose(-1, -2).contiguous()
    
    # Step 5: Final output projection
    # Shape: (batch_size, seq_len, hidden_size)
    output = F.linear(y, out_proj_weight, out_proj_bias)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.045726 ms |
| 1st place | AKO4ALL_L1 | 0.637793 | 0.158345 ms |
| - | Scoring Baseline | 0.500000 | 0.237965 ms |
| - | Reference Implementation | 0.447830 | 0.281826 ms |
| 2nd place | Quick Impala | 0.392238 | 0.339645 ms |
