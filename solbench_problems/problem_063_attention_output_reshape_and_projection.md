## Description

Fused attention output reshape and projection. Takes multi-head attention output from [batch, num_heads, seq_len, v_head_dim] format, transposes to [batch, seq_len, num_heads, v_head_dim], reshapes to [batch, seq_len, num_heads * v_head_dim] (requires contiguous copy), then projects to [batch, seq_len, hidden_size] via output projection. This is a critical memory bandwidth bottleneck due to non-contiguous transpose, memory reallocation for contiguous, and large intermediate dimension (16384) before projection to hidden_size (7168).
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, num_heads, seq_len, v_head_dim] | bfloat16 |
| o_proj_weight | [hidden_size, intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | num_heads | v_head_dim | hidden_size | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 128 | 7168 | 16384 | 1 | 131 | 0.0871 | 0.0318 |
| 2 | 128 | 128 | 7168 | 16384 | 1 | 293 | 0.1085 | 0.0384 |
| 3 | 128 | 128 | 7168 | 16384 | 1 | 256 | 0.0889 | 0.0336 |
| 4 | 128 | 128 | 7168 | 16384 | 4 | 512 | 0.5112 | 0.2660 |
| 5 | 128 | 128 | 7168 | 16384 | 1 | 997 | 0.2341 | 0.1297 |
| 6 | 128 | 128 | 7168 | 16384 | 4 | 211 | 0.2073 | 0.1098 |
| 7 | 128 | 128 | 7168 | 16384 | 1 | 2048 | 0.4999 | 0.2660 |
| 8 | 128 | 128 | 7168 | 16384 | 4 | 1024 | 0.9483 | 0.5316 |
| 9 | 128 | 128 | 7168 | 16384 | 2 | 2048 | 0.9515 | 0.5316 |
| 10 | 128 | 128 | 7168 | 16384 | 2 | 1024 | 0.5049 | 0.2660 |
| 11 | 128 | 128 | 7168 | 16384 | 2 | 512 | 0.2662 | 0.1332 |
| 12 | 128 | 128 | 7168 | 16384 | 8 | 512 | 0.9466 | 0.5316 |
| 13 | 128 | 128 | 7168 | 16384 | 4 | 128 | 0.1488 | 0.0668 |
| 14 | 128 | 128 | 7168 | 16384 | 4 | 449 | 0.4105 | 0.2333 |
| 15 | 128 | 128 | 7168 | 16384 | 8 | 131 | 0.2527 | 0.1363 |
| 16 | 128 | 128 | 7168 | 16384 | 32 | 128 | 0.9467 | 0.5316 |

```python
import torch

@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    """
    Fused attention output reshape and projection.
    
    Args:
        attn_output: [batch_size, num_heads, seq_len, v_head_dim] - attention output
        o_proj_weight: [hidden_size, intermediate_size] - output projection weight
        
    Returns:
        output: [batch_size, seq_len, hidden_size] - projected output
    """
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    hidden_size = o_proj_weight.shape[0]
    intermediate_size = num_heads * v_head_dim
    
    # Step 1: Transpose [batch, num_heads, seq_len, v_head_dim] -> [batch, seq_len, num_heads, v_head_dim]
    # This creates a non-contiguous tensor with strided memory access
    attn_output_transposed = attn_output.transpose(1, 2)
    
    # Step 2: Reshape to [batch, seq_len, num_heads * v_head_dim]
    # The contiguous() call triggers a full memory copy
    attn_output_reshaped = attn_output_transposed.reshape(bsz, seq_len, intermediate_size)
    
    # Step 3: Output projection via matrix multiplication
    # [batch, seq_len, intermediate_size] @ [intermediate_size, hidden_size] -> [batch, seq_len, hidden_size]
    # Using F.linear equivalent: output = input @ weight.T
    output = torch.matmul(attn_output_reshaped, o_proj_weight.t())
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.163007 ms |
| - | Scoring Baseline | 0.500000 | 0.327444 ms |
| - | Reference Implementation | 0.494416 | 0.332016 ms |
