## Description

Fused attention output projection (o_proj) with residual addition. After computing attention outputs, this operation performs the final linear projection and adds the residual connection. The projection is a matmul (hidden_size x hidden_size) fused with elementwise residual add to eliminate intermediate memory traffic.
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2560 | 16 | 512 | 0.1047 | 0.0597 |
| 2 | 2560 | 4 | 128 | 0.0271 | 0.0041 |
| 3 | 2560 | 8 | 1024 | 0.1039 | 0.0597 |
| 4 | 2560 | 1 | 1571 | 0.0375 | 0.0118 |
| 5 | 2560 | 4 | 1024 | 0.0616 | 0.0300 |
| 6 | 2560 | 2 | 2053 | 0.0614 | 0.0301 |
| 7 | 2560 | 8 | 997 | 0.1030 | 0.0581 |
| 8 | 2560 | 16 | 256 | 0.0612 | 0.0300 |
| 9 | 2560 | 64 | 128 | 0.1036 | 0.0597 |
| 10 | 2560 | 32 | 256 | 0.1053 | 0.0597 |
| 11 | 2560 | 8 | 512 | 0.0611 | 0.0300 |
| 12 | 2560 | 1 | 1024 | 0.0309 | 0.0078 |
| 13 | 2560 | 16 | 128 | 0.0418 | 0.0152 |
| 14 | 2560 | 2 | 293 | 0.0287 | 0.0046 |
| 15 | 2560 | 1 | 2048 | 0.0416 | 0.0152 |
| 16 | 2560 | 1 | 256 | 0.0239 | 0.0023 |

```python
import torch

@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fused attention output projection with residual addition.
    
    This performs:
    1. Linear projection: attn_output @ o_proj_weight.T
    2. Residual addition: projected + residual
    
    In a custom CUDA kernel, these operations would be fused to:
    - Compute matmul tiles in registers
    - Add residual directly to register-held results
    - Write final result to global memory (eliminating intermediate write)
    
    Args:
        attn_output: Attention output of shape (batch, seq_len, hidden_size)
        residual: Original input before attention of shape (batch, seq_len, hidden_size)
        o_proj_weight: Output projection weight of shape (hidden_size, hidden_size)
    
    Returns:
        Output with residual added, shape (batch, seq_len, hidden_size)
    """
    # Linear projection: (batch, seq_len, hidden_size) @ (hidden_size, hidden_size).T
    # -> (batch, seq_len, hidden_size)
    projected = torch.matmul(attn_output, o_proj_weight.t())
    
    # Residual addition
    output = projected + residual
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.019789 ms |
| - | Scoring Baseline | 0.500000 | 0.054723 ms |
| 1st place | Bob Huang | 0.477493 | 0.058391 ms |
| 2nd place | AKO4ALL_L1 | 0.450425 | 0.060334 ms |
| - | Reference Implementation | 0.415949 | 0.067383 ms |
