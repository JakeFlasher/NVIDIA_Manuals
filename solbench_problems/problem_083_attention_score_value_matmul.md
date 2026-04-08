## Description

Fused attention score-value matrix multiplication that computes attention_weights @ V followed by transpose and reshape. This is the second major matmul in attention (after QK^T softmax) and appears in every self-attention and cross-attention layer throughout SDXL UNet.
| Name | Shape | Dtype |
| --- | --- | --- |
| attention_weights | [batch_size, num_heads, seq_len_q, seq_len_kv] | bfloat16 |
| value | [batch_size, num_heads, seq_len_kv, head_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len_q, hidden_size] | bfloat16 |

| # | num_heads | head_dim | hidden_size | batch_size | seq_len_q | seq_len_kv | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 64 | 1280 | 4 | 1024 | 1024 | 0.0756 | 0.0250 |
| 2 | 20 | 64 | 1280 | 2 | 541 | 77 | 0.0312 | 0.0012 |
| 3 | 20 | 64 | 1280 | 4 | 256 | 256 | 0.0295 | 0.0025 |
| 4 | 20 | 64 | 1280 | 2 | 1024 | 1024 | 0.0486 | 0.0127 |
| 5 | 20 | 64 | 1280 | 4 | 1024 | 77 | 0.0516 | 0.0035 |
| 6 | 20 | 64 | 1280 | 2 | 2053 | 2053 | 0.2257 | 0.0471 |
| 7 | 20 | 64 | 1280 | 2 | 293 | 293 | 0.0357 | 0.0017 |
| 8 | 20 | 64 | 1280 | 32 | 256 | 77 | 0.0795 | 0.0072 |
| 9 | 20 | 64 | 1280 | 8 | 256 | 256 | 0.0351 | 0.0045 |
| 10 | 20 | 64 | 1280 | 2 | 256 | 77 | 0.0276 | 0.0008 |
| 11 | 20 | 64 | 1280 | 2 | 4096 | 77 | 0.0791 | 0.0065 |
| 12 | 20 | 64 | 1280 | 2 | 2048 | 77 | 0.0517 | 0.0035 |
| 13 | 20 | 64 | 1280 | 2 | 256 | 256 | 0.0291 | 0.0014 |
| 14 | 20 | 64 | 1280 | 4 | 4096 | 77 | 0.1425 | 0.0126 |
| 15 | 20 | 64 | 1280 | 2 | 1571 | 77 | 0.0482 | 0.0028 |
| 16 | 20 | 64 | 1280 | 1 | 64 | 64 | 0.0290 | 0.0005 |

```python
import torch

@torch.no_grad()
def run(
    attention_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fused attention output: attention_weights @ value with transpose and reshape.
    
    Args:
        attention_weights: Softmax-normalized attention scores [B, H, Q, K]
        value: Value matrix [B, H, K, D]
        
    Returns:
        output: Attention output [B, Q, H*D] ready for output projection
    """
    batch_size = attention_weights.shape[0]
    seq_len_q = attention_weights.shape[2]
    num_heads = 20
    head_dim = 64
    hidden_size = num_heads * head_dim
    
    # Core attention matmul: [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
    attn_output = torch.matmul(attention_weights, value)
    
    # Transpose: [B, H, Q, D] -> [B, Q, H, D]
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # Reshape: [B, Q, H, D] -> [B, Q, H*D]
    attn_output = attn_output.reshape(batch_size, seq_len_q, hidden_size)
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003975 ms |
| - | Scoring Baseline | 0.500000 | 0.051666 ms |
| - | Reference Implementation | 0.468084 | 0.057745 ms |
