## Description

Fused attention weights @ value matmul with transpose and reshape. Performs attn_weights @ value_states followed by transpose from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, num_heads, head_dim) and reshape to (batch, seq_len, hidden_size). This is the second major matmul in attention that combines attention weights with value states, appearing 64 times per forward pass.
| Name | Shape | Dtype |
| --- | --- | --- |
| attn_weights | [batch_size, num_attention_heads, seq_len, seq_len] | bfloat16 |
| value_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | num_attention_heads | head_dim | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 128 | 5120 | 32 | 128 | 0.0825 | 0.0168 |
| 2 | 40 | 128 | 5120 | 16 | 256 | 0.0821 | 0.0223 |
| 3 | 40 | 128 | 5120 | 1 | 256 | 0.0242 | 0.0018 |
| 4 | 40 | 128 | 5120 | 64 | 128 | 0.0960 | 0.0332 |
| 5 | 40 | 128 | 5120 | 4 | 512 | 0.0730 | 0.0168 |
| 6 | 40 | 128 | 5120 | 2 | 541 | 0.1161 | 0.0094 |
| 7 | 40 | 128 | 5120 | 1 | 293 | 0.0363 | 0.0021 |
| 8 | 40 | 128 | 5120 | 1 | 131 | 0.0248 | 0.0009 |
| 9 | 40 | 128 | 5120 | 1 | 853 | 0.0901 | 0.0103 |
| 10 | 40 | 128 | 5120 | 8 | 512 | 0.0893 | 0.0332 |
| 11 | 40 | 128 | 5120 | 8 | 128 | 0.0421 | 0.0045 |
| 12 | 40 | 128 | 5120 | 2 | 1024 | 0.0822 | 0.0278 |
| 13 | 40 | 128 | 5120 | 8 | 256 | 0.0644 | 0.0113 |
| 14 | 40 | 128 | 5120 | 1 | 2048 | 0.0905 | 0.0496 |
| 15 | 40 | 128 | 5120 | 4 | 691 | 0.2790 | 0.0277 |
| 16 | 40 | 128 | 5120 | 1 | 997 | 0.1708 | 0.0134 |

```python
import torch

@torch.no_grad()
def run(
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """
    Fused attention weights @ value matmul with transpose and reshape.
    
    Args:
        attn_weights: Attention weights after softmax, shape (batch, 40, seq_len, seq_len)
        value_states: Value states after GQA expansion, shape (batch, 40, seq_len, 128)
        
    Returns:
        Reshaped attention output ready for output projection, shape (batch, seq_len, 5120)
    """
    # Constants
    num_attention_heads = 40
    head_dim = 128
    hidden_size = 5120
    
    # Step 1: Attention weights @ value states
    # (batch, 40, seq_len, seq_len) @ (batch, 40, seq_len, 128) -> (batch, 40, seq_len, 128)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Step 2: Transpose to move heads dimension
    # (batch, 40, seq_len, 128) -> (batch, seq_len, 40, 128)
    attn_output = attn_output.transpose(1, 2)
    
    # Step 3: Reshape to combine heads
    # (batch, seq_len, 40, 128) -> (batch, seq_len, 5120)
    batch_size, seq_len = attn_output.shape[0], attn_output.shape[1]
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # Ensure contiguous for next operation
    attn_output = attn_output.contiguous()
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.011179 ms |
| - | Scoring Baseline | 0.500000 | 0.074654 ms |
| - | Reference Implementation | 0.416358 | 0.099976 ms |
