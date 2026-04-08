## Description

Fused attention value projection with reshape and transpose for GQA attention. Projects hidden states from hidden_size=5120 to num_kv_heads*head_dim=1024, then reshapes and transposes to [batch, num_kv_heads, seq_len, head_dim] format for attention computation.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| v_proj_weight | [kv_projection_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| value_states | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |

| # | hidden_size | num_kv_heads | head_dim | kv_projection_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 8 | 128 | 1024 | 2 | 128 | 0.0275 | 0.0022 |
| 2 | 5120 | 8 | 128 | 1024 | 4 | 541 | 0.0399 | 0.0129 |
| 3 | 5120 | 8 | 128 | 1024 | 16 | 128 | 0.0380 | 0.0123 |
| 4 | 5120 | 8 | 128 | 1024 | 4 | 512 | 0.0376 | 0.0123 |
| 5 | 5120 | 8 | 128 | 1024 | 4 | 256 | 0.0312 | 0.0063 |
| 6 | 5120 | 8 | 128 | 1024 | 32 | 128 | 0.0553 | 0.0241 |
| 7 | 5120 | 8 | 128 | 1024 | 8 | 128 | 0.0317 | 0.0063 |
| 8 | 5120 | 8 | 128 | 1024 | 1 | 256 | 0.0269 | 0.0022 |
| 9 | 5120 | 8 | 128 | 1024 | 64 | 128 | 0.0875 | 0.0478 |
| 10 | 5120 | 8 | 128 | 1024 | 1 | 8192 | 0.0880 | 0.0478 |
| 11 | 5120 | 8 | 128 | 1024 | 1 | 512 | 0.0273 | 0.0034 |
| 12 | 5120 | 8 | 128 | 1024 | 1 | 128 | 0.0285 | 0.0020 |
| 13 | 5120 | 8 | 128 | 1024 | 16 | 256 | 0.0554 | 0.0241 |
| 14 | 5120 | 8 | 128 | 1024 | 1 | 1571 | 0.0356 | 0.0095 |
| 15 | 5120 | 8 | 128 | 1024 | 8 | 1024 | 0.0878 | 0.0478 |
| 16 | 5120 | 8 | 128 | 1024 | 1 | 2048 | 0.0376 | 0.0123 |

```python
import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    """
    Fused value projection with reshape and transpose for GQA attention.
    
    Args:
        hidden_states: Input tensor of shape [batch_size, seq_len, 5120]
        v_proj_weight: Weight matrix of shape [1024, 5120]
        
    Returns:
        value_states: Reshaped and transposed tensor of shape [batch_size, 8, seq_len, 128]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_kv_heads = 8
    head_dim = 128
    
    # Project to value space: [batch, seq_len, 5120] @ [5120, 1024] -> [batch, seq_len, 1024]
    # F.linear computes input @ weight.T, so with weight [1024, 5120], we get [batch, seq_len, 1024]
    value_proj = torch.nn.functional.linear(hidden_states, v_proj_weight)
    
    # Reshape to separate heads: [batch, seq_len, 1024] -> [batch, seq_len, 8, 128]
    value_states = value_proj.view(batch_size, seq_len, num_kv_heads, head_dim)
    
    # Transpose to attention format: [batch, seq_len, 8, 128] -> [batch, 8, seq_len, 128]
    value_states = value_states.transpose(1, 2).contiguous()
    
    return value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.010406 ms |
| - | Scoring Baseline | 0.500000 | 0.041890 ms |
| - | Reference Implementation | 0.406913 | 0.054543 ms |
| 1st place | Clever Scorpion | 0.358789 | 0.063670 ms |
