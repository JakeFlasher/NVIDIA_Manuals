## Description

Fused QKV projection with Grouped Query Attention (GQA) reshaping. Performs three linear projections (Q: 2048->2048, K: 2048->512, V: 2048->512) followed by reshape and transpose to prepare tensors for attention computation. Uses GQA pattern with 16 query heads and 4 key-value heads (4x compression).
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_weight | [q_size, hidden_size] | bfloat16 |
| k_weight | [kv_size, hidden_size] | bfloat16 |
| v_weight | [kv_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |

| # | hidden_size | num_heads | num_kv_heads | head_dim | q_size | kv_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 16 | 4 | 128 | 2048 | 512 | 4 | 128 | 0.0392 | 0.0040 |
| 2 | 2048 | 16 | 4 | 128 | 2048 | 512 | 2 | 211 | 0.0352 | 0.0033 |
| 3 | 2048 | 16 | 4 | 128 | 2048 | 512 | 32 | 128 | 0.0708 | 0.0289 |
| 4 | 2048 | 16 | 4 | 128 | 2048 | 512 | 1 | 449 | 0.0357 | 0.0035 |
| 5 | 2048 | 16 | 4 | 128 | 2048 | 512 | 16 | 512 | 0.1064 | 0.0573 |
| 6 | 2048 | 16 | 4 | 128 | 2048 | 512 | 1 | 131 | 0.0324 | 0.0021 |
| 7 | 2048 | 16 | 4 | 128 | 2048 | 512 | 1 | 4096 | 0.0697 | 0.0289 |
| 8 | 2048 | 16 | 4 | 128 | 2048 | 512 | 16 | 128 | 0.0524 | 0.0146 |
| 9 | 2048 | 16 | 4 | 128 | 2048 | 512 | 1 | 512 | 0.0366 | 0.0040 |
| 10 | 2048 | 16 | 4 | 128 | 2048 | 512 | 2 | 512 | 0.0415 | 0.0075 |
| 11 | 2048 | 16 | 4 | 128 | 2048 | 512 | 1 | 1024 | 0.0413 | 0.0075 |
| 12 | 2048 | 16 | 4 | 128 | 2048 | 512 | 4 | 1024 | 0.0702 | 0.0289 |
| 13 | 2048 | 16 | 4 | 128 | 2048 | 512 | 2 | 2048 | 0.0703 | 0.0289 |
| 14 | 2048 | 16 | 4 | 128 | 2048 | 512 | 1 | 2048 | 0.0522 | 0.0146 |
| 15 | 2048 | 16 | 4 | 128 | 2048 | 512 | 64 | 128 | 0.1072 | 0.0573 |
| 16 | 2048 | 16 | 4 | 128 | 2048 | 512 | 2 | 541 | 0.0413 | 0.0079 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
):
    """
    Fused QKV projection with GQA reshaping.
    
    Args:
        hidden_states: [batch_size, seq_len, 2048]
        q_weight: [2048, 2048] - Query projection weight
        k_weight: [512, 2048] - Key projection weight
        v_weight: [512, 2048] - Value projection weight
    
    Returns:
        query_states: [batch_size, 16, seq_len, 128]
        key_states: [batch_size, 4, seq_len, 128]
        value_states: [batch_size, 4, seq_len, 128]
    """
    # Constants
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    
    bsz, q_len, _ = hidden_states.size()
    
    # Three linear projections (no bias)
    # Q: [bsz, q_len, 2048] @ [2048, 2048].T -> [bsz, q_len, 2048]
    query_states = torch.matmul(hidden_states, q_weight.t())
    
    # K: [bsz, q_len, 2048] @ [512, 2048].T -> [bsz, q_len, 512]
    key_states = torch.matmul(hidden_states, k_weight.t())
    
    # V: [bsz, q_len, 2048] @ [512, 2048].T -> [bsz, q_len, 512]
    value_states = torch.matmul(hidden_states, v_weight.t())
    
    # Reshape and transpose for attention computation
    # Query: [bsz, q_len, 2048] -> [bsz, q_len, 16, 128] -> [bsz, 16, q_len, 128]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    
    # Key: [bsz, q_len, 512] -> [bsz, q_len, 4, 128] -> [bsz, 4, q_len, 128]
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Value: [bsz, q_len, 512] -> [bsz, q_len, 4, 128] -> [bsz, 4, q_len, 128]
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    return query_states, key_states, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.011424 ms |
| - | Scoring Baseline | 0.500000 | 0.052308 ms |
| - | Reference Implementation | 0.408703 | 0.070999 ms |
