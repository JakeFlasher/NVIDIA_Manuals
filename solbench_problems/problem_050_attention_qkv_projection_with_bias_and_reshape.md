## Description

Fused QKV projection with bias and reshape for Gemma3 attention. Performs three separate linear projections (Q, K, V) with bias addition, then reshapes outputs from (batch, seq_len, hidden_size) to (batch, seq_len, num_heads, head_dim) format for multi-head attention.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_weight | [q_out_dim, hidden_size] | bfloat16 |
| q_bias | [q_out_dim] | bfloat16 |
| k_weight | [kv_out_dim, hidden_size] | bfloat16 |
| k_bias | [kv_out_dim] | bfloat16 |
| v_weight | [kv_out_dim, hidden_size] | bfloat16 |
| v_bias | [kv_out_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| query_states | [batch_size, seq_len, num_attention_heads, head_dim] | bfloat16 |
| key_states | [batch_size, seq_len, num_key_value_heads, head_dim] | bfloat16 |
| value_states | [batch_size, seq_len, num_key_value_heads, head_dim] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_out_dim | kv_out_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 640 | 4 | 1 | 256 | 1024 | 256 | 32 | 128 | 0.0481 | 0.0048 |
| 2 | 640 | 4 | 1 | 256 | 1024 | 256 | 8 | 128 | 0.0445 | 0.0015 |
| 3 | 640 | 4 | 1 | 256 | 1024 | 256 | 1 | 1571 | 0.0422 | 0.0021 |
| 4 | 640 | 4 | 1 | 256 | 1024 | 256 | 2 | 128 | 0.0423 | 0.0008 |
| 5 | 640 | 4 | 1 | 256 | 1024 | 256 | 1 | 512 | 0.0512 | 0.0010 |
| 6 | 640 | 4 | 1 | 256 | 1024 | 256 | 4 | 512 | 0.0398 | 0.0026 |
| 7 | 640 | 4 | 1 | 256 | 1024 | 256 | 4 | 1024 | 0.0417 | 0.0048 |
| 8 | 640 | 4 | 1 | 256 | 1024 | 256 | 2 | 512 | 0.0407 | 0.0015 |
| 9 | 640 | 4 | 1 | 256 | 1024 | 256 | 2 | 1024 | 0.0396 | 0.0026 |
| 10 | 640 | 4 | 1 | 256 | 1024 | 256 | 4 | 256 | 0.0398 | 0.0015 |
| 11 | 640 | 4 | 1 | 256 | 1024 | 256 | 1 | 4096 | 0.0405 | 0.0048 |
| 12 | 640 | 4 | 1 | 256 | 1024 | 256 | 4 | 2048 | 0.0475 | 0.0093 |
| 13 | 640 | 4 | 1 | 256 | 1024 | 256 | 4 | 541 | 0.0409 | 0.0027 |
| 14 | 640 | 4 | 1 | 256 | 1024 | 256 | 1 | 131 | 0.0395 | 0.0007 |
| 15 | 640 | 4 | 1 | 256 | 1024 | 256 | 4 | 128 | 0.0401 | 0.0010 |
| 16 | 640 | 4 | 1 | 256 | 1024 | 256 | 64 | 128 | 0.0577 | 0.0093 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
):
    """
    Fused QKV projection with bias and reshape for Gemma3 attention.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, 640)
        q_weight: Query projection weight (1024, 640)
        q_bias: Query projection bias (1024,)
        k_weight: Key projection weight (256, 640)
        k_bias: Key projection bias (256,)
        v_weight: Value projection weight (256, 640)
        v_bias: Value projection bias (256,)
        
    Returns:
        Tuple of (query_states, key_states, value_states) where:
            query_states: (batch_size, seq_len, 16, 128)
            key_states: (batch_size, seq_len, 2, 128)
            value_states: (batch_size, seq_len, 2, 128)
    """
    batch_size, seq_len, _ = hidden_states.shape
    
    # Constants
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256
    
    # Q projection: (batch, seq, 640) @ (1024, 640).T + bias -> (batch, seq, 1024)
    query_states = torch.matmul(hidden_states, q_weight.t()) + q_bias
    
    # K projection: (batch, seq, 640) @ (256, 640).T + bias -> (batch, seq, 256)
    key_states = torch.matmul(hidden_states, k_weight.t()) + k_bias
    
    # V projection: (batch, seq, 640) @ (256, 640).T + bias -> (batch, seq, 256)
    value_states = torch.matmul(hidden_states, v_weight.t()) + v_bias
    
    # Reshape for multi-head attention
    # Query: (batch, seq_len, 1024) -> (batch, seq_len, 4, 256)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    
    # Key: (batch, seq_len, 256) -> (batch, seq_len, 1, 256)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Value: (batch, seq_len, 256) -> (batch, seq_len, 1, 256)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    return query_states, key_states, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002318 ms |
| - | Scoring Baseline | 0.500000 | 0.043245 ms |
| - | Reference Implementation | 0.358091 | 0.075449 ms |
