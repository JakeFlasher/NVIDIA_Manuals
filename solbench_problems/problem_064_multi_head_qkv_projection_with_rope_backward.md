## Description

Backward pass for multi-head QKV projection with RoPE. Computes gradients for hidden_states, q_weight, k_weight, and v_weight given gradients of query, key, and value states.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_query | [batch_size, num_heads, seq_len, head_dim] | float32 |
| grad_key | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |
| grad_value | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| q_weight | [q_out_dim, hidden_size] | float32 |
| k_weight | [kv_out_dim, hidden_size] | float32 |
| v_weight | [kv_out_dim, hidden_size] | float32 |
| cos | [batch_size, seq_len, head_dim] | float32 |
| sin | [batch_size, seq_len, head_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| grad_q_weight | [q_out_dim, hidden_size] | float32 |
| grad_k_weight | [kv_out_dim, hidden_size] | float32 |
| grad_v_weight | [kv_out_dim, hidden_size] | float32 |

| # | hidden_size | num_heads | num_kv_heads | head_dim | q_out_dim | kv_out_dim | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 8 | 256 | 0.1825 | 0.0217 |
| 2 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 1 | 2048 | 0.2040 | 0.0217 |
| 3 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 4 | 2048 | 0.3407 | 0.0858 |
| 4 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 2 | 128 | 0.1518 | 0.0031 |
| 5 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 16 | 128 | 0.2082 | 0.0217 |
| 6 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 8 | 512 | 0.2649 | 0.0431 |
| 7 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 64 | 128 | 0.3512 | 0.0858 |
| 8 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 2 | 293 | 0.1499 | 0.0065 |
| 9 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 4 | 256 | 0.1567 | 0.0111 |
| 10 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 1 | 8192 | 0.3515 | 0.0858 |
| 11 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 4 | 541 | 0.1765 | 0.0230 |
| 12 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 1 | 4096 | 0.2305 | 0.0431 |
| 13 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 4 | 1024 | 0.2290 | 0.0431 |
| 14 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 2 | 1321 | 0.1896 | 0.0279 |
| 15 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 32 | 128 | 0.2456 | 0.0431 |
| 16 | 1536 | 24 | 4 | 96 | 2304 | 384 | 48 | 2 | 1024 | 0.1663 | 0.0217 |

```python
import torch

@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 24
    num_kv_heads = 4
    head_dim = 96
    half_head_dim = head_dim // 2
    
    # Expand cos/sin for broadcasting: (batch, 1, seq_len, head_dim)
    cos_unsqueezed = cos.unsqueeze(1)
    sin_unsqueezed = sin.unsqueeze(1)
    
    # Backward through RoPE for Q
    # Forward: q_rope = q * cos + rotate_half(q) * sin
    # where rotate_half([x1, x2]) = [-x2, x1]
    # Backward: grad_q = grad_q_rope * cos + rotate_half_inverse(grad_q_rope * sin)
    # where rotate_half_inverse([y1, y2]) = [y2, -y1]
    grad_q_cos_term = grad_query * cos_unsqueezed
    grad_q_sin_term = grad_query * sin_unsqueezed
    
    # Reverse rotate_half for sin term: [y1, y2] -> [y2, -y1]
    grad_q_sin_1 = grad_q_sin_term[..., :half_head_dim]
    grad_q_sin_2 = grad_q_sin_term[..., half_head_dim:]
    grad_q_sin_reversed = torch.cat((grad_q_sin_2, -grad_q_sin_1), dim=-1)
    
    grad_query_unrotated = grad_q_cos_term + grad_q_sin_reversed
    
    # Backward through RoPE for K
    grad_k_cos_term = grad_key * cos_unsqueezed
    grad_k_sin_term = grad_key * sin_unsqueezed
    
    grad_k_sin_1 = grad_k_sin_term[..., :half_head_dim]
    grad_k_sin_2 = grad_k_sin_term[..., half_head_dim:]
    grad_k_sin_reversed = torch.cat((grad_k_sin_2, -grad_k_sin_1), dim=-1)
    
    grad_key_unrotated = grad_k_cos_term + grad_k_sin_reversed
    
    # Backward through transpose and reshape for Q
    # Forward: (B, S, N*D) -> view -> (B, S, N, D) -> transpose -> (B, N, S, D)
    # Backward: (B, N, S, D) -> transpose -> (B, S, N, D) -> view -> (B, S, N*D)
    grad_query_transposed = grad_query_unrotated.transpose(1, 2)
    grad_query_proj = grad_query_transposed.contiguous().view(
        batch_size, seq_len, num_heads * head_dim
    )
    
    # Backward through transpose and reshape for K
    grad_key_transposed = grad_key_unrotated.transpose(1, 2)
    grad_key_proj = grad_key_transposed.contiguous().view(
        batch_size, seq_len, num_kv_heads * head_dim
    )
    
    # Backward through transpose and reshape for V (no RoPE)
    grad_value_transposed = grad_value.transpose(1, 2)
    grad_value_proj = grad_value_transposed.contiguous().view(
        batch_size, seq_len, num_kv_heads * head_dim
    )
    
    # Backward through linear projections
    # Forward: output = input @ weight.T
    # Backward: grad_input = grad_output @ weight
    #           grad_weight = grad_output.T @ input
    
    # Gradient w.r.t. hidden_states (sum contributions from Q, K, V)
    grad_hidden_from_q = torch.matmul(grad_query_proj, q_weight)
    grad_hidden_from_k = torch.matmul(grad_key_proj, k_weight)
    grad_hidden_from_v = torch.matmul(grad_value_proj, v_weight)
    grad_hidden_states = grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v
    
    # Gradient w.r.t. q_weight
    grad_query_proj_2d = grad_query_proj.view(-1, num_heads * head_dim)
    hidden_states_2d = hidden_states.view(-1, hidden_size)
    grad_q_weight = torch.matmul(grad_query_proj_2d.t(), hidden_states_2d)
    
    # Gradient w.r.t. k_weight
    grad_key_proj_2d = grad_key_proj.view(-1, num_kv_heads * head_dim)
    grad_k_weight = torch.matmul(grad_key_proj_2d.t(), hidden_states_2d)
    
    # Gradient w.r.t. v_weight
    grad_value_proj_2d = grad_value_proj.view(-1, num_kv_heads * head_dim)
    grad_v_weight = torch.matmul(grad_value_proj_2d.t(), hidden_states_2d)
    
    return grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.026765 ms |
| - | Scoring Baseline | 0.500000 | 0.215849 ms |
| - | Reference Implementation | 0.120952 | 1.534223 ms |
