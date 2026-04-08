## Description

Flash Attention for Grouped Query Attention with ultra-long context. Fuses QKV projection, RoPE application, GQA KV repetition, scaled dot-product attention with causal masking, and output projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| cos | [batch_size, seq_len, head_dim] | float32 |
| sin | [batch_size, seq_len, head_dim] | float32 |
| q_proj_weight | [q_proj_out, hidden_size] | float32 |
| k_proj_weight | [kv_proj_out, hidden_size] | float32 |
| v_proj_weight | [kv_proj_out, hidden_size] | float32 |
| o_proj_weight | [hidden_size, q_proj_out] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | float32 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | q_proj_out | kv_proj_out | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 128 | 0.2445 | 0.0126 |
| 2 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 512 | 0.2725 | 0.0526 |
| 3 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 2053 | 0.4055 | 0.1336 |
| 4 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 541 | 0.3114 | 0.0558 |
| 5 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 293 | 0.2478 | 0.0147 |
| 6 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 8 | 512 | 0.5355 | 0.2091 |
| 7 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 1024 | 0.3800 | 0.1142 |
| 8 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 512 | 0.2461 | 0.0265 |
| 9 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 64 | 128 | 0.8375 | 0.3893 |
| 10 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 997 | 0.2831 | 0.0556 |
| 11 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 256 | 0.2475 | 0.0253 |
| 12 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 128 | 0.2536 | 0.0075 |
| 13 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 256 | 0.2457 | 0.0128 |
| 14 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 16 | 256 | 0.5243 | 0.1996 |
| 15 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 2048 | 0.5926 | 0.2660 |
| 16 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 131 | 0.2476 | 0.0075 |
| 17 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 2 | 8192 | 2.4490 | 1.9733 |
| 18 | 4096 | 32 | 8 | 128 | 4 | 4096 | 1024 | 1 | 16384 | 3.4179 | 3.1874 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
):
    # Constants
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    scaling = head_dim ** -0.5
    
    batch_size, seq_len, _ = hidden_states.shape
    
    # QKV projection
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape to [batch, seq_len, num_heads, head_dim] then transpose to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    
    # Rotate half for query
    q1 = query_states[..., : head_dim // 2]
    q2 = query_states[..., head_dim // 2 :]
    query_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (query_rotated * sin_expanded)
    
    # Rotate half for key
    k1 = key_states[..., : head_dim // 2]
    k2 = key_states[..., head_dim // 2 :]
    key_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (key_rotated * sin_expanded)
    
    # Repeat KV for GQA: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply causal mask
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
        diagonal=1
    )
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    output = F.linear(attn_output, o_proj_weight)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.075296 ms |
| - | Scoring Baseline | 0.500000 | 0.429356 ms |
| - | Reference Implementation | 0.100293 | 4.491758 ms |
