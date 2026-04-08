## Description

Grouped Query Attention with RoPE and Query/Key RMSNorm. Features 32 query heads, 8 KV heads (4x grouping), per-head RMSNorm on Q/K before RoPE application, and causal attention. This is the core attention mechanism from Qwen3-8B.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| attention_mask | [batch_size, one, seq_len, seq_len] | bfloat16 |
| q_proj_weight | [q_proj_out_dim, hidden_size] | bfloat16 |
| k_proj_weight | [kv_proj_out_dim, hidden_size] | bfloat16 |
| v_proj_weight | [kv_proj_out_dim, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, q_proj_out_dim] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| rms_norm_eps | scalar | float32 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | one | q_proj_out_dim | kv_proj_out_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 4096 | 0.6070 | 0.3419 |
| 2 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 8 | 1024 | 0.8608 | 0.4557 |
| 3 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 4 | 128 | 0.2310 | 0.0247 |
| 4 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 16 | 512 | 0.8457 | 0.4177 |
| 5 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 2 | 541 | 0.3039 | 0.0558 |
| 6 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 32 | 128 | 0.5156 | 0.1948 |
| 7 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 8192 | 1.2117 | 0.9868 |
| 8 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 2048 | 0.3855 | 0.1332 |
| 9 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 2 | 128 | 0.2152 | 0.0126 |
| 10 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 131 | 0.2154 | 0.0075 |
| 11 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 4 | 1024 | 0.5411 | 0.2280 |
| 12 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 512 | 0.2370 | 0.0265 |
| 13 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 997 | 0.2776 | 0.0556 |
| 14 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 64 | 128 | 0.8309 | 0.3893 |
| 15 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 1 | 293 | 0.2257 | 0.0147 |
| 16 | 4096 | 32 | 8 | 128 | 4 | 1 | 4096 | 1024 | 16 | 256 | 0.5105 | 0.1996 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    scaling: float,
):
    # Constants
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    
    batch_size, seq_len, _ = hidden_states.shape
    
    # QKV Projections
    # Q: (batch, seq_len, 4096) @ (4096, 4096).T -> (batch, seq_len, 4096)
    query_states = F.linear(hidden_states, q_proj_weight)
    # K: (batch, seq_len, 4096) @ (1024, 4096).T -> (batch, seq_len, 1024)
    key_states = F.linear(hidden_states, k_proj_weight)
    # V: (batch, seq_len, 4096) @ (1024, 4096).T -> (batch, seq_len, 1024)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape to (batch, seq_len, num_heads, head_dim)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Per-head RMSNorm on Q and K
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight * x_normed.to(input_dtype))
    
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    # Transpose to (batch, num_heads, seq_len, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Apply RoPE
    # cos, sin: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    query_states = (query_states * cos_expanded) + (rotate_half(query_states) * sin_expanded)
    key_states = (key_states * cos_expanded) + (rotate_half(key_states) * sin_expanded)
    
    # Repeat KV heads for grouped query attention (8 -> 32 heads)
    # key_states: (batch, 8, seq_len, 128) -> (batch, 32, seq_len, 128)
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    # (batch, 32, seq_len, 128) @ (batch, 32, 128, seq_len) -> (batch, 32, seq_len, seq_len)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply causal mask
    attn_weights = attn_weights + attention_mask
    
    # Softmax (compute in float32 for numerical stability)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    # (batch, 32, seq_len, seq_len) @ (batch, 32, seq_len, 128) -> (batch, 32, seq_len, 128)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape: (batch, 32, seq_len, 128) -> (batch, seq_len, 32, 128) -> (batch, seq_len, 4096)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.098863 ms |
| - | Scoring Baseline | 0.500000 | 0.426079 ms |
| - | Reference Implementation | 0.158605 | 1.976164 ms |
