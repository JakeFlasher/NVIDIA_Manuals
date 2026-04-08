## Description

Grouped Query Attention with QK normalization for GLM-4. Includes Q/K/V projections, RMSNorm on Q/K, RoPE, GQA with 96 query heads and 8 KV heads (12x repetition), scaled dot-product attention with causal mask, and output projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_proj_weight | [q_out_features, hidden_size] | bfloat16 |
| q_proj_bias | [q_out_features] | bfloat16 |
| k_proj_weight | [kv_out_features, hidden_size] | bfloat16 |
| k_proj_bias | [kv_out_features] | bfloat16 |
| v_proj_weight | [kv_out_features, hidden_size] | bfloat16 |
| v_proj_bias | [kv_out_features] | bfloat16 |
| o_proj_weight | [hidden_size, q_out_features] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_out_features | kv_out_features | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 1 | 512 | 0.2618 | 0.0692 |
| 2 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 4 | 128 | 0.2590 | 0.0638 |
| 3 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 2 | 256 | 0.2565 | 0.0656 |
| 4 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 16 | 256 | 0.9606 | 0.5221 |
| 5 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 8 | 373 | 0.8016 | 0.3899 |
| 6 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 1 | 2048 | 0.6505 | 0.3608 |
| 7 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 1 | 1024 | 0.3840 | 0.1522 |
| 8 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 2 | 2048 | 1.1098 | 0.7213 |
| 9 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 4 | 512 | 0.5802 | 0.2755 |
| 10 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 8 | 512 | 0.9757 | 0.5505 |
| 11 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 16 | 128 | 0.5839 | 0.2541 |
| 12 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 4 | 1024 | 1.0201 | 0.6074 |
| 13 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 1 | 128 | 0.2474 | 0.0163 |
| 14 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 32 | 256 | 1.7281 | 1.0438 |
| 15 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 2 | 512 | 0.3709 | 0.1379 |
| 16 | 4096 | 96 | 8 | 128 | 12288 | 1024 | 4 | 293 | 0.4332 | 0.1508 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, seq_length, _ = hidden_states.shape
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 12
    scaling = head_dim ** -0.5
    
    # Q/K/V projections
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    
    # Reshape to separate heads
    query_states = query_states.view(batch_size, seq_length, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
    
    # Apply QK RMSNorm
    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)
    
    query_states = rms_norm(query_states, q_norm_weight)
    key_states = rms_norm(key_states, k_norm_weight)
    
    # Transpose to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Apply RoPE
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)
    
    # Rotate half for Q
    q1, q2 = query_states[..., :64], query_states[..., 64:]
    q_rot_half = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (q_rot_half * sin_expanded)
    
    # Rotate half for K
    k1, k2 = key_states[..., :64], key_states[..., 64:]
    k_rot_half = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (k_rot_half * sin_expanded)
    
    # Repeat KV heads for GQA: [batch, 8, seq_len, 128] -> [batch, 96, seq_len, 128]
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    ).reshape(batch_size, num_attention_heads, seq_length, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    ).reshape(batch_size, num_attention_heads, seq_length, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply causal mask
    causal_mask = torch.triu(
        torch.full((seq_length, seq_length), float('-inf'), device=hidden_states.device, dtype=attn_weights.dtype),
        diagonal=1
    )
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Transpose and reshape
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, num_attention_heads * head_dim)
    
    # Output projection (no bias)
    output = F.linear(attn_output, o_proj_weight, None)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.213995 ms |
| - | Scoring Baseline | 0.500000 | 0.557058 ms |
| - | Reference Implementation | 0.137329 | 2.519632 ms |
