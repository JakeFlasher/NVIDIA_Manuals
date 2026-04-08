## Description

Grouped Query Attention with Multi-modal 3D Rotary Position Embeddings. Supports 28 query heads and 4 KV heads (7:1 GQA ratio) with specialized 3D RoPE for temporal/height/width dimensions in vision tokens.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_weight | [q_proj_dim, hidden_size] | bfloat16 |
| q_bias | [q_proj_dim] | bfloat16 |
| k_weight | [kv_proj_dim, hidden_size] | bfloat16 |
| k_bias | [kv_proj_dim] | bfloat16 |
| v_weight | [kv_proj_dim, hidden_size] | bfloat16 |
| v_bias | [kv_proj_dim] | bfloat16 |
| o_weight | [hidden_size, q_proj_dim] | bfloat16 |
| cos | [rope_dim_0, batch_size, seq_len, head_dim] | bfloat16 |
| sin | [rope_dim_0, batch_size, seq_len, head_dim] | bfloat16 |
| attention_mask | [batch_size, one, seq_len, seq_len] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_heads | num_kv_heads | head_dim | num_kv_groups | rope_dim_0 | one | q_proj_dim | kv_proj_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 4 | 128 | 0.2421 | 0.0175 |
| 2 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 64 | 256 | 1.0238 | 0.5648 |
| 3 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 4 | 1024 | 0.4284 | 0.1664 |
| 4 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 32 | 128 | 0.3192 | 0.1373 |
| 5 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 32 | 541 | 1.1268 | 0.6358 |
| 6 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 2 | 1024 | 0.2803 | 0.0834 |
| 7 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 1 | 1024 | 0.3027 | 0.0419 |
| 8 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 32 | 256 | 0.5382 | 0.2826 |
| 9 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 1 | 512 | 0.2419 | 0.0191 |
| 10 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 4 | 256 | 0.2847 | 0.0357 |
| 11 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 1 | 128 | 0.2316 | 0.0047 |
| 12 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 1 | 613 | 0.2595 | 0.0232 |
| 13 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 1 | 4096 | 3.1884 | 0.2660 |
| 14 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 8 | 512 | 0.3211 | 0.1498 |
| 15 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 8 | 128 | 0.2874 | 0.0346 |
| 16 | 3584 | 28 | 4 | 128 | 7 | 3 | 1 | 3584 | 512 | 1 | 256 | 0.2341 | 0.0092 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Grouped Query Attention with Multi-modal 3D Rotary Position Embeddings.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        q_weight: [num_heads * head_dim, hidden_size]
        q_bias: [num_heads * head_dim]
        k_weight: [num_kv_heads * head_dim, hidden_size]
        k_bias: [num_kv_heads * head_dim]
        v_weight: [num_kv_heads * head_dim, hidden_size]
        v_bias: [num_kv_heads * head_dim]
        o_weight: [hidden_size, num_heads * head_dim]
        cos: [3, batch_size, seq_len, head_dim]
        sin: [3, batch_size, seq_len, head_dim]
        attention_mask: [batch_size, 1, seq_len, seq_len]
    
    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    # Constants
    num_heads = 28
    num_kv_heads = 4
    num_kv_groups = 7
    head_dim = 128
    scaling = head_dim ** -0.5
    mrope_section = [16, 24, 24]  # Channel splits for temporal/height/width
    
    bsz, q_len, _ = hidden_states.size()
    
    # Project to Q, K, V using linear operations
    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Apply multi-modal rotary position embeddings
    # mrope_section * 2 for cos/sin pairs: [32, 48, 48]
    mrope_section_doubled = [s * 2 for s in mrope_section]
    
    # Split cos/sin into 3 sections
    cos_splits = cos.split(mrope_section_doubled, dim=-1)
    sin_splits = sin.split(mrope_section_doubled, dim=-1)
    
    # Concatenate sections: [temporal, height, width] pattern
    cos_combined = torch.cat([m[i % 3] for i, m in enumerate(cos_splits)], dim=-1).unsqueeze(1)
    sin_combined = torch.cat([m[i % 3] for i, m in enumerate(sin_splits)], dim=-1).unsqueeze(1)
    
    # Rotate half helper
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotary embeddings
    query_states = (query_states * cos_combined) + (rotate_half(query_states) * sin_combined)
    key_states = (key_states * cos_combined) + (rotate_half(key_states) * sin_combined)
    
    # Repeat K, V for grouped query attention (4 -> 28 heads)
    # Expand from [batch, num_kv_heads, seq_len, head_dim] to [batch, num_heads, seq_len, head_dim]
    key_states = key_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim
    ).reshape(bsz, num_heads, q_len, head_dim)
    
    value_states = value_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim
    ).reshape(bsz, num_heads, q_len, head_dim)
    
    # Compute attention scores: Q @ K^T
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply attention mask (causal)
    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output: softmax(QK^T) @ V
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    output = F.linear(attn_output, o_weight)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.066128 ms |
| - | Scoring Baseline | 0.500000 | 0.403676 ms |
| - | Reference Implementation | 0.266118 | 1.006034 ms |
