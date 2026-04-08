## Description

Backward pass for Grouped Query Attention with GLM-style Rotary Position Embeddings. Computes gradients for hidden_states and all projection weights (Q, K, V, O) through the complete attention mechanism including RoPE, GQA head expansion, scaled dot-product attention, and output projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_weight | [hidden_size, hidden_size] | bfloat16 |
| k_weight | [kv_hidden_size, hidden_size] | bfloat16 |
| v_weight | [kv_hidden_size, hidden_size] | bfloat16 |
| o_weight | [hidden_size, hidden_size] | bfloat16 |
| inv_freq | [half_head_dim] | float32 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_q_weight | [hidden_size, hidden_size] | bfloat16 |
| grad_k_weight | [kv_hidden_size, hidden_size] | bfloat16 |
| grad_v_weight | [kv_hidden_size, hidden_size] | bfloat16 |
| grad_o_weight | [hidden_size, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | kv_hidden_size | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 64 | 8 | 128 | 1024 | 64 | 1 | 373 | 0.7908 | 0.1669 |
| 2 | 8192 | 64 | 8 | 128 | 1024 | 64 | 1 | 541 | 1.1447 | 0.2468 |
| 3 | 8192 | 64 | 8 | 128 | 1024 | 64 | 4 | 919 | 3.1011 | 1.7500 |
| 4 | 8192 | 64 | 8 | 128 | 1024 | 64 | 1 | 128 | 0.6061 | 0.1036 |
| 5 | 8192 | 64 | 8 | 128 | 1024 | 64 | 1 | 2048 | 2.0794 | 1.1007 |
| 6 | 8192 | 64 | 8 | 128 | 1024 | 64 | 64 | 853 | 44.6302 | 25.7888 |
| 7 | 8192 | 64 | 8 | 128 | 1024 | 64 | 8 | 256 | 2.3039 | 0.9015 |
| 8 | 8192 | 64 | 8 | 128 | 1024 | 64 | 16 | 256 | 3.9489 | 1.8026 |
| 9 | 8192 | 64 | 8 | 128 | 1024 | 64 | 32 | 128 | 3.4699 | 1.7741 |
| 10 | 8192 | 64 | 8 | 128 | 1024 | 64 | 2 | 449 | 1.1756 | 0.4049 |
| 11 | 8192 | 64 | 8 | 128 | 1024 | 64 | 2 | 1024 | 2.0397 | 0.9868 |
| 12 | 8192 | 64 | 8 | 128 | 1024 | 64 | 8 | 691 | 4.5584 | 2.5631 |
| 13 | 8192 | 64 | 8 | 128 | 1024 | 64 | 1 | 1024 | 1.2472 | 0.4936 |
| 14 | 8192 | 64 | 8 | 128 | 1024 | 64 | 64 | 128 | 6.7464 | 3.5478 |
| 15 | 8192 | 64 | 8 | 128 | 1024 | 64 | 2 | 293 | 1.5618 | 0.2594 |
| 16 | 8192 | 64 | 8 | 128 | 1024 | 64 | 2 | 128 | 0.6981 | 0.1113 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for GQA with RoPE.
    
    Computes gradients through:
    1. Output projection
    2. Attention mechanism (softmax, matmul)
    3. GQA head expansion
    4. RoPE application
    5. QKV projections
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 8
    device = hidden_states.device
    original_dtype = hidden_states.dtype
    
    # ========== FORWARD RECOMPUTATION ==========
    # QKV projections
    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)
    
    # Reshape to multi-head format
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Compute RoPE embeddings
    position_ids = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    inv_freq_expanded = inv_freq[None, :, None].float()
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    # Prepare cos/sin for interleaved pattern
    cos_unsqueezed = cos.unsqueeze(1)
    sin_unsqueezed = sin.unsqueeze(1)
    cos_interleaved = cos_unsqueezed[..., : cos_unsqueezed.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin_interleaved = sin_unsqueezed[..., : sin_unsqueezed.shape[-1] // 2].repeat_interleave(2, dim=-1)
    
    # Apply rotary position embeddings
    q_float = query_states.float()
    k_float = key_states.float()
    
    # Rotate half for query and key
    q1 = q_float[..., 0::2]
    q2 = q_float[..., 1::2]
    q_rotated = torch.stack((-q2, q1), dim=-1).flatten(-2)
    
    k1 = k_float[..., 0::2]
    k2 = k_float[..., 1::2]
    k_rotated = torch.stack((-k2, k1), dim=-1).flatten(-2)
    
    query_states_rope = (q_float * cos_interleaved) + (q_rotated * sin_interleaved)
    key_states_rope = (k_float * cos_interleaved) + (k_rotated * sin_interleaved)
    
    query_states_rope = query_states_rope.to(original_dtype)
    key_states_rope = key_states_rope.to(original_dtype)
    
    # Expand key/value heads
    key_states_expanded = key_states_rope[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    value_states_expanded = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states_rope, key_states_expanded.transpose(2, 3)) * scaling
    
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attn_weights_float = F.softmax(attn_weights.float(), dim=-1)
    attn_weights_dtype = attn_weights_float.to(original_dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights_dtype, value_states_expanded)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # ========== BACKWARD PASS ==========
    # Gradient through output projection
    grad_attn_output = F.linear(grad_output, o_weight.t())
    grad_o_weight = grad_output.reshape(-1, hidden_size).t() @ attn_output.reshape(-1, hidden_size)
    
    # Reshape gradient for attention output
    grad_attn_output = grad_attn_output.view(batch_size, seq_len, num_attention_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(1, 2)
    
    # Gradient through attention matmul: attn_output = attn_weights @ value_states_expanded
    grad_attn_weights = torch.matmul(grad_attn_output, value_states_expanded.transpose(2, 3))
    grad_value_states_expanded = torch.matmul(attn_weights_dtype.transpose(2, 3), grad_attn_output)
    
    # Gradient through softmax
    grad_attn_weights_float = grad_attn_weights.float()
    attn_weights_float_saved = attn_weights_float
    grad_attn_scores = attn_weights_float_saved * (grad_attn_weights_float - (grad_attn_weights_float * attn_weights_float_saved).sum(dim=-1, keepdim=True))
    
    # Gradient through scaling
    grad_attn_scores = grad_attn_scores * scaling
    
    # Gradient through attention matmul: attn_scores = query_states_rope @ key_states_expanded.T
    grad_query_states_rope = torch.matmul(grad_attn_scores.to(original_dtype), key_states_expanded)
    grad_key_states_expanded = torch.matmul(query_states_rope.transpose(2, 3), grad_attn_scores.to(original_dtype)).transpose(2, 3)
    
    # Gradient through key-value expansion
    grad_key_states_rope = grad_key_states_expanded.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    
    grad_value_states = grad_value_states_expanded.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    
    # Gradient through RoPE for query
    grad_query_states_rope_float = grad_query_states_rope.float()
    
    # Gradient w.r.t. q_float before rotation
    grad_q_float = grad_query_states_rope_float * cos_interleaved
    
    # Gradient w.r.t. q_rotated
    grad_q_rotated = grad_query_states_rope_float * sin_interleaved
    
    # Gradient through rotate_half for query
    grad_q_rotated_reshaped = grad_q_rotated.view(*grad_q_rotated.shape[:-1], head_dim // 2, 2)
    grad_q1_from_rotated = grad_q_rotated_reshaped[..., 1]
    grad_q2_from_rotated = -grad_q_rotated_reshaped[..., 0]
    
    # Combine gradients
    grad_q_float[..., 0::2] = grad_q_float[..., 0::2] + grad_q1_from_rotated
    grad_q_float[..., 1::2] = grad_q_float[..., 1::2] + grad_q2_from_rotated
    
    grad_query_states = grad_q_float.to(original_dtype)
    
    # Gradient through RoPE for key
    grad_key_states_rope_float = grad_key_states_rope.float()
    
    # Gradient w.r.t. k_float before rotation
    grad_k_float = grad_key_states_rope_float * cos_interleaved
    
    # Gradient w.r.t. k_rotated
    grad_k_rotated = grad_key_states_rope_float * sin_interleaved
    
    # Gradient through rotate_half for key
    grad_k_rotated_reshaped = grad_k_rotated.view(*grad_k_rotated.shape[:-1], head_dim // 2, 2)
    grad_k1_from_rotated = grad_k_rotated_reshaped[..., 1]
    grad_k2_from_rotated = -grad_k_rotated_reshaped[..., 0]
    
    # Combine gradients
    grad_k_float[..., 0::2] = grad_k_float[..., 0::2] + grad_k1_from_rotated
    grad_k_float[..., 1::2] = grad_k_float[..., 1::2] + grad_k2_from_rotated
    
    grad_key_states = grad_k_float.to(original_dtype)
    
    # Gradient through reshape and transpose for Q, K, V
    grad_query_states = grad_query_states.transpose(1, 2).contiguous()
    grad_key_states = grad_key_states.transpose(1, 2).contiguous()
    grad_value_states = grad_value_states.transpose(1, 2).contiguous()
    
    grad_query_states = grad_query_states.view(batch_size, seq_len, hidden_size)
    grad_key_states = grad_key_states.view(batch_size, seq_len, num_key_value_heads * head_dim)
    grad_value_states = grad_value_states.view(batch_size, seq_len, num_key_value_heads * head_dim)
    
    # Gradient through QKV projections
    grad_hidden_states_q = F.linear(grad_query_states, q_weight.t())
    grad_hidden_states_k = F.linear(grad_key_states, k_weight.t())
    grad_hidden_states_v = F.linear(grad_value_states, v_weight.t())
    
    grad_hidden_states = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v
    
    grad_q_weight = grad_query_states.reshape(-1, hidden_size).t() @ hidden_states.reshape(-1, hidden_size)
    grad_k_weight = grad_key_states.reshape(-1, num_key_value_heads * head_dim).t() @ hidden_states.reshape(-1, hidden_size)
    grad_v_weight = grad_value_states.reshape(-1, num_key_value_heads * head_dim).t() @ hidden_states.reshape(-1, hidden_size)
    
    return (
        grad_hidden_states.to(original_dtype),
        grad_q_weight.to(original_dtype),
        grad_k_weight.to(original_dtype),
        grad_v_weight.to(original_dtype),
        grad_o_weight.to(original_dtype)
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.805875 ms |
| - | Scoring Baseline | 0.500000 | 2.270819 ms |
| - | Reference Implementation | 0.255225 | 5.195940 ms |
