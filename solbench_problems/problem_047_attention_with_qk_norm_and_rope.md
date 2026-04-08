## Description

Gemma3 attention mechanism with query/key RMSNorm normalization, rotary position embeddings (RoPE), grouped-query attention (GQA) with 24 query heads and 8 KV heads, attention logit softcapping at 50.0, and causal masking. Includes Q/K/V/O projections.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| q_proj_weight | [qkv_out_dim, hidden_size] | bfloat16 |
| k_proj_weight | [kv_out_dim, hidden_size] | bfloat16 |
| v_proj_weight | [kv_out_dim, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, qkv_out_dim] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| attn_logit_softcapping | scalar | float32 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | qkv_out_dim | kv_out_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 4 | 2053 | 3.5068 | 0.3430 |
| 2 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 3557 | 2.4786 | 0.1851 |
| 3 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 128 | 0.2184 | 0.0049 |
| 4 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 32 | 256 | 0.9238 | 0.2423 |
| 5 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 16 | 256 | 0.5630 | 0.1213 |
| 6 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 4 | 256 | 0.2608 | 0.0306 |
| 7 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 1024 | 0.3966 | 0.0360 |
| 8 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 2 | 293 | 0.2464 | 0.0178 |
| 9 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 8 | 613 | 0.9869 | 0.1571 |
| 10 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 16 | 128 | 0.3111 | 0.0591 |
| 11 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 8192 | 11.1791 | 0.6833 |
| 12 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 8 | 256 | 0.3635 | 0.0609 |
| 13 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 2 | 512 | 0.3212 | 0.0324 |
| 14 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 32 | 128 | 0.4822 | 0.1178 |
| 15 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 512 | 0.2393 | 0.0164 |
| 16 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 4 | 512 | 0.4622 | 0.0644 |

```python
import torch
import torch.nn.functional as F

def rms_norm(x, weight, eps):
    """RMSNorm with (1 + weight) scaling as used in Gemma3."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    output = x_normed * (1.0 + weight.float())
    return output.type_as(x)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    """Repeats key/value heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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
    attn_logit_softcapping: float,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    
    # Project to Q, K, V using F.linear
    query_states = F.linear(hidden_states, q_proj_weight)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    value_states = F.linear(hidden_states, v_proj_weight)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply Q/K normalization (unique to Gemma3)
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    # Apply RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat K/V for GQA (8 KV heads -> 24 query heads)
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Compute attention scores with scaling
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply attention logit softcapping (unique to Gemma3)
    attn_weights = attn_weights / attn_logit_softcapping
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * attn_logit_softcapping
    
    # Apply causal mask
    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.069243 ms |
| - | Scoring Baseline | 0.500000 | 0.624699 ms |
| - | Reference Implementation | 0.290238 | 1.416486 ms |
