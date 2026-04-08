## Description

Complete decoder layer with pre-normalization, attention, MLP, and residual connections. Pattern: x = x + attention(norm(x)), x = x + mlp(norm(x)). This is the fundamental building block executed N times per forward pass.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, half_head_dim] | bfloat16 |
| sin | [batch_size, seq_len, half_head_dim] | bfloat16 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| input_layernorm_weight | [hidden_size] | bfloat16 |
| q_proj_weight | [qkv_hidden, hidden_size] | bfloat16 |
| k_proj_weight | [kv_hidden, hidden_size] | bfloat16 |
| v_proj_weight | [kv_hidden, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, qkv_hidden] | bfloat16 |
| post_attention_layernorm_weight | [hidden_size] | bfloat16 |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | num_attention_heads | num_key_value_heads | head_dim | qkv_hidden | kv_hidden | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 8 | 128 | 1.0053 | 0.5077 |
| 2 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 2 | 512 | 1.0455 | 0.5130 |
| 3 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 4 | 256 | 1.0190 | 0.5095 |
| 4 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 4 | 128 | 0.6213 | 0.2541 |
| 5 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 4 | 293 | 1.3260 | 0.5836 |
| 6 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 2 | 1024 | 1.8287 | 1.0399 |
| 7 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 2 | 2048 | 3.8652 | 2.1363 |
| 8 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 32 | 256 | 6.7587 | 4.0731 |
| 9 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 8 | 256 | 1.7167 | 1.0186 |
| 10 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 64 | 128 | 7.0236 | 4.0588 |
| 11 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 16 | 128 | 1.6492 | 1.0150 |
| 12 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 16 | 449 | 6.6507 | 3.5907 |
| 13 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 4 | 512 | 1.7546 | 1.0257 |
| 14 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 1 | 1024 | 1.0957 | 0.5202 |
| 15 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 8 | 512 | 3.3082 | 2.0510 |
| 16 | 6144 | 19648 | 64 | 8 | 96 | 6144 | 768 | 48 | 1 | 128 | 0.4257 | 0.0638 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    # Constants
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 96
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    half_dim = head_dim // 2
    
    batch_size, seq_len, _ = hidden_states.shape
    
    # Helper: RMSNorm
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight * x_normed.to(input_dtype))
    
    # Helper: Repeat KV heads for GQA
    def repeat_kv(x, n_rep):
        batch, num_kv_heads, slen, hdim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hdim)
        return x.reshape(batch, num_kv_heads * n_rep, slen, hdim)
    
    # ===== ATTENTION BLOCK =====
    residual1 = hidden_states
    
    # Pre-attention normalization
    hidden_states = rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)
    
    # QKV projections
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape for attention
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE - cos and sin have shape (batch, seq_len, half_head_dim)
    # Expand to (batch, 1, seq_len, half_head_dim) for broadcasting
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    # Rotate query: split into first half and second half
    q1 = query_states[..., :half_dim]
    q2 = query_states[..., half_dim:]
    query_states = torch.cat([q1 * cos_expanded - q2 * sin_expanded, q1 * sin_expanded + q2 * cos_expanded], dim=-1)
    
    # Rotate key: split into first half and second half
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    key_states = torch.cat([k1 * cos_expanded - k2 * sin_expanded, k1 * sin_expanded + k2 * cos_expanded], dim=-1)
    
    # Repeat KV heads for GQA
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_output = F.linear(attn_output, o_proj_weight)
    
    # First residual connection
    hidden_states = residual1 + attn_output
    
    # ===== MLP BLOCK =====
    residual2 = hidden_states
    
    # Pre-MLP normalization
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, rms_norm_eps)
    
    # SwiGLU MLP
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(gate * up, down_proj_weight)
    
    # Second residual connection
    output = residual2 + hidden_states
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.900610 ms |
| - | Scoring Baseline | 0.500000 | 1.824638 ms |
| - | Reference Implementation | 0.308383 | 2.938036 ms |
