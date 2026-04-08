## Description

Grouped Query Attention with YARN RoPE scaling and Q/K normalization. 40 query heads, 8 key-value heads (5:1 GQA ratio), head dimension 128, YARN RoPE with 8x scaling for 65536 max positions, sliding window attention 4096 tokens, RMS normalization on Q/K before attention.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| position_ids | [batch_size, seq_len] | int64 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| q_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| k_proj_weight | [kv_hidden_size, hidden_size] | bfloat16 |
| v_proj_weight | [kv_hidden_size, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| q_norm_weight | [hidden_size] | bfloat16 |
| k_norm_weight | [kv_hidden_size] | bfloat16 |
| inv_freq | [half_head_dim] | float32 |
| rms_norm_eps | scalar | float32 |
| attention_factor | scalar | float32 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | kv_hidden_size | num_key_value_groups | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 64 | 128 | 1.0161 | 0.5814 |
| 2 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 2 | 1571 | 0.5973 | 0.2745 |
| 3 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 8 | 256 | 0.3596 | 0.1486 |
| 4 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 4 | 2131 | 1.2727 | 0.7980 |
| 5 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 2 | 541 | 0.2948 | 0.0822 |
| 6 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 1 | 293 | 0.2294 | 0.0217 |
| 7 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 1 | 131 | 0.2070 | 0.0107 |
| 8 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 32 | 128 | 0.6208 | 0.2909 |
| 9 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 16 | 256 | 0.6579 | 0.2968 |
| 10 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 1 | 997 | 0.2807 | 0.0809 |
| 11 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 1 | 2048 | 0.4042 | 0.1901 |
| 12 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 4 | 128 | 0.2106 | 0.0367 |
| 13 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 2 | 1024 | 0.4575 | 0.1664 |
| 14 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 8 | 131 | 0.2884 | 0.0748 |
| 15 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 4 | 773 | 0.5483 | 0.2422 |
| 16 | 5120 | 40 | 8 | 128 | 1024 | 5 | 64 | 2 | 2048 | 0.6839 | 0.3798 |

```python
import torch
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 5120
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    kv_hidden_size = num_key_value_heads * head_dim
    half_head_dim = head_dim // 2
    sliding_window = 4096
    
    dtype = torch.bfloat16
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create sliding window causal mask
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    # Apply sliding window
    for i in range(seq_len):
        start = max(0, i - sliding_window + 1)
        if start > 0:
            mask[i, :start] = float("-inf")
    attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    
    q_proj_weight = torch.randn(hidden_size, hidden_size, dtype=dtype, device=device) * 0.02
    k_proj_weight = torch.randn(kv_hidden_size, hidden_size, dtype=dtype, device=device) * 0.02
    v_proj_weight = torch.randn(kv_hidden_size, hidden_size, dtype=dtype, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, hidden_size, dtype=dtype, device=device) * 0.02
    
    q_norm_weight = torch.ones(hidden_size, dtype=dtype, device=device)
    k_norm_weight = torch.ones(kv_hidden_size, dtype=dtype, device=device)
    
    # Compute YARN RoPE inv_freq
    rope_theta = 500000.0
    rope_scaling_factor = 8.0
    beta_fast = 32
    beta_slow = 1
    original_max_position_embeddings = 8192
    
    inv_freq_base = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    old_context_len = original_max_position_embeddings
    low_freq_wavelen = old_context_len / beta_fast
    high_freq_wavelen = old_context_len / beta_slow
    wavelen = 2 * math.pi / inv_freq_base
    smooth_factor = (old_context_len / wavelen - beta_fast) / (beta_slow - beta_fast)
    smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
    inv_freq = (1 - smooth_factor) * (inv_freq_base / rope_scaling_factor) + smooth_factor * inv_freq_base
    
    rms_norm_eps = 1e-6
    attention_factor_val = 1.2079441541679836
    scaling_val = head_dim ** -0.5
    
    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "o_proj_weight": o_proj_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "inv_freq": inv_freq,
        "rms_norm_eps": rms_norm_eps,
        "attention_factor": attention_factor_val,
        "scaling": scaling_val,
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
    attention_factor: float,
    scaling: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 5
    
    # Project Q, K, V
    query_states = torch.nn.functional.linear(hidden_states, q_proj_weight)
    key_states = torch.nn.functional.linear(hidden_states, k_proj_weight)
    value_states = torch.nn.functional.linear(hidden_states, v_proj_weight)
    
    # Apply Q normalization (RMS norm)
    q_fp32 = query_states.to(torch.float32)
    q_variance = q_fp32.pow(2).mean(-1, keepdim=True)
    q_normed = q_fp32 * torch.rsqrt(q_variance + rms_norm_eps)
    query_states = (q_norm_weight * q_normed).to(hidden_states.dtype)
    
    # Apply K normalization (RMS norm)
    k_fp32 = key_states.to(torch.float32)
    k_variance = k_fp32.pow(2).mean(-1, keepdim=True)
    k_normed = k_fp32 * torch.rsqrt(k_variance + rms_norm_eps)
    key_states = (k_norm_weight * k_normed).to(hidden_states.dtype)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply YARN RoPE
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_factor).unsqueeze(1).to(query_states.dtype)
    sin = (emb.sin() * attention_factor).unsqueeze(1).to(query_states.dtype)
    
    # Rotate half helper inline
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    
    # Repeat KV for GQA
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply attention mask (includes sliding window)
    attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # Output projection
    output = torch.nn.functional.linear(attn_output, o_proj_weight)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.139323 ms |
| - | Scoring Baseline | 0.500000 | 0.438639 ms |
| - | Reference Implementation | 0.154723 | 1.873055 ms |
