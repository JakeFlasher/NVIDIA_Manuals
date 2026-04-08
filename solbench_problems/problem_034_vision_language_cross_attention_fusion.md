## Description

Cross-attention fusion layer for Qwen3-VL multimodal processing. Performs cross-attention between language queries (with 1D RoPE) and vision key-values (with 3D RoPE for temporal-height-width positions). Uses Grouped Query Attention (GQA) with 32 query heads and 8 key-value heads.
| Name | Shape | Dtype |
| --- | --- | --- |
| language_hidden_states | [batch_size, lang_seq_len, hidden_size] | bfloat16 |
| vision_hidden_states | [batch_size, vision_seq_len, hidden_size] | bfloat16 |
| language_position_ids | [batch_size, lang_seq_len] | int64 |
| vision_grid_thw | [batch_size, vision_seq_len, thw_dim] | int64 |
| q_proj_weight | [q_proj_dim, hidden_size] | bfloat16 |
| q_proj_bias | [q_proj_dim] | bfloat16 |
| k_proj_weight | [kv_proj_dim, hidden_size] | bfloat16 |
| k_proj_bias | [kv_proj_dim] | bfloat16 |
| v_proj_weight | [kv_proj_dim, hidden_size] | bfloat16 |
| v_proj_bias | [kv_proj_dim] | bfloat16 |
| o_proj_weight | [hidden_size, q_proj_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, lang_seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | thw_dim | q_proj_dim | kv_proj_dim | batch_size | lang_seq_len | vision_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 2 | 256 | 392 | 0.2483 | 0.0284 |
| 2 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 64 | 256 | 392 | 1.7323 | 0.8979 |
| 3 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 1 | 512 | 784 | 0.2654 | 0.0303 |
| 4 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 2 | 1024 | 1568 | 0.4728 | 0.1344 |
| 5 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 16 | 512 | 784 | 1.0623 | 0.4782 |
| 6 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 1 | 4096 | 6272 | 0.9381 | 0.4426 |
| 7 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 4 | 541 | 980 | 0.4780 | 0.1361 |
| 8 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 2 | 512 | 784 | 0.3015 | 0.0601 |
| 9 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 8 | 128 | 196 | 0.3073 | 0.0547 |
| 10 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 8 | 256 | 392 | 0.4127 | 0.1126 |
| 11 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 1 | 8192 | 12544 | 2.1482 | 1.3497 |
| 12 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 1 | 1024 | 1568 | 0.3008 | 0.0674 |
| 13 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 32 | 373 | 588 | 1.5295 | 0.6804 |
| 14 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 4 | 512 | 784 | 0.4242 | 0.1199 |
| 15 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 16 | 128 | 196 | 0.4195 | 0.1090 |
| 16 | 4096 | 32 | 8 | 128 | 3 | 4096 | 1024 | 2 | 128 | 196 | 0.2677 | 0.0140 |

```python
import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    lang_seq_len = axes_and_scalars["lang_seq_len"]
    vision_seq_len = axes_and_scalars["vision_seq_len"]
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    
    language_hidden_states = torch.randn(batch_size, lang_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    vision_hidden_states = torch.randn(batch_size, vision_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Language position IDs: sequential positions
    language_position_ids = torch.arange(lang_seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    
    # Vision grid THW: generate valid 3D positions
    vision_grid_thw = torch.zeros(batch_size, vision_seq_len, 3, dtype=torch.int64, device=device)
    for b in range(batch_size):
        for i in range(vision_seq_len):
            t = i // 196
            spatial_idx = i % 196
            h = spatial_idx // 14
            w = spatial_idx % 14
            vision_grid_thw[b, i, 0] = t
            vision_grid_thw[b, i, 1] = h
            vision_grid_thw[b, i, 2] = w
    
    q_proj_weight = torch.randn(num_attention_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    q_proj_bias = torch.randn(num_attention_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_bias = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_bias = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, num_attention_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    
    return {
        "language_hidden_states": language_hidden_states,
        "vision_hidden_states": vision_hidden_states,
        "language_position_ids": language_position_ids,
        "vision_grid_thw": vision_grid_thw,
        "q_proj_weight": q_proj_weight,
        "q_proj_bias": q_proj_bias,
        "k_proj_weight": k_proj_weight,
        "k_proj_bias": k_proj_bias,
        "v_proj_weight": v_proj_weight,
        "v_proj_bias": v_proj_bias,
        "o_proj_weight": o_proj_weight,
    }

@torch.no_grad()
def run(
    language_hidden_states: torch.Tensor,
    vision_hidden_states: torch.Tensor,
    language_position_ids: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
):
    # Constants
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_kv_groups = num_attention_heads // num_key_value_heads
    rope_theta = 10000.0
    
    batch_size, lang_seq_len, _ = language_hidden_states.shape
    vision_seq_len = vision_hidden_states.shape[1]
    
    # Compute RoPE inverse frequencies
    inv_freq_1d = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=language_hidden_states.device) / head_dim))
    # For 3D RoPE: head_dim=128, split into 3 parts of ~42/43 each
    # We use 42, 42, 44 split (42+42+44=128)
    dim_t = 42
    dim_h = 42
    dim_w = 44
    inv_freq_t = 1.0 / (rope_theta ** (torch.arange(0, dim_t, 2, dtype=torch.float32, device=language_hidden_states.device) / dim_t))
    inv_freq_h = 1.0 / (rope_theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32, device=language_hidden_states.device) / dim_h))
    inv_freq_w = 1.0 / (rope_theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32, device=language_hidden_states.device) / dim_w))
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb_1d(q, position_ids, inv_freq):
        # q: (batch, num_heads, seq_len, head_dim)
        # position_ids: (batch, seq_len)
        position_ids_expanded = position_ids.unsqueeze(1).unsqueeze(-1).float()  # (batch, 1, seq_len, 1)
        freqs = position_ids_expanded * inv_freq  # (batch, 1, seq_len, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, 1, seq_len, head_dim)
        cos = emb.cos().to(q.dtype)
        sin = emb.sin().to(q.dtype)
        return (q * cos) + (rotate_half(q) * sin)
    
    def apply_rotary_pos_emb_3d(k, vision_grid_thw, inv_freq_t, inv_freq_h, inv_freq_w, dim_t, dim_h, dim_w):
        # k: (batch, num_kv_heads, vision_seq_len, head_dim)
        # vision_grid_thw: (batch, vision_seq_len, 3)
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Split k into t, h, w parts
        k_t = k[..., :dim_t]  # (batch, num_heads, seq_len, dim_t)
        k_h = k[..., dim_t:dim_t+dim_h]  # (batch, num_heads, seq_len, dim_h)
        k_w = k[..., dim_t+dim_h:]  # (batch, num_heads, seq_len, dim_w)
        
        t_pos = vision_grid_thw[..., 0]  # (batch, seq_len)
        h_pos = vision_grid_thw[..., 1]
        w_pos = vision_grid_thw[..., 2]
        
        def apply_1d_rope_component(x, pos, inv_freq):
            # x: (batch, num_heads, seq_len, dim)
            # pos: (batch, seq_len)
            # inv_freq: (dim//2,)
            pos_expanded = pos.unsqueeze(1).unsqueeze(-1).float()  # (batch, 1, seq_len, 1)
            freqs = pos_expanded * inv_freq  # (batch, 1, seq_len, dim//2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (batch, 1, seq_len, dim)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
            return (x * cos) + (rotate_half(x) * sin)
        
        k_t_rotated = apply_1d_rope_component(k_t, t_pos, inv_freq_t)
        k_h_rotated = apply_1d_rope_component(k_h, h_pos, inv_freq_h)
        k_w_rotated = apply_1d_rope_component(k_w, w_pos, inv_freq_w)
        
        return torch.cat([k_t_rotated, k_h_rotated, k_w_rotated], dim=-1)
    
    def repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
    # Project queries from language tokens
    query_states = F.linear(language_hidden_states, q_proj_weight, q_proj_bias)
    query_states = query_states.view(batch_size, lang_seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    # Project keys and values from vision tokens
    key_states = F.linear(vision_hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(vision_hidden_states, v_proj_weight, v_proj_bias)
    
    key_states = key_states.view(batch_size, vision_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, vision_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply rotary position embeddings
    query_states = apply_rotary_pos_emb_1d(query_states, language_position_ids, inv_freq_1d)
    key_states = apply_rotary_pos_emb_3d(key_states, vision_grid_thw, inv_freq_t, inv_freq_h, inv_freq_w, dim_t, dim_h, dim_w)
    
    # Repeat k/v heads for GQA
    key_states = repeat_kv(key_states, num_kv_groups)
    value_states = repeat_kv(value_states, num_kv_groups)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, lang_seq_len, num_attention_heads * head_dim)
    output = F.linear(attn_output, o_proj_weight)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.135252 ms |
| - | Scoring Baseline | 0.500000 | 0.535901 ms |
| - | Reference Implementation | 0.178708 | 2.046093 ms |
