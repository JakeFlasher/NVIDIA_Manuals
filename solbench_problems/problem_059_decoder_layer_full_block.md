## Description

Complete Qwen3 decoder layer block fusing: input RMSNorm -> QKV projection with QK norm -> RoPE -> GQA attention -> output projection -> residual -> post-attention RMSNorm -> SwiGLU MLP -> final residual. Hidden size 5120, 40 query heads, 8 KV heads, head dim 128, intermediate size 17408.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| position_ids | [batch_size, seq_len] | int64 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| input_layernorm_weight | [hidden_size] | bfloat16 |
| q_proj_weight | [num_heads_x_head_dim, hidden_size] | bfloat16 |
| k_proj_weight | [num_kv_heads_x_head_dim, hidden_size] | bfloat16 |
| v_proj_weight | [num_kv_heads_x_head_dim, hidden_size] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| o_proj_weight | [hidden_size, num_heads_x_head_dim] | bfloat16 |
| post_attention_layernorm_weight | [hidden_size] | bfloat16 |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 |
| inv_freq | [half_head_dim] | float32 |
| rms_norm_eps | scalar | float32 |
| attention_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | intermediate_size | num_heads_x_head_dim | num_kv_heads_x_head_dim | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 1 | 131 | 0.3423 | 0.0484 |
| 2 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 1 | 293 | 0.4566 | 0.1082 |
| 3 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 8 | 512 | 2.3180 | 1.5180 |
| 4 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 2 | 512 | 0.8390 | 0.3798 |
| 5 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 8 | 1024 | 4.7825 | 3.0831 |
| 6 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 2 | 256 | 0.4724 | 0.1886 |
| 7 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 2 | 1024 | 1.3715 | 0.7711 |
| 8 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 1 | 8192 | 8.2205 | 3.7470 |
| 9 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 1 | 512 | 0.4758 | 0.1901 |
| 10 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 1 | 2048 | 1.5351 | 0.7948 |
| 11 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 1 | 997 | 1.1376 | 0.3753 |
| 12 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 2 | 2048 | 2.7018 | 1.5892 |
| 13 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 4 | 256 | 0.8239 | 0.3768 |
| 14 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 4 | 512 | 1.2854 | 0.7592 |
| 15 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 16 | 128 | 1.2236 | 0.7503 |
| 16 | 5120 | 40 | 8 | 128 | 17408 | 5120 | 1024 | 64 | 16 | 256 | 2.2579 | 1.5062 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
    attention_scale: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)
    
    residual = hidden_states
    
    # 1. Input RMSNorm
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)
    
    # 2. QKV Projection
    q = F.linear(hidden_states, q_proj_weight)
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight)
    
    # Reshape to separate heads
    q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
    k = k.view(batch_size, seq_len, num_key_value_heads, head_dim)
    v = v.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # 3. QK Normalization
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)
    
    # Transpose to [batch, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # 4. Compute RoPE embeddings
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    
    # Apply RoPE
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k_rotated = torch.cat((-k2, k1), dim=-1)
    q = (q * cos) + (q_rotated * sin)
    k = (k * cos) + (k_rotated * sin)
    
    # 5. Repeat KV heads for GQA
    k = k[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)
    k = k.reshape(batch_size, num_attention_heads, seq_len, head_dim)
    v = v[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)
    v = v.reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # 6. Attention computation
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * attention_scale
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # 7. Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    # 8. First residual
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # 9. Post-attention RMSNorm
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)
    
    # 10. SwiGLU MLP
    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
    gate = F.silu(gate)
    intermediate = gate * up
    mlp_output = F.linear(intermediate, down_proj_weight)
    
    # 11. Final residual
    output = residual + mlp_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.576236 ms |
| - | Scoring Baseline | 0.500000 | 1.278351 ms |
| - | Reference Implementation | 0.233798 | 2.817865 ms |
