## Description

Complete decoder transformer layer combining pre-norm RMSNorm, GQA self-attention with RoPE (16 query heads, 4 KV heads), residual connection, pre-norm RMSNorm for cross-attention, GQA cross-attention (16 query heads, 16 KV heads), residual connection, pre-norm RMSNorm for MLP, SiLU-gated MLP (8192 intermediate), and final residual. This represents the full decoder layer that repeats 18 times.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| encoder_hidden_states | [batch_size, encoder_seq_len, cross_hidden_size] | bfloat16 |
| self_attn_norm_weight | [hidden_size] | bfloat16 |
| self_attn_q_weight | [self_attn_q_dim, hidden_size] | bfloat16 |
| self_attn_k_weight | [self_attn_kv_dim, hidden_size] | bfloat16 |
| self_attn_v_weight | [self_attn_kv_dim, hidden_size] | bfloat16 |
| self_attn_o_weight | [hidden_size, self_attn_q_dim] | bfloat16 |
| cross_attn_norm_weight | [hidden_size] | bfloat16 |
| cross_attn_q_weight | [cross_attn_q_dim, hidden_size] | bfloat16 |
| cross_attn_k_weight | [cross_attn_kv_dim, cross_hidden_size] | bfloat16 |
| cross_attn_v_weight | [cross_attn_kv_dim, cross_hidden_size] | bfloat16 |
| cross_attn_o_weight | [hidden_size, cross_attn_q_dim] | bfloat16 |
| mlp_norm_weight | [hidden_size] | bfloat16 |
| mlp_gate_weight | [intermediate_size, hidden_size] | bfloat16 |
| mlp_up_weight | [intermediate_size, hidden_size] | bfloat16 |
| mlp_down_weight | [hidden_size, intermediate_size] | bfloat16 |
| norm_eps | scalar | float32 |
| rope_theta | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | intermediate_size | num_attention_heads | num_key_value_heads | head_dim | cross_num_attention_heads | cross_num_key_value_heads | cross_head_dim | cross_hidden_size | self_attn_q_dim | self_attn_kv_dim | cross_attn_q_dim | cross_attn_kv_dim | batch_size | seq_len | encoder_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 1 | 256 | 512 | 0.3320 | 0.0232 |
| 2 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 2 | 293 | 512 | 0.3725 | 0.0521 |
| 3 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 1 | 1024 | 2048 | 0.4361 | 0.1024 |
| 4 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 32 | 256 | 512 | 1.3835 | 0.7308 |
| 5 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 2 | 256 | 512 | 0.3520 | 0.0460 |
| 6 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 4 | 512 | 1024 | 0.5509 | 0.1901 |
| 7 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 4 | 128 | 256 | 0.4204 | 0.0452 |
| 8 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 1 | 512 | 1024 | 0.3756 | 0.0478 |
| 9 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 8 | 256 | 512 | 0.5320 | 0.1830 |
| 10 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 8 | 512 | 1024 | 0.8622 | 0.3798 |
| 11 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 4 | 2213 | 2560 | 1.6383 | 0.9154 |
| 12 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 2 | 1801 | 2048 | 0.8444 | 0.3573 |
| 13 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 1 | 3072 | 3072 | 0.8015 | 0.3347 |
| 14 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 2 | 128 | 256 | 0.3480 | 0.0228 |
| 15 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 16 | 512 | 1024 | 1.4283 | 0.7592 |
| 16 | 2048 | 8192 | 16 | 4 | 128 | 16 | 16 | 128 | 1024 | 2048 | 512 | 2048 | 2048 | 1 | 131 | 256 | 0.3585 | 0.0118 |

```python
import torch
import torch.nn.functional as F
import math

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
) -> tuple:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    freqs = torch.outer(position_ids.float().flatten(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    
    batch_size, num_heads, seq_len, hd = q.shape
    cos = cos.view(batch_size, 1, seq_len, hd)
    sin = sin.view(batch_size, 1, seq_len, hd)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    self_attn_norm_weight: torch.Tensor,
    self_attn_q_weight: torch.Tensor,
    self_attn_k_weight: torch.Tensor,
    self_attn_v_weight: torch.Tensor,
    self_attn_o_weight: torch.Tensor,
    cross_attn_norm_weight: torch.Tensor,
    cross_attn_q_weight: torch.Tensor,
    cross_attn_k_weight: torch.Tensor,
    cross_attn_v_weight: torch.Tensor,
    cross_attn_o_weight: torch.Tensor,
    mlp_norm_weight: torch.Tensor,
    mlp_gate_weight: torch.Tensor,
    mlp_up_weight: torch.Tensor,
    mlp_down_weight: torch.Tensor,
    norm_eps: float,
    rope_theta: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    encoder_seq_len = encoder_hidden_states.shape[1]
    
    num_attention_heads = 16
    num_key_value_heads = 4
    head_dim = 128
    cross_num_attention_heads = 16
    cross_num_key_value_heads = 16
    cross_head_dim = 128
    
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
    
    residual = hidden_states
    
    # Self-Attention Block
    hidden_states = rms_norm(hidden_states, self_attn_norm_weight, norm_eps)
    
    query_states = F.linear(hidden_states, self_attn_q_weight)
    key_states = F.linear(hidden_states, self_attn_k_weight)
    value_states = F.linear(hidden_states, self_attn_v_weight)
    
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    query_states, key_states = apply_rope(query_states, key_states, position_ids, head_dim, rope_theta)
    
    key_states = repeat_kv(key_states, num_attention_heads // num_key_value_heads)
    value_states = repeat_kv(value_states, num_attention_heads // num_key_value_heads)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
    
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_attention_heads * head_dim)
    attn_output = F.linear(attn_output, self_attn_o_weight)
    
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # Cross-Attention Block
    hidden_states = rms_norm(hidden_states, cross_attn_norm_weight, norm_eps)
    
    query_states = F.linear(hidden_states, cross_attn_q_weight)
    key_states = F.linear(encoder_hidden_states, cross_attn_k_weight)
    value_states = F.linear(encoder_hidden_states, cross_attn_v_weight)
    
    query_states = query_states.view(batch_size, seq_len, cross_num_attention_heads, cross_head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, encoder_seq_len, cross_num_key_value_heads, cross_head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, encoder_seq_len, cross_num_key_value_heads, cross_head_dim).transpose(1, 2)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(cross_head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, cross_num_attention_heads * cross_head_dim)
    attn_output = F.linear(attn_output, cross_attn_o_weight)
    
    hidden_states = residual + attn_output
    residual = hidden_states
    
    # MLP Block
    hidden_states = rms_norm(hidden_states, mlp_norm_weight, norm_eps)
    
    gate = F.linear(hidden_states, mlp_gate_weight)
    up = F.linear(hidden_states, mlp_up_weight)
    hidden_states = F.silu(gate) * up
    hidden_states = F.linear(hidden_states, mlp_down_weight)
    
    output = residual + hidden_states
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.122748 ms |
| - | Scoring Baseline | 0.500000 | 0.588518 ms |
| - | Reference Implementation | 0.217434 | 1.766130 ms |
