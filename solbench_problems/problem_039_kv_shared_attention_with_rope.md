## Description

KV-Sharing Attention with RoPE for Gemma3n. Implements novel KV-sharing mechanism where later layers reuse key-value states from earlier non-shared layers. Supports both global and sliding window attention patterns with RoPE embeddings, query/key/value normalization, and attention logit soft-capping.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| position_ids | [batch_size, seq_len] | int64 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| q_proj_weight | [qkv_out_dim, hidden_size] | bfloat16 |
| k_proj_weight | [kv_out_dim, hidden_size] | bfloat16 |
| v_proj_weight | [kv_out_dim, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, qkv_out_dim] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| rope_theta | scalar | float32 |
| softcap | scalar | float32 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| key_states_out | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| value_states_out | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | qkv_out_dim | kv_out_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 2 | 256 | 0.2363 | 0.0070 |
| 2 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 1 | 131 | 0.2203 | 0.0023 |
| 3 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 32 | 256 | 0.7479 | 0.1059 |
| 4 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 2 | 512 | 4.6659 | 0.0148 |
| 5 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 16 | 128 | 0.2868 | 0.0256 |
| 6 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 2 | 449 | 0.3100 | 0.0128 |
| 7 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 2 | 1024 | 0.4134 | 0.0339 |
| 8 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 1 | 2048 | 0.4573 | 0.0434 |
| 9 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 64 | 128 | 0.6173 | 0.1012 |
| 10 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 32 | 128 | 0.4031 | 0.0508 |
| 11 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 1 | 1024 | 0.3217 | 0.0171 |
| 12 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 16 | 997 | 2.8515 | 0.2593 |
| 13 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 16 | 256 | 0.4322 | 0.0532 |
| 14 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 4 | 256 | 0.2674 | 0.0136 |
| 15 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 32 | 1163 | 7.4817 | 0.6325 |
| 16 | 2304 | 8 | 1 | 256 | 8 | 2048 | 256 | 64 | 256 | 1.1966 | 0.2114 |

```python
import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_attention_heads = axes_and_scalars["num_attention_heads"]
    num_key_value_heads = axes_and_scalars["num_key_value_heads"]
    head_dim = axes_and_scalars["head_dim"]
    
    qkv_out_dim = num_attention_heads * head_dim
    kv_out_dim = num_key_value_heads * head_dim
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create causal mask
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attention_mask = attention_mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    q_proj_weight = torch.randn(qkv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_weight = torch.randn(kv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_weight = torch.randn(kv_out_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, qkv_out_dim, dtype=torch.bfloat16, device=device) * 0.02
    
    q_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    
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
        "rope_theta": 10000.0,
        "softcap": 30.0,
        "rms_norm_eps": 1e-6
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
    rope_theta: float,
    softcap: float,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 8
    num_key_value_heads = 1
    head_dim = 256
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    # Q projection
    query_states = F.linear(hidden_states, q_proj_weight)  # [batch, seq, num_heads * head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    
    # Q RMSNorm
    q_variance = query_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    query_states = query_states * torch.rsqrt(q_variance + rms_norm_eps)
    query_states = (query_states * q_norm_weight).to(hidden_states.dtype)
    
    # K projection
    key_states = F.linear(hidden_states, k_proj_weight)  # [batch, seq, num_kv_heads * head_dim]
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # K RMSNorm
    k_variance = key_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    key_states = key_states * torch.rsqrt(k_variance + rms_norm_eps)
    key_states = (key_states * k_norm_weight).to(hidden_states.dtype)
    
    # V projection
    value_states = F.linear(hidden_states, v_proj_weight)  # [batch, seq, num_kv_heads * head_dim]
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # V RMSNorm (without scale)
    v_variance = value_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    value_states = (value_states * torch.rsqrt(v_variance + rms_norm_eps)).to(hidden_states.dtype)
    
    # Compute RoPE embeddings
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=hidden_states.device) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [batch, seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq, head_dim]
    cos = emb.cos().to(hidden_states.dtype)
    sin = emb.sin().to(hidden_states.dtype)
    
    # Apply RoPE to Q
    cos_q = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin_q = sin.unsqueeze(2)
    q1 = query_states[..., :head_dim // 2]
    q2 = query_states[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_q) + (q_rotated * sin_q)
    
    # Apply RoPE to K
    cos_k = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin_k = sin.unsqueeze(2)
    k1 = key_states[..., :head_dim // 2]
    k2 = key_states[..., head_dim // 2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_k) + (k_rotated * sin_k)
    
    # Transpose for attention: [batch, heads, seq, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Store KV states for sharing before repeat
    key_states_out = key_states.clone()
    value_states_out = value_states.clone()
    
    # Repeat KV heads for GQA
    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = attn_weights / math.sqrt(head_dim)
    
    # Apply soft-capping
    attn_weights = attn_weights / softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * softcap
    
    # Apply attention mask
    attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output, key_states_out, value_states_out
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.038841 ms |
| - | Scoring Baseline | 0.500000 | 0.636255 ms |
| - | Reference Implementation | 0.333415 | 1.208753 ms |
