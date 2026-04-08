## Description

KV-shared attention with dual RoPE (global and local), Q/K/V normalization, soft-capping, and grouped-query attention. Implements attention where later layers can reuse KV states from earlier layers, with dual rotary position embeddings.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| position_ids | [batch_size, seq_len] | int64 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| q_proj_weight | [q_proj_dim, hidden_size] | bfloat16 |
| k_proj_weight | [kv_proj_dim, hidden_size] | bfloat16 |
| v_proj_weight | [kv_proj_dim, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, q_proj_dim] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| shared_key_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| shared_value_states | [batch_size, num_key_value_heads, seq_len, head_dim] | bfloat16 |
| rope_theta | scalar | float32 |
| softcap | scalar | float32 |
| rms_norm_eps | scalar | float32 |
| use_shared_kv | scalar | bool |

| Name | Shape | Dtype |
| --- | --- | --- |
| attn_output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | num_key_value_groups | q_proj_dim | kv_proj_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 4 | 128 | 0.5716 | 0.0142 |
| 2 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 4 | 1024 | 2.8585 | 0.1356 |
| 3 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 1 | 4096 | 8.6837 | 0.2209 |
| 4 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 32 | 128 | 1.4145 | 0.1107 |
| 5 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 1 | 997 | 1.0952 | 0.0331 |
| 6 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 4 | 541 | 1.5196 | 0.0647 |
| 7 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 8 | 512 | 2.0118 | 0.1213 |
| 8 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 2 | 128 | 0.5585 | 0.0073 |
| 9 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 16 | 512 | 3.6719 | 0.2423 |
| 10 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 32 | 256 | 2.8665 | 0.2280 |
| 11 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 1 | 256 | 0.5300 | 0.0075 |
| 12 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 1 | 131 | 0.5320 | 0.0045 |
| 13 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 4 | 512 | 1.2610 | 0.0609 |
| 14 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 4 | 256 | 0.6872 | 0.0289 |
| 15 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 1 | 1024 | 0.9660 | 0.0342 |
| 16 | 3072 | 32 | 8 | 96 | 4 | 3072 | 768 | 2 | 2048 | 4.8557 | 0.1640 |

```python
import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 3072
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 96

    # Use fixed values for scalars
    rope_theta = 10000.0
    softcap = 30.0
    rms_norm_eps = 1e-6
    use_shared_kv = False

    dtype = torch.bfloat16

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Create causal mask
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).contiguous()

    # Projection weights: Xavier init with 1/sqrt(fan_in) where fan_in is the last dim
    q_proj_weight = torch.randn(num_attention_heads * head_dim, hidden_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    k_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    v_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    o_proj_weight = torch.randn(hidden_size, num_attention_heads * head_dim, dtype=dtype, device=device) / math.sqrt(num_attention_heads * head_dim)

    q_norm_weight = torch.ones(head_dim, dtype=dtype, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=dtype, device=device)

    shared_key_states = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim, dtype=dtype, device=device)
    shared_value_states = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim, dtype=dtype, device=device)

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
        "shared_key_states": shared_key_states,
        "shared_value_states": shared_value_states,
        "rope_theta": rope_theta,
        "softcap": softcap,
        "rms_norm_eps": rms_norm_eps,
        "use_shared_kv": use_shared_kv,
    }

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)

def rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)

def compute_rope_embeddings(x: torch.Tensor, position_ids: torch.Tensor, theta: float, head_dim: int) -> tuple:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)
    sin = emb.sin().to(x.dtype)
    return cos, sin

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

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
    shared_key_states: torch.Tensor,
    shared_value_states: torch.Tensor,
    rope_theta: float,
    softcap: float,
    rms_norm_eps: float,
    use_shared_kv: bool,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 96
    num_key_value_groups = 4
    
    # Query projection: [batch, seq_len, hidden_size] @ [hidden_size, num_heads * head_dim].T
    query_states = F.linear(hidden_states, q_proj_weight)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    
    # Query normalization
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    
    # Compute RoPE embeddings
    cos, sin = compute_rope_embeddings(query_states, position_ids, rope_theta, head_dim)
    
    # Apply RoPE to queries
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    
    if use_shared_kv:
        # Use shared KV states from earlier layer
        key_states = shared_key_states
        value_states = shared_value_states
    else:
        # Compute new KV states
        key_states = F.linear(hidden_states, k_proj_weight)
        key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
        key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        value_states = F.linear(hidden_states, v_proj_weight)
        value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
        value_states = rms_norm_no_scale(value_states, rms_norm_eps)
        value_states = value_states.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
    
    # Repeat KV for grouped-query attention
    key_states_expanded = repeat_kv(key_states, num_key_value_groups)
    value_states_expanded = repeat_kv(value_states, num_key_value_groups)
    
    # Attention computation: Q @ K^T
    attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3))
    
    # Apply soft-capping: softcap * tanh(attn_weights / softcap)
    attn_weights = attn_weights / softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * softcap
    
    # Apply attention mask
    attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output: attn_weights @ V
    attn_output = torch.matmul(attn_weights, value_states_expanded)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.050232 ms |
| - | Scoring Baseline | 0.500000 | 1.465504 ms |
| - | Reference Implementation | 0.497584 | 1.485250 ms |
