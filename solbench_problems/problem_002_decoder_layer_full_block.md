## Description

Complete LLaMA decoder layer block fusing: input RMSNorm, multi-head GQA attention with RoPE, attention residual, post-attention RMSNorm, SwiGLU MLP, and MLP residual. This represents the atomic computational unit repeated across all layers.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| cos | [batch_size, seq_len, head_dim] | float32 |
| sin | [batch_size, seq_len, head_dim] | float32 |
| attention_mask | [batch_size, one, seq_len, seq_len] | float32 |
| input_layernorm_weight | [hidden_size] | float32 |
| q_proj_weight | [qkv_head_dim, hidden_size] | float32 |
| k_proj_weight | [kv_head_dim, hidden_size] | float32 |
| v_proj_weight | [kv_head_dim, hidden_size] | float32 |
| o_proj_weight | [hidden_size, qkv_head_dim] | float32 |
| post_attention_layernorm_weight | [hidden_size] | float32 |
| gate_proj_weight | [intermediate_size, hidden_size] | float32 |
| up_proj_weight | [intermediate_size, hidden_size] | float32 |
| down_proj_weight | [hidden_size, intermediate_size] | float32 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | float32 |

| # | hidden_size | intermediate_size | num_attention_heads | num_key_value_heads | head_dim | one | qkv_head_dim | kv_head_dim | num_key_value_groups | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 1 | 131 | 0.5128 | 0.0321 |
| 2 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 2 | 1024 | 1.2494 | 0.5126 |
| 3 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 32 | 256 | 3.7835 | 1.9923 |
| 4 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 1 | 512 | 0.6318 | 0.1261 |
| 5 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 16 | 449 | 3.5845 | 1.7597 |
| 6 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 16 | 128 | 1.1530 | 0.4960 |
| 7 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 4 | 128 | 0.6268 | 0.1243 |
| 8 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 1 | 613 | 0.7940 | 0.1514 |
| 9 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 64 | 128 | 3.6468 | 1.9828 |
| 10 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 1 | 256 | 0.5402 | 0.0626 |
| 11 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 2 | 2048 | 2.3075 | 1.0627 |
| 12 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 2 | 128 | 0.5229 | 0.0623 |
| 13 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 2 | 211 | 0.6540 | 0.1028 |
| 14 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 8 | 256 | 1.1622 | 0.4984 |
| 15 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 32 | 128 | 1.9543 | 0.9916 |
| 16 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 8 | 512 | 2.1503 | 1.0058 |
| 17 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 4 | 4096 | 9.7353 | 4.5532 |
| 18 | 4096 | 14336 | 32 | 8 | 128 | 1 | 4096 | 1024 | 4 | 2 | 8192 | 12.8005 | 5.1603 |

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
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    
    batch_size, seq_len, _ = hidden_states.shape
    residual = hidden_states
    
    # === ATTENTION BLOCK ===
    # 1. Input RMSNorm
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = input_layernorm_weight * x.to(hidden_states.dtype)
    
    # 2. QKV projections
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape for attention
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # 3. Apply RoPE
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)
    
    # Rotate half for query
    q1 = query_states[..., : head_dim // 2]
    q2 = query_states[..., head_dim // 2 :]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (q_rotated * sin_expanded)
    
    # Rotate half for key
    k1 = key_states[..., : head_dim // 2]
    k2 = key_states[..., head_dim // 2 :]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (k_rotated * sin_expanded)
    
    # 4. Repeat KV for GQA
    # key_states: [batch, num_kv_heads, seq_len, head_dim]
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # 5. Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, num_attention_heads * head_dim
    )
    
    # 6. Output projection and residual
    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output
    
    # === MLP BLOCK ===
    residual = hidden_states
    
    # 7. Post-attention RMSNorm
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = post_attention_layernorm_weight * x.to(hidden_states.dtype)
    
    # 8. SwiGLU MLP
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    mlp_output = F.linear(gate * up, down_proj_weight)
    
    # 9. MLP residual
    output = residual + mlp_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.453944 ms |
| - | Scoring Baseline | 0.500000 | 1.559616 ms |
| 1st place | AKO4ALL_L2 | 0.429491 | 1.872921 ms |
| - | Reference Implementation | 0.308629 | 2.870794 ms |
