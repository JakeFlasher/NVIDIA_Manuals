## Description

Complete Mllama text decoder layer combining self-attention with RoPE, RMSNorm, gated MLP, and residual connections. This is the dominant computational pattern in the text model (32 out of 40 layers). Architecture: (1) RMSNorm + self-attention with RoPE and GQA (32 heads, 8 KV heads) + residual, (2) RMSNorm + gated MLP with SiLU + residual.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| attention_mask | [batch_size, 1, seq_len, seq_len] | bfloat16 |
| q_proj_weight | [qkv_head_dim, hidden_size] | bfloat16 |
| k_proj_weight | [kv_head_dim, hidden_size] | bfloat16 |
| v_proj_weight | [kv_head_dim, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, qkv_head_dim] | bfloat16 |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 |
| input_layernorm_weight | [hidden_size] | bfloat16 |
| post_attention_layernorm_weight | [hidden_size] | bfloat16 |
| rms_norm_eps | scalar | float32 |
| rope_theta | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_heads | num_key_value_heads | head_dim | num_key_value_groups | intermediate_size | qkv_head_dim | kv_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 1 | 131 | 0.3904 | 0.0321 |
| 2 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 2 | 256 | 0.4713 | 0.1249 |
| 3 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 2 | 512 | 0.6794 | 0.2518 |
| 4 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 32 | 256 | 3.1029 | 1.9923 |
| 5 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 8 | 128 | 0.6283 | 0.2482 |
| 6 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 16 | 128 | 0.9843 | 0.4960 |
| 7 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 1 | 256 | 0.3980 | 0.0626 |
| 8 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 1 | 1024 | 0.6707 | 0.2565 |
| 9 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 1 | 4096 | 2.4647 | 1.1386 |
| 10 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 4 | 128 | 0.4605 | 0.1243 |
| 11 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 2 | 211 | 0.5668 | 0.1028 |
| 12 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 4 | 512 | 1.0204 | 0.5031 |
| 13 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 64 | 256 | 6.0076 | 3.9841 |
| 14 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 1 | 8192 | 5.9607 | 2.5803 |
| 15 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 64 | 128 | 3.0214 | 1.9828 |
| 16 | 4096 | 32 | 8 | 128 | 4 | 14336 | 4096 | 1024 | 8 | 256 | 0.9946 | 0.4984 |

```python
import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_theta: float,
) -> torch.Tensor:
    # Constants
    num_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    scaling = head_dim ** -0.5
    
    bsz, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    
    # RMSNorm helper
    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return weight * x.to(input_dtype)
    
    # RoPE helper
    def apply_rope(q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        
        inv_freq_expanded = inv_freq[None, :, None].float().expand(bsz, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        
        q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_embed = (q * cos) + (q_rotated * sin)
        
        k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = (k * cos) + (k_rotated * sin)
        
        return q_embed.to(q.dtype), k_embed.to(k.dtype)
    
    # Repeat KV helper
    def repeat_kv(x: torch.Tensor) -> torch.Tensor:
        batch, n_kv_heads, slen, hdim = x.shape
        if num_key_value_groups == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, num_key_value_groups, slen, hdim)
        return x.reshape(batch, n_kv_heads * num_key_value_groups, slen, hdim)
    
    # Self-attention block with residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)
    
    # Project Q, K, V
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape for multi-head attention
    query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE
    query_states, key_states = apply_rope(query_states, key_states, seq_len)
    
    # Repeat K/V for GQA
    key_states = repeat_kv(key_states)
    value_states = repeat_kv(value_states)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    causal_mask = attention_mask[:, :, :seq_len, :seq_len]
    attn_weights = attn_weights + causal_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, seq_len, -1)
    
    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output
    
    # MLP block with residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)
    
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(gate * up, down_proj_weight)
    hidden_states = residual + hidden_states
    
    return hidden_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.390772 ms |
| - | Scoring Baseline | 0.500000 | 1.098926 ms |
| - | Reference Implementation | 0.280575 | 2.153351 ms |
