## Description

Backward pass for Gemma3 attention with Q/K RMSNorm, RoPE, and soft-capping. Computes gradients for hidden_states, Q/K/V/O projection weights, and Q/K norm weights.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | float32 |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| cos | [batch_size, seq_len, head_dim] | float32 |
| sin | [batch_size, seq_len, head_dim] | float32 |
| q_weight | [q_proj_out, hidden_size] | float32 |
| k_weight | [kv_proj_out, hidden_size] | float32 |
| v_weight | [kv_proj_out, hidden_size] | float32 |
| o_weight | [hidden_size, q_proj_out] | float32 |
| q_norm_weight | [head_dim] | float32 |
| k_norm_weight | [head_dim] | float32 |
| scaling | scalar | float32 |
| attn_logit_softcapping | scalar | float32 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| grad_q_weight | [q_proj_out, hidden_size] | float32 |
| grad_k_weight | [kv_proj_out, hidden_size] | float32 |
| grad_v_weight | [kv_proj_out, hidden_size] | float32 |
| grad_o_weight | [hidden_size, q_proj_out] | float32 |
| grad_q_norm_weight | [head_dim] | float32 |
| grad_k_norm_weight | [head_dim] | float32 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_proj_out | kv_proj_out | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 4 | 128 | 0.5736 | 0.0391 |
| 2 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 4 | 1024 | 2.7710 | 0.3845 |
| 3 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 2048 | 2.1773 | 0.2352 |
| 4 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 2 | 512 | 0.9052 | 0.0858 |
| 5 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 16 | 256 | 1.9619 | 0.3205 |
| 6 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 64 | 256 | 6.3241 | 1.2809 |
| 7 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 2 | 541 | 1.2208 | 0.0912 |
| 8 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 8 | 512 | 2.2342 | 0.3419 |
| 9 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 128 | 0.4588 | 0.0185 |
| 10 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 512 | 0.6518 | 0.0431 |
| 11 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 32 | 128 | 1.8992 | 0.3099 |
| 12 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 1 | 256 | 0.5199 | 0.0204 |
| 13 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 2 | 1024 | 1.6246 | 0.1925 |
| 14 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 2 | 2048 | 3.8809 | 0.4699 |
| 15 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 4 | 256 | 0.8301 | 0.0804 |
| 16 | 3072 | 24 | 8 | 128 | 3072 | 1024 | 8 | 128 | 0.8177 | 0.0778 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate realistic inputs for backward pass by running the forward pass."""
    # Extract axes
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_attention_heads = axes_and_scalars["num_attention_heads"]
    num_key_value_heads = axes_and_scalars["num_key_value_heads"]
    head_dim = axes_and_scalars["head_dim"]

    # Extract scalar inputs
    scaling = 0.08838834764831845
    attn_logit_softcapping = 50.0
    rms_norm_eps = 1e-6

    q_proj_out = num_attention_heads * head_dim
    kv_proj_out = num_key_value_heads * head_dim

    with torch.no_grad():
        # --- Generate realistic weights ---
        # Kaiming uniform scale for linear layers: sqrt(2 / fan_in)
        q_weight = torch.randn(q_proj_out, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        k_weight = torch.randn(kv_proj_out, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        v_weight = torch.randn(kv_proj_out, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        o_weight = torch.randn(hidden_size, q_proj_out, device=device) * (2.0 / q_proj_out) ** 0.5

        # RMSNorm weights initialized near 0 (so effective weight is ~1.0, since code uses 1 + weight)
        q_norm_weight = torch.randn(head_dim, device=device) * 0.02
        k_norm_weight = torch.randn(head_dim, device=device) * 0.02

        # --- Generate realistic hidden states (unit-variance, like post-LayerNorm) ---
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device) * (1.0 / hidden_size ** 0.5)

        # --- Generate RoPE cos/sin from actual position embeddings ---
        # Gemma3 uses rope_theta=10000.0 by default
        rope_theta = 10000.0
        half_dim = head_dim // 2
        freq_seq = torch.arange(0, half_dim, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))  # [half_dim]
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)  # [seq_len]
        freqs = torch.outer(position_ids, inv_freq)  # [seq_len, half_dim]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        cos_vals = emb.cos()  # [seq_len, head_dim]
        sin_vals = emb.sin()  # [seq_len, head_dim]
        # Broadcast to [batch_size, seq_len, head_dim]
        cos = cos_vals.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        sin = sin_vals.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # --- Generate grad_output at unit scale ---
        grad_output = torch.randn(batch_size, seq_len, hidden_size, device=device)

    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "cos": cos,
        "sin": sin,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "o_weight": o_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "scaling": scaling,
        "attn_logit_softcapping": attn_logit_softcapping,
        "rms_norm_eps": rms_norm_eps,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    scaling: float,
    attn_logit_softcapping: float,
    rms_norm_eps: float,
):
    # Constants
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # === FORWARD RECOMPUTATION ===
    # Q, K, V projections
    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)
    
    # Reshape to multi-head format
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # RMSNorm for Q
    q_variance = query_states.pow(2).mean(-1, keepdim=True)
    q_normed = query_states * torch.rsqrt(q_variance + rms_norm_eps)
    query_states_normalized = q_normed * (1.0 + q_norm_weight)
    
    # RMSNorm for K
    k_variance = key_states.pow(2).mean(-1, keepdim=True)
    k_normed = key_states * torch.rsqrt(k_variance + rms_norm_eps)
    key_states_normalized = k_normed * (1.0 + k_norm_weight)
    
    # Apply RoPE
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    q_half_dim = head_dim // 2
    q1 = query_states_normalized[..., :q_half_dim]
    q2 = query_states_normalized[..., q_half_dim:]
    q_rotated_half = torch.cat((-q2, q1), dim=-1)
    query_states_rope = (query_states_normalized * cos_expanded) + (q_rotated_half * sin_expanded)
    
    k1 = key_states_normalized[..., :q_half_dim]
    k2 = key_states_normalized[..., q_half_dim:]
    k_rotated_half = torch.cat((-k2, k1), dim=-1)
    key_states_rope = (key_states_normalized * cos_expanded) + (k_rotated_half * sin_expanded)
    
    # Repeat KV for GQA
    key_states_repeated = key_states_rope[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states_repeated = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    
    # Attention scores
    attn_weights = torch.matmul(query_states_rope, key_states_repeated.transpose(2, 3)) * scaling
    
    # Soft-capping
    attn_weights_for_tanh = attn_weights / attn_logit_softcapping
    attn_weights_tanh = torch.tanh(attn_weights_for_tanh)
    attn_weights_capped = attn_weights_tanh * attn_logit_softcapping
    
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
    attn_weights_capped = attn_weights_capped.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    attn_weights_softmax = F.softmax(attn_weights_capped, dim=-1, dtype=torch.float32)
    
    # Attention output
    attn_output = torch.matmul(attn_weights_softmax, value_states_repeated)
    
    # Reshape
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output_flat = attn_output.view(batch_size, seq_len, num_attention_heads * head_dim)
    
    # === BACKWARD PASS ===
    # Gradient of output projection
    grad_attn_output_flat = F.linear(grad_output, o_weight.t())
    grad_o_weight = grad_output.view(-1, hidden_size).t() @ attn_output_flat.view(-1, num_attention_heads * head_dim)
    
    # Reshape gradient back to multi-head format
    grad_attn_output = grad_attn_output_flat.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    # Gradient through attention matmul
    grad_attn_weights_dropped = torch.matmul(grad_attn_output, value_states_repeated.transpose(2, 3))
    grad_value_states_repeated = torch.matmul(attn_weights_softmax.transpose(2, 3), grad_attn_output)
    
    # Gradient through softmax
    sum_grad = (grad_attn_weights_dropped * attn_weights_softmax).sum(dim=-1, keepdim=True)
    grad_attn_weights = attn_weights_softmax * (grad_attn_weights_dropped - sum_grad)
    
    # Gradient through soft-capping
    grad_attn_weights_uncapped = grad_attn_weights * (1.0 - attn_weights_tanh.pow(2))
    
    # Gradient through attention score computation
    grad_attn_weights_scaled = grad_attn_weights_uncapped * scaling
    grad_query_states_rope = torch.matmul(grad_attn_weights_scaled, key_states_repeated)
    grad_key_states_repeated = torch.matmul(grad_attn_weights_scaled.transpose(2, 3), query_states_rope)
    
    # Gradient through KV repetition (GQA)
    grad_key_states_rope = grad_key_states_repeated.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    grad_value_states = grad_value_states_repeated.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    
    # Gradient through RoPE for Q
    grad_q_from_cos = grad_query_states_rope * cos_expanded
    grad_q_rotated = grad_query_states_rope * sin_expanded
    grad_q_rot_1 = grad_q_rotated[..., :q_half_dim]
    grad_q_rot_2 = grad_q_rotated[..., q_half_dim:]
    grad_q_from_sin = torch.cat((grad_q_rot_2, -grad_q_rot_1), dim=-1)
    grad_query_states_normalized = grad_q_from_cos + grad_q_from_sin
    
    # Gradient through RoPE for K
    grad_k_from_cos = grad_key_states_rope * cos_expanded
    grad_k_rotated = grad_key_states_rope * sin_expanded
    grad_k_rot_1 = grad_k_rotated[..., :q_half_dim]
    grad_k_rot_2 = grad_k_rotated[..., q_half_dim:]
    grad_k_from_sin = torch.cat((grad_k_rot_2, -grad_k_rot_1), dim=-1)
    grad_key_states_normalized = grad_k_from_cos + grad_k_from_sin
    
    # Gradient through Q RMSNorm
    grad_q_normed = grad_query_states_normalized * (1.0 + q_norm_weight)
    grad_q_norm_weight = (grad_query_states_normalized * q_normed).sum(dim=(0, 1, 2))
    
    rsqrt_q_var = torch.rsqrt(q_variance + rms_norm_eps)
    grad_q_var = -0.5 * (grad_q_normed * query_states).sum(dim=-1, keepdim=True) * rsqrt_q_var.pow(3)
    grad_query_states = grad_q_normed * rsqrt_q_var + 2.0 * query_states * grad_q_var / head_dim
    
    # Gradient through K RMSNorm
    grad_k_normed = grad_key_states_normalized * (1.0 + k_norm_weight)
    grad_k_norm_weight = (grad_key_states_normalized * k_normed).sum(dim=(0, 1, 2))
    
    rsqrt_k_var = torch.rsqrt(k_variance + rms_norm_eps)
    grad_k_var = -0.5 * (grad_k_normed * key_states).sum(dim=-1, keepdim=True) * rsqrt_k_var.pow(3)
    grad_key_states = grad_k_normed * rsqrt_k_var + 2.0 * key_states * grad_k_var / head_dim
    
    # Reshape gradients back
    grad_query_states = grad_query_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    grad_key_states = grad_key_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    grad_value_states = grad_value_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    # Gradient through Q/K/V projections
    grad_hidden_states_q = F.linear(grad_query_states, q_weight.t())
    grad_q_weight = grad_query_states.view(-1, num_attention_heads * head_dim).t() @ hidden_states.view(-1, hidden_size)
    
    grad_hidden_states_k = F.linear(grad_key_states, k_weight.t())
    grad_k_weight = grad_key_states.view(-1, num_key_value_heads * head_dim).t() @ hidden_states.view(-1, hidden_size)
    
    grad_hidden_states_v = F.linear(grad_value_states, v_weight.t())
    grad_v_weight = grad_value_states.view(-1, num_key_value_heads * head_dim).t() @ hidden_states.view(-1, hidden_size)
    
    # Sum gradients for hidden_states
    grad_hidden_states = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v
    
    return (
        grad_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_o_weight,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.134207 ms |
| - | Scoring Baseline | 0.500000 | 1.358134 ms |
| - | Reference Implementation | 0.279032 | 3.305604 ms |
