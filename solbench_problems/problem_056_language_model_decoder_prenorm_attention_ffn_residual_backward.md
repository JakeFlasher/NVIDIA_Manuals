## Description

Backward pass for complete decoder transformer layer with pre-norm attention and FFN with residual connections. Computes gradients for all weights and input through RMSNorm, GQA attention with RoPE, and SwiGLU FFN.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| residual | [batch_size, seq_len, hidden_size] | bfloat16 |
| attn_input | [batch_size, seq_len, hidden_size] | bfloat16 |
| query_states | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| query_states_rotated | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| key_states_rotated | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| key_states_repeated | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| value_states_repeated | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| attn_weights | [batch_size, num_heads, seq_len, seq_len] | bfloat16 |
| attn_output | [batch_size, seq_len, num_heads_x_head_dim] | bfloat16 |
| residual2 | [batch_size, seq_len, hidden_size] | bfloat16 |
| ffn_input | [batch_size, seq_len, hidden_size] | bfloat16 |
| gate | [batch_size, seq_len, intermediate_size] | bfloat16 |
| up | [batch_size, seq_len, intermediate_size] | bfloat16 |
| silu_up | [batch_size, seq_len, intermediate_size] | bfloat16 |
| swiglu_output | [batch_size, seq_len, intermediate_size] | bfloat16 |
| input_ln_weight | [hidden_size] | bfloat16 |
| q_weight | [num_heads_x_head_dim, hidden_size] | bfloat16 |
| k_weight | [num_kv_heads_x_head_dim, hidden_size] | bfloat16 |
| v_weight | [num_kv_heads_x_head_dim, hidden_size] | bfloat16 |
| o_weight | [hidden_size, num_heads_x_head_dim] | bfloat16 |
| post_attn_ln_weight | [hidden_size] | bfloat16 |
| gate_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_weight | [hidden_size, intermediate_size] | bfloat16 |
| variance1 | [batch_size, seq_len, 1] | float32 |
| variance2 | [batch_size, seq_len, 1] | float32 |
| hidden_states_normalized1 | [batch_size, seq_len, hidden_size] | float32 |
| hidden_states_normalized2 | [batch_size, seq_len, hidden_size] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_input | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_input_ln_weight | [hidden_size] | float32 |
| grad_q_weight | [num_heads_x_head_dim, hidden_size] | bfloat16 |
| grad_k_weight | [num_kv_heads_x_head_dim, hidden_size] | bfloat16 |
| grad_v_weight | [num_kv_heads_x_head_dim, hidden_size] | bfloat16 |
| grad_o_weight | [hidden_size, num_heads_x_head_dim] | bfloat16 |
| grad_post_attn_ln_weight | [hidden_size] | float32 |
| grad_gate_weight | [intermediate_size, hidden_size] | bfloat16 |
| grad_up_weight | [intermediate_size, hidden_size] | bfloat16 |
| grad_down_weight | [hidden_size, intermediate_size] | bfloat16 |

| # | hidden_size | num_heads | num_kv_heads | head_dim | intermediate_size | num_heads_x_head_dim | num_kv_heads_x_head_dim | half_head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 1 | 256 | 0.8206 | 0.1634 |
| 2 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 4 | 293 | 1.7694 | 0.7477 |
| 3 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 8 | 256 | 2.4994 | 1.3046 |
| 4 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 4 | 128 | 1.1060 | 0.3250 |
| 5 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 2 | 512 | 1.6873 | 0.6584 |
| 6 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 8 | 373 | 3.5090 | 1.9086 |
| 7 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 1 | 613 | 1.2427 | 0.3957 |
| 8 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 32 | 256 | 8.6170 | 5.2172 |
| 9 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 1 | 128 | 0.7678 | 0.1534 |
| 10 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 2 | 1024 | 2.9553 | 1.3402 |
| 11 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 16 | 449 | 8.1430 | 4.6066 |
| 12 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 2 | 211 | 1.0546 | 0.2687 |
| 13 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 2 | 691 | 2.1133 | 0.8941 |
| 14 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 1 | 1024 | 1.8165 | 0.6703 |
| 15 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 32 | 128 | 4.3180 | 2.5969 |
| 16 | 5120 | 32 | 8 | 160 | 14336 | 5120 | 1280 | 80 | 4 | 256 | 1.6020 | 0.6525 |

```python
import torch
import torch.nn.functional as F
import math

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor x. cos/sin shape: (batch, seq, head_dim), x shape: (batch, heads, seq, head_dim)."""
    cos_expanded = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin_expanded = sin.unsqueeze(1)
    return (x * cos_expanded) + (rotate_half(x) * sin_expanded)

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate realistic inputs by running the forward pass to produce consistent intermediates."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_heads = axes_and_scalars["num_heads"]
    num_kv_heads = axes_and_scalars["num_kv_heads"]
    head_dim = axes_and_scalars["head_dim"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    eps = 1e-6
    num_key_value_groups = num_heads // num_kv_heads

    with torch.no_grad():
        # ---- Weights with realistic initialization ----
        # RMSNorm weights near 1.0
        input_ln_weight = (1.0 + 0.02 * torch.randn(hidden_size, device=device)).to(torch.bfloat16)
        post_attn_ln_weight = (1.0 + 0.02 * torch.randn(hidden_size, device=device)).to(torch.bfloat16)

        # Linear projection weights with Kaiming-like init
        q_weight = (torch.randn(num_heads * head_dim, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5).to(torch.bfloat16)
        k_weight = (torch.randn(num_kv_heads * head_dim, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5).to(torch.bfloat16)
        v_weight = (torch.randn(num_kv_heads * head_dim, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5).to(torch.bfloat16)
        o_weight = (torch.randn(hidden_size, num_heads * head_dim, device=device) * (2.0 / (num_heads * head_dim)) ** 0.5).to(torch.bfloat16)
        gate_weight = (torch.randn(intermediate_size, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5).to(torch.bfloat16)
        up_weight = (torch.randn(intermediate_size, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5).to(torch.bfloat16)
        down_weight = (torch.randn(hidden_size, intermediate_size, device=device) * (2.0 / intermediate_size) ** 0.5).to(torch.bfloat16)

        # ---- RoPE cos/sin (actual trigonometric values) ----
        # Mistral-style RoPE with base frequency
        rope_base = 1000000.0
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        cos = emb.cos().unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)  # (batch, seq, head_dim)
        sin = emb.sin().unsqueeze(0).expand(batch_size, -1, -1).to(torch.bfloat16)

        # ---- Input residual (unit-scale) ----
        residual = (torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.02).to(torch.bfloat16)

        # ---- RMSNorm 1: residual -> attn_input ----
        residual_fp32 = residual.to(torch.float32)
        variance1 = residual_fp32.pow(2).mean(-1, keepdim=True)  # (batch, seq, 1)
        hidden_states_normalized1 = residual_fp32 * torch.rsqrt(variance1 + eps)  # (batch, seq, hidden)
        attn_input = (hidden_states_normalized1 * input_ln_weight.to(torch.float32)).to(torch.bfloat16)

        # ---- QKV projections ----
        query_states = F.linear(attn_input, q_weight).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = F.linear(attn_input, k_weight).reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = F.linear(attn_input, v_weight).reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # ---- Apply RoPE ----
        query_states_rotated = apply_rotary_pos_emb(query_states, cos, sin)
        key_states_rotated = apply_rotary_pos_emb(key_states, cos, sin)

        # ---- GQA: repeat KV heads ----
        key_states_repeated = key_states_rotated.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)
        value_states_repeated = value_states.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)

        # ---- Scaled dot-product attention ----
        attn_logits = torch.matmul(query_states_rotated, key_states_repeated.transpose(2, 3)) / math.sqrt(head_dim)
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = torch.softmax(attn_logits.to(torch.float32), dim=-1).to(torch.bfloat16)

        # ---- Attention output ----
        attn_out_heads = torch.matmul(attn_weights, value_states_repeated)  # (batch, heads, seq, head_dim)
        attn_output = attn_out_heads.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

        # ---- Output projection + residual ----
        residual2 = residual + F.linear(attn_output, o_weight)

        # ---- RMSNorm 2: residual2 -> ffn_input ----
        residual2_fp32 = residual2.to(torch.float32)
        variance2 = residual2_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normalized2 = residual2_fp32 * torch.rsqrt(variance2 + eps)
        ffn_input = (hidden_states_normalized2 * post_attn_ln_weight.to(torch.float32)).to(torch.bfloat16)

        # ---- FFN: SwiGLU ----
        gate = F.linear(ffn_input, gate_weight)
        up = F.linear(ffn_input, up_weight)
        silu_up = F.silu(up.to(torch.float32)).to(torch.bfloat16)
        swiglu_output = gate * silu_up

        # ---- grad_output (unit-scale random) ----
        grad_output = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    return {
        "grad_output": grad_output,
        "residual": residual,
        "attn_input": attn_input,
        "query_states": query_states,
        "key_states": key_states,
        "value_states": value_states,
        "query_states_rotated": query_states_rotated,
        "key_states_rotated": key_states_rotated,
        "key_states_repeated": key_states_repeated,
        "value_states_repeated": value_states_repeated,
        "cos": cos,
        "sin": sin,
        "attn_weights": attn_weights,
        "attn_output": attn_output,
        "residual2": residual2,
        "ffn_input": ffn_input,
        "gate": gate,
        "up": up,
        "silu_up": silu_up,
        "swiglu_output": swiglu_output,
        "input_ln_weight": input_ln_weight,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "o_weight": o_weight,
        "post_attn_ln_weight": post_attn_ln_weight,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
        "variance1": variance1,
        "variance2": variance2,
        "hidden_states_normalized1": hidden_states_normalized1,
        "hidden_states_normalized2": hidden_states_normalized2,
        "eps": eps,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_states_rotated: torch.Tensor,
    key_states_rotated: torch.Tensor,
    key_states_repeated: torch.Tensor,
    value_states_repeated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    residual2: torch.Tensor,
    ffn_input: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    silu_up: torch.Tensor,
    swiglu_output: torch.Tensor,
    input_ln_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    post_attn_ln_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    variance1: torch.Tensor,
    variance2: torch.Tensor,
    hidden_states_normalized1: torch.Tensor,
    hidden_states_normalized2: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    num_kv_heads = 8
    head_dim = 160
    intermediate_size = 14336
    num_key_value_groups = num_heads // num_kv_heads
    
    # ============ Backward through FFN Block ============
    # Gradient through final residual
    grad_residual2 = grad_output
    grad_ffn_output = grad_output
    
    # Gradient through down projection
    grad_swiglu_output = F.linear(grad_ffn_output, down_weight.t())
    grad_down_weight = grad_ffn_output.reshape(-1, hidden_size).t() @ swiglu_output.reshape(-1, intermediate_size)
    
    # Gradient through SwiGLU: gate * silu(up)
    grad_gate = grad_swiglu_output * silu_up
    sigmoid_up = torch.sigmoid(up.to(torch.float32))
    grad_silu = sigmoid_up * (1.0 + up.to(torch.float32) * (1.0 - sigmoid_up))
    grad_up = grad_swiglu_output * gate * grad_silu.to(up.dtype)
    
    # Gradient through gate and up projections
    grad_ffn_input_gate = F.linear(grad_gate, gate_weight.t())
    grad_ffn_input_up = F.linear(grad_up, up_weight.t())
    grad_ffn_input = grad_ffn_input_gate + grad_ffn_input_up
    
    grad_gate_weight = grad_gate.reshape(-1, intermediate_size).t() @ ffn_input.reshape(-1, hidden_size)
    grad_up_weight = grad_up.reshape(-1, intermediate_size).t() @ ffn_input.reshape(-1, hidden_size)
    
    # Gradient through post-attention layernorm
    grad_ffn_input_fp32 = grad_ffn_input.to(torch.float32)
    grad_post_attn_ln_weight = (grad_ffn_input_fp32 * hidden_states_normalized2).sum(dim=[0, 1])
    
    N = hidden_size
    rsqrt_var2 = torch.rsqrt(variance2 + eps)
    grad_normalized2 = grad_ffn_input_fp32 * post_attn_ln_weight.to(torch.float32)
    grad_hidden_states2 = grad_normalized2 * rsqrt_var2
    grad_var2 = -0.5 * (grad_normalized2 * residual2.to(torch.float32)).sum(dim=-1, keepdim=True) * rsqrt_var2.pow(3)
    grad_hidden_states2 = grad_hidden_states2 + (2.0 / N) * residual2.to(torch.float32) * grad_var2
    grad_hidden_states2 = grad_hidden_states2.to(residual2.dtype)
    
    # Accumulate gradient from residual2
    grad_hidden_states_attn = grad_residual2 + grad_hidden_states2
    
    # ============ Backward through Attention Block ============
    # Gradient through attention residual
    grad_residual1 = grad_hidden_states_attn
    grad_attn_output_proj = grad_hidden_states_attn
    
    # Gradient through output projection
    grad_attn_output = F.linear(grad_attn_output_proj, o_weight.t())
    grad_o_weight = grad_attn_output_proj.reshape(-1, hidden_size).t() @ attn_output.reshape(-1, num_heads * head_dim)
    
    # Reshape gradient
    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, num_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(1, 2)
    
    # Gradient through attention output = matmul(attn_weights, value_states_repeated)
    grad_attn_weights = torch.matmul(grad_attn_output, value_states_repeated.transpose(2, 3))
    grad_value_states_repeated = torch.matmul(attn_weights.transpose(2, 3), grad_attn_output)
    
    # Gradient through softmax
    grad_attn_weights_fp32 = grad_attn_weights.to(torch.float32)
    attn_weights_fp32 = attn_weights.to(torch.float32)
    grad_attn_logits = attn_weights_fp32 * (grad_attn_weights_fp32 - (grad_attn_weights_fp32 * attn_weights_fp32).sum(dim=-1, keepdim=True))
    
    # Gradient through scaling
    grad_attn_logits = grad_attn_logits / math.sqrt(head_dim)
    grad_attn_logits = grad_attn_logits.to(query_states_rotated.dtype)
    
    # Gradient through attention matmul: attn_logits = query @ key.T
    grad_query_states_rotated = torch.matmul(grad_attn_logits, key_states_repeated)
    grad_key_states_repeated = torch.matmul(grad_attn_logits.transpose(2, 3), query_states_rotated)
    
    # Gradient through KV repetition (GQA)
    grad_key_states_repeated = grad_key_states_repeated.reshape(
        batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim
    )
    grad_key_states_rotated = grad_key_states_repeated.sum(dim=2)
    
    grad_value_states_repeated = grad_value_states_repeated.reshape(
        batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim
    )
    grad_value_states = grad_value_states_repeated.sum(dim=2)
    
    # Gradient through RoPE
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    grad_query_states = (grad_query_states_rotated * cos_expanded) + (rotate_half(grad_query_states_rotated) * (-sin_expanded))
    grad_key_states = (grad_key_states_rotated * cos_expanded) + (rotate_half(grad_key_states_rotated) * (-sin_expanded))
    
    # Reshape gradients back
    grad_query_states = grad_query_states.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    grad_key_states = grad_key_states.transpose(1, 2).reshape(batch_size, seq_len, num_kv_heads * head_dim)
    grad_value_states = grad_value_states.transpose(1, 2).reshape(batch_size, seq_len, num_kv_heads * head_dim)
    
    # Gradient through QKV projections
    grad_attn_input_q = F.linear(grad_query_states, q_weight.t())
    grad_attn_input_k = F.linear(grad_key_states, k_weight.t())
    grad_attn_input_v = F.linear(grad_value_states, v_weight.t())
    grad_attn_input = grad_attn_input_q + grad_attn_input_k + grad_attn_input_v
    
    grad_q_weight = grad_query_states.reshape(-1, num_heads * head_dim).t() @ attn_input.reshape(-1, hidden_size)
    grad_k_weight = grad_key_states.reshape(-1, num_kv_heads * head_dim).t() @ attn_input.reshape(-1, hidden_size)
    grad_v_weight = grad_value_states.reshape(-1, num_kv_heads * head_dim).t() @ attn_input.reshape(-1, hidden_size)
    
    # Gradient through input layernorm
    grad_attn_input_fp32 = grad_attn_input.to(torch.float32)
    grad_input_ln_weight = (grad_attn_input_fp32 * hidden_states_normalized1).sum(dim=[0, 1])
    
    rsqrt_var1 = torch.rsqrt(variance1 + eps)
    grad_normalized1 = grad_attn_input_fp32 * input_ln_weight.to(torch.float32)
    grad_hidden_states1 = grad_normalized1 * rsqrt_var1
    grad_var1 = -0.5 * (grad_normalized1 * residual.to(torch.float32)).sum(dim=-1, keepdim=True) * rsqrt_var1.pow(3)
    grad_hidden_states1 = grad_hidden_states1 + (2.0 / N) * residual.to(torch.float32) * grad_var1
    grad_hidden_states1 = grad_hidden_states1.to(residual.dtype)
    
    # Accumulate gradient from residual1
    grad_input = grad_residual1 + grad_hidden_states1
    
    return (
        grad_input,
        grad_input_ln_weight,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_o_weight,
        grad_post_attn_ln_weight,
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.810574 ms |
| - | Scoring Baseline | 0.500000 | 2.089142 ms |
| - | Reference Implementation | 0.336452 | 3.245349 ms |
