## Description

Complete residual path through a JointTransformerBlock including dual-stream attention, adaptive normalization, and feedforward with residual connections for SD3.5.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, inner_dim] | float32 |
| encoder_hidden_states | [batch_size, context_len, context_dim] | float32 |
| temb | [batch_size, inner_dim] | float32 |
| norm1_weight | [mod_dim_6x, inner_dim] | float32 |
| norm1_bias | [mod_dim_6x] | float32 |
| norm1_context_weight | [context_mod_dim_6x, inner_dim] | float32 |
| norm1_context_bias | [context_mod_dim_6x] | float32 |
| to_q_weight | [inner_dim, inner_dim] | float32 |
| to_q_bias | [inner_dim] | float32 |
| to_k_weight | [inner_dim, inner_dim] | float32 |
| to_k_bias | [inner_dim] | float32 |
| to_v_weight | [inner_dim, inner_dim] | float32 |
| to_v_bias | [inner_dim] | float32 |
| add_q_proj_weight | [inner_dim, context_dim] | float32 |
| add_q_proj_bias | [inner_dim] | float32 |
| add_k_proj_weight | [inner_dim, context_dim] | float32 |
| add_k_proj_bias | [inner_dim] | float32 |
| add_v_proj_weight | [inner_dim, context_dim] | float32 |
| add_v_proj_bias | [inner_dim] | float32 |
| to_out_weight | [inner_dim, inner_dim] | float32 |
| to_out_bias | [inner_dim] | float32 |
| to_add_out_weight | [context_dim, inner_dim] | float32 |
| to_add_out_bias | [context_dim] | float32 |
| ff_linear1_weight | [ff_inner_dim, inner_dim] | float32 |
| ff_linear1_bias | [ff_inner_dim] | float32 |
| ff_linear2_weight | [inner_dim, ff_inner_dim] | float32 |
| ff_linear2_bias | [inner_dim] | float32 |
| ff_context_linear1_weight | [ff_inner_dim, context_dim] | float32 |
| ff_context_linear1_bias | [ff_inner_dim] | float32 |
| ff_context_linear2_weight | [context_dim, ff_inner_dim] | float32 |
| ff_context_linear2_bias | [context_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output_encoder_hidden_states | [batch_size, context_len, context_dim] | float32 |
| output_hidden_states | [batch_size, seq_len, inner_dim] | float32 |

| # | num_attention_heads | attention_head_dim | inner_dim | context_dim | ff_inner_dim | mod_dim_6x | context_mod_dim_6x | batch_size | seq_len | context_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 1 | 1024 | 77 | 2.6785 | 0.0384 |
| 2 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 1 | 2048 | 77 | 5.5578 | 0.0816 |
| 3 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 32 | 256 | 77 | 16.2236 | 0.3272 |
| 4 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 64 | 128 | 77 | 18.5762 | 0.3829 |
| 5 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 1 | 2053 | 77 | 6.5513 | 0.0818 |
| 6 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 1 | 8192 | 77 | 36.7361 | 0.4903 |
| 7 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 2 | 256 | 154 | 1.9756 | 0.0248 |
| 8 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 2 | 2048 | 77 | 10.1371 | 0.1627 |
| 9 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 1 | 293 | 77 | 1.2520 | 0.0119 |
| 10 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 8 | 256 | 77 | 4.8798 | 0.0821 |
| 11 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 4 | 512 | 77 | 4.6090 | 0.0765 |
| 12 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 16 | 512 | 77 | 15.5146 | 0.3046 |
| 13 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 1 | 128 | 154 | 1.0956 | 0.0083 |
| 14 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 4 | 1024 | 77 | 8.7032 | 0.1522 |
| 15 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 2 | 4096 | 77 | 25.8483 | 0.3783 |
| 16 | 24 | 64 | 1536 | 1152 | 6144 | 9216 | 6912 | 4 | 2048 | 77 | 19.0513 | 0.3251 |

```python
import math

import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    context_len = axes_and_scalars["context_len"]

    dim = 1536
    context_dim = 1152
    ff_inner_dim = 6144

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def xavier(out_f, in_f):
        return torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)

    return {
        # Activation tensors
        "hidden_states": torch.randn(batch_size, seq_len, dim, device=device, generator=g),
        "encoder_hidden_states": torch.randn(batch_size, context_len, context_dim, device=device, generator=g),
        # Time embedding — small magnitude
        "temb": torch.randn(batch_size, dim, device=device, generator=g) * 0.1,
        # Modulation projection weights (NOT norm weights — they project temb to 6*dim)
        "norm1_weight": xavier(6 * dim, dim),
        "norm1_bias": torch.randn(6 * dim, device=device, generator=g),
        "norm1_context_weight": xavier(6 * context_dim, dim),
        "norm1_context_bias": torch.randn(6 * context_dim, device=device, generator=g),
        # Image stream QKV
        "to_q_weight": xavier(dim, dim),
        "to_q_bias": torch.randn(dim, device=device, generator=g),
        "to_k_weight": xavier(dim, dim),
        "to_k_bias": torch.randn(dim, device=device, generator=g),
        "to_v_weight": xavier(dim, dim),
        "to_v_bias": torch.randn(dim, device=device, generator=g),
        # Context stream QKV
        "add_q_proj_weight": xavier(dim, context_dim),
        "add_q_proj_bias": torch.randn(dim, device=device, generator=g),
        "add_k_proj_weight": xavier(dim, context_dim),
        "add_k_proj_bias": torch.randn(dim, device=device, generator=g),
        "add_v_proj_weight": xavier(dim, context_dim),
        "add_v_proj_bias": torch.randn(dim, device=device, generator=g),
        # Output projections
        "to_out_weight": xavier(dim, dim),
        "to_out_bias": torch.randn(dim, device=device, generator=g),
        "to_add_out_weight": xavier(context_dim, dim),
        "to_add_out_bias": torch.randn(context_dim, device=device, generator=g),
        # FF — image stream
        "ff_linear1_weight": xavier(ff_inner_dim, dim),
        "ff_linear1_bias": torch.randn(ff_inner_dim, device=device, generator=g),
        "ff_linear2_weight": xavier(dim, ff_inner_dim),
        "ff_linear2_bias": torch.randn(dim, device=device, generator=g),
        # FF — context stream
        "ff_context_linear1_weight": xavier(ff_inner_dim, context_dim),
        "ff_context_linear1_bias": torch.randn(ff_inner_dim, device=device, generator=g),
        "ff_context_linear2_weight": xavier(context_dim, ff_inner_dim),
        "ff_context_linear2_bias": torch.randn(context_dim, device=device, generator=g),
    }

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    norm1_context_weight: torch.Tensor,
    norm1_context_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_q_bias: torch.Tensor,
    to_k_weight: torch.Tensor,
    to_k_bias: torch.Tensor,
    to_v_weight: torch.Tensor,
    to_v_bias: torch.Tensor,
    add_q_proj_weight: torch.Tensor,
    add_q_proj_bias: torch.Tensor,
    add_k_proj_weight: torch.Tensor,
    add_k_proj_bias: torch.Tensor,
    add_v_proj_weight: torch.Tensor,
    add_v_proj_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    to_add_out_weight: torch.Tensor,
    to_add_out_bias: torch.Tensor,
    ff_linear1_weight: torch.Tensor,
    ff_linear1_bias: torch.Tensor,
    ff_linear2_weight: torch.Tensor,
    ff_linear2_bias: torch.Tensor,
    ff_context_linear1_weight: torch.Tensor,
    ff_context_linear1_bias: torch.Tensor,
    ff_context_linear2_weight: torch.Tensor,
    ff_context_linear2_bias: torch.Tensor,
):
    batch_size = hidden_states.shape[0]
    image_seq_len = hidden_states.shape[1]
    context_seq_len = encoder_hidden_states.shape[1]
    
    dim = 1536
    context_dim = 1152
    num_heads = 24
    head_dim = 64
    scale = head_dim ** -0.5
    
    # AdaLayerNormZero modulation
    norm_hidden_states = F.layer_norm(hidden_states, (dim,), eps=1e-6)
    norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, (context_dim,), eps=1e-6)
    
    # Get modulation parameters from timestep embedding
    temb_silu = F.silu(temb)
    mod_params = F.linear(temb_silu, norm1_weight, norm1_bias)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6, dim=-1)
    
    mod_params_context = F.linear(temb_silu, norm1_context_weight, norm1_context_bias)
    c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = mod_params_context.chunk(6, dim=-1)
    
    # Apply modulation to normalized states
    norm_hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)
    
    # Dual-stream attention
    # Image stream QKV
    q = F.linear(norm_hidden_states, to_q_weight, to_q_bias)
    k = F.linear(norm_hidden_states, to_k_weight, to_k_bias)
    v = F.linear(norm_hidden_states, to_v_weight, to_v_bias)
    
    # Text stream QKV
    c_q = F.linear(norm_encoder_hidden_states, add_q_proj_weight, add_q_proj_bias)
    c_k = F.linear(norm_encoder_hidden_states, add_k_proj_weight, add_k_proj_bias)
    c_v = F.linear(norm_encoder_hidden_states, add_v_proj_weight, add_v_proj_bias)
    
    # Concatenate for joint attention
    q = torch.cat([q, c_q], dim=1)
    k = torch.cat([k, c_k], dim=1)
    v = torch.cat([v, c_v], dim=1)
    
    # Reshape for multi-head attention
    total_seq_len = q.shape[1]
    q = q.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, total_seq_len, -1)
    
    # Split back into image and text streams
    attn_output_img, c_attn_output = attn_output.split([image_seq_len, context_seq_len], dim=1)
    
    # Output projections
    attn_output_img = F.linear(attn_output_img, to_out_weight, to_out_bias)
    c_attn_output = F.linear(c_attn_output, to_add_out_weight, to_add_out_bias)
    
    # Gated residual connection for attention
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output_img
    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * c_attn_output
    
    # Feedforward path
    # Normalize
    norm_hidden_states = F.layer_norm(hidden_states, (dim,), eps=1e-6)
    norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, (context_dim,), eps=1e-6)
    
    # Apply modulation
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)
    
    # Feedforward with GELU-approximate
    ff_hidden = F.linear(norm_hidden_states, ff_linear1_weight, ff_linear1_bias)
    ff_hidden = F.gelu(ff_hidden, approximate='tanh')
    ff_output = F.linear(ff_hidden, ff_linear2_weight, ff_linear2_bias)
    
    c_ff_hidden = F.linear(norm_encoder_hidden_states, ff_context_linear1_weight, ff_context_linear1_bias)
    c_ff_hidden = F.gelu(c_ff_hidden, approximate='tanh')
    c_ff_output = F.linear(c_ff_hidden, ff_context_linear2_weight, ff_context_linear2_bias)
    
    # Gated residual connection for feedforward
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * c_ff_output
    
    return encoder_hidden_states, hidden_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.105190 ms |
| - | Scoring Baseline | 0.500000 | 7.127473 ms |
| - | Reference Implementation | 0.492257 | 7.350030 ms |
