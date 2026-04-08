## Description

Backward pass for SAM-HQ vision attention with 2D decomposed relative positional embeddings. Computes gradients through output projection, attention mechanism, softmax, relative position bias, and QKV projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, height, width, hidden_size] | float32 |
| hidden_states | [batch_size, height, width, hidden_size] | float32 |
| qkv_weight | [qkv_out_size, hidden_size] | float32 |
| qkv_bias | [qkv_out_size] | float32 |
| proj_weight | [hidden_size, hidden_size] | float32 |
| proj_bias | [hidden_size] | float32 |
| rel_pos_h | [rel_pos_h_size, head_dim] | float32 |
| rel_pos_w | [rel_pos_w_size, head_dim] | float32 |
| scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, height, width, hidden_size] | float32 |
| grad_qkv_weight | [qkv_out_size, hidden_size] | float32 |
| grad_qkv_bias | [qkv_out_size] | float32 |
| grad_proj_weight | [hidden_size, hidden_size] | float32 |
| grad_proj_bias | [hidden_size] | float32 |
| grad_rel_pos_h | [rel_pos_h_size, head_dim] | float32 |
| grad_rel_pos_w | [rel_pos_w_size, head_dim] | float32 |

| # | hidden_size | num_attention_heads | head_dim | qkv_out_size | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 768 | 12 | 64 | 2304 | 1 | 48 | 48 | 0.9832 | 0.0439 |
| 2 | 768 | 12 | 64 | 2304 | 1 | 7 | 7 | 0.1711 | 0.0011 |
| 3 | 768 | 12 | 64 | 2304 | 1 | 32 | 32 | 0.3615 | 0.0131 |
| 4 | 768 | 12 | 64 | 2304 | 8 | 32 | 32 | 1.5336 | 0.1018 |
| 5 | 768 | 12 | 64 | 2304 | 4 | 28 | 28 | 0.7552 | 0.0354 |
| 6 | 768 | 12 | 64 | 2304 | 4 | 64 | 64 | 8.9858 | 0.4593 |
| 7 | 768 | 12 | 64 | 2304 | 4 | 32 | 32 | 0.8875 | 0.0511 |
| 8 | 768 | 12 | 64 | 2304 | 1 | 16 | 16 | 0.2190 | 0.0026 |
| 9 | 768 | 12 | 64 | 2304 | 2 | 32 | 32 | 1.2584 | 0.0258 |
| 10 | 768 | 12 | 64 | 2304 | 1 | 56 | 56 | 3.9214 | 0.0729 |
| 11 | 768 | 12 | 64 | 2304 | 1 | 14 | 14 | 0.4003 | 0.0020 |
| 12 | 768 | 12 | 64 | 2304 | 2 | 14 | 14 | 0.4263 | 0.0036 |
| 13 | 768 | 12 | 64 | 2304 | 2 | 48 | 48 | 4.4364 | 0.0875 |
| 14 | 768 | 12 | 64 | 2304 | 8 | 14 | 14 | 0.6307 | 0.0132 |
| 15 | 768 | 12 | 64 | 2304 | 64 | 7 | 7 | 0.8459 | 0.0237 |
| 16 | 768 | 12 | 64 | 2304 | 32 | 16 | 16 | 1.7861 | 0.0698 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    height = axes_and_scalars["height"]
    width = axes_and_scalars["width"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_attention_heads = axes_and_scalars["num_attention_heads"]
    head_dim = axes_and_scalars["head_dim"]
    qkv_out_size = hidden_size * 3
    rel_pos_h_size = height * 2 - 1
    rel_pos_w_size = width * 2 - 1

    scale = 1.0 / (head_dim ** 0.5)

    # Realistic weight initialization (Xavier/Kaiming-style)
    qkv_weight = torch.randn(qkv_out_size, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
    qkv_bias = torch.zeros(qkv_out_size, device=device)
    proj_weight = torch.randn(hidden_size, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
    proj_bias = torch.zeros(hidden_size, device=device)

    # Relative position embeddings at small scale (like learned embeddings init)
    rel_pos_h = torch.randn(rel_pos_h_size, head_dim, device=device) * 0.02
    rel_pos_w = torch.randn(rel_pos_w_size, head_dim, device=device) * 0.02

    # Input hidden states at small scale (like outputs of a layer norm)
    hidden_states = torch.randn(batch_size, height, width, hidden_size, device=device) * 0.1

    # Unit-scale grad_output
    grad_output = torch.randn(batch_size, height, width, hidden_size, device=device)

    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "proj_weight": proj_weight,
        "proj_bias": proj_bias,
        "rel_pos_h": rel_pos_h,
        "rel_pos_w": rel_pos_w,
        "scale": scale,
    }

def get_rel_pos_interpolated(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Interpolate relative positional embeddings."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    rel_pos_resized = F.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
        size=max_rel_dist,
        mode="linear",
    )
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    
    q_coords = torch.arange(q_size, dtype=rel_pos.dtype, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, dtype=rel_pos.dtype, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    
    return rel_pos_resized[relative_coords.long()]

def get_rel_pos_interpolated_backward(q_size: int, k_size: int, rel_pos: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Backward pass for relative position interpolation."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    q_coords = torch.arange(q_size, dtype=rel_pos.dtype, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, dtype=rel_pos.dtype, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    indices = relative_coords.long()
    
    grad_rel_pos_resized = torch.zeros(
        max_rel_dist, rel_pos.shape[1],
        dtype=grad_output.dtype,
        device=grad_output.device
    )
    
    grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
    indices_flat = indices.reshape(-1)
    grad_rel_pos_resized.index_add_(0, indices_flat, grad_output_flat)
    
    grad_rel_pos_resized = grad_rel_pos_resized.permute(1, 0).reshape(1, -1, max_rel_dist)
    
    grad_rel_pos_input = F.interpolate(
        grad_rel_pos_resized,
        size=rel_pos.shape[0],
        mode="linear",
    )
    
    grad_rel_pos = grad_rel_pos_input.reshape(-1, rel_pos.shape[0]).permute(1, 0)
    
    scale_factor = rel_pos.shape[0] / max_rel_dist
    grad_rel_pos = grad_rel_pos * scale_factor
    
    return grad_rel_pos

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    scale: float,
):
    """Backward pass for SAM-HQ vision attention with relative position."""
    batch_size, height, width, hidden_size = hidden_states.shape
    num_attention_heads = 12
    head_dim = hidden_size // num_attention_heads
    
    # Forward pass recomputation for saved tensors
    hidden_states_flat = hidden_states.reshape(batch_size * height * width, hidden_size)
    qkv_out = F.linear(hidden_states_flat, qkv_weight, qkv_bias)
    qkv_out = qkv_out.reshape(batch_size, height * width, 3, num_attention_heads, head_dim)
    qkv_out = qkv_out.permute(2, 0, 3, 1, 4)
    
    query, key, value = qkv_out.reshape(3, batch_size * num_attention_heads, height * width, head_dim).unbind(0)
    
    attn_weights = (query * scale) @ key.transpose(-2, -1)
    
    rel_pos_h_interp = get_rel_pos_interpolated(height, height, rel_pos_h)
    rel_pos_w_interp = get_rel_pos_interpolated(width, width, rel_pos_w)
    
    query_2d = query.reshape(batch_size, num_attention_heads, height, width, head_dim)
    query_2d = query_2d.reshape(batch_size * num_attention_heads, height, width, head_dim)
    
    rel_h = torch.einsum("bhwc,hkc->bhwk", query_2d, rel_pos_h_interp)
    rel_w = torch.einsum("bhwc,wkc->bhwk", query_2d, rel_pos_w_interp)
    
    rel_pos_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    rel_pos_bias = rel_pos_bias.reshape(batch_size, num_attention_heads, height * width, height * width)
    
    attn_weights_reshaped = attn_weights.reshape(batch_size, num_attention_heads, height * width, height * width)
    attn_weights_reshaped = attn_weights_reshaped + rel_pos_bias
    attn_weights = attn_weights_reshaped.reshape(batch_size * num_attention_heads, height * width, height * width)
    
    attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_probs = attn_weights_softmax
    
    attn_output = attn_probs @ value
    attn_output = attn_output.reshape(batch_size, num_attention_heads, height, width, head_dim)
    attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, hidden_size)
    attn_output_flat = attn_output.reshape(batch_size * height * width, hidden_size)
    
    # Backward pass
    grad_output_flat = grad_output.reshape(batch_size * height * width, hidden_size)
    
    grad_proj_weight = grad_output_flat.t() @ attn_output_flat
    grad_proj_bias = grad_output_flat.sum(dim=0)
    
    grad_attn_output_flat = grad_output_flat @ proj_weight
    grad_attn_output = grad_attn_output_flat.reshape(batch_size, height, width, hidden_size)
    
    grad_attn_output = grad_attn_output.reshape(batch_size, height, width, num_attention_heads, head_dim)
    grad_attn_output = grad_attn_output.permute(0, 3, 1, 2, 4)
    grad_attn_output = grad_attn_output.reshape(batch_size * num_attention_heads, height * width, head_dim)
    
    grad_attn_probs = grad_attn_output @ value.transpose(-2, -1)
    grad_value = attn_probs.transpose(-2, -1) @ grad_attn_output
    
    grad_attn_weights_softmax = grad_attn_probs
    
    sum_grad = (grad_attn_weights_softmax * attn_weights_softmax).sum(dim=-1, keepdim=True)
    grad_attn_weights = attn_weights_softmax * (grad_attn_weights_softmax - sum_grad)
    
    grad_rel_pos_bias = grad_attn_weights.reshape(
        batch_size, num_attention_heads, height * width, height * width
    )
    
    grad_query = (grad_attn_weights @ key) * scale
    grad_key = grad_attn_weights.transpose(-2, -1) @ (query * scale)
    
    grad_rel_pos_bias_5d = grad_rel_pos_bias.reshape(
        batch_size * num_attention_heads, height, width, height, width
    )
    
    grad_rel_h = grad_rel_pos_bias_5d.sum(dim=-1)
    grad_rel_w = grad_rel_pos_bias_5d.sum(dim=-2)
    
    grad_query_from_rel_h = torch.einsum("bhwk,hkc->bhwc", grad_rel_h, rel_pos_h_interp)
    grad_rel_pos_h_interp = torch.einsum("bhwc,bhwk->hkc", query_2d, grad_rel_h)
    
    grad_query_from_rel_w = torch.einsum("bhwk,wkc->bhwc", grad_rel_w, rel_pos_w_interp)
    grad_rel_pos_w_interp = torch.einsum("bhwc,bhwk->wkc", query_2d, grad_rel_w)
    
    grad_query_2d = grad_query_from_rel_h + grad_query_from_rel_w
    
    grad_query_from_relpos = grad_query_2d.reshape(
        batch_size * num_attention_heads, height * width, head_dim
    )
    grad_query = grad_query + grad_query_from_relpos
    
    grad_rel_pos_h = get_rel_pos_interpolated_backward(
        height, height, rel_pos_h, grad_rel_pos_h_interp
    )
    
    grad_rel_pos_w = get_rel_pos_interpolated_backward(
        width, width, rel_pos_w, grad_rel_pos_w_interp
    )
    
    grad_qkv = torch.stack([grad_query, grad_key, grad_value], dim=0)
    grad_qkv = grad_qkv.reshape(3, batch_size, num_attention_heads, height * width, head_dim)
    grad_qkv = grad_qkv.permute(1, 3, 0, 2, 4)
    grad_qkv_flat = grad_qkv.reshape(batch_size * height * width, 3 * hidden_size)
    
    grad_hidden_states_flat = grad_qkv_flat @ qkv_weight
    grad_hidden_states = grad_hidden_states_flat.reshape(batch_size, height, width, hidden_size)
    
    grad_qkv_weight = grad_qkv_flat.t() @ hidden_states_flat
    grad_qkv_bias = grad_qkv_flat.sum(dim=0)
    
    return (
        grad_hidden_states,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_proj_weight,
        grad_proj_bias,
        grad_rel_pos_h,
        grad_rel_pos_w,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.022247 ms |
| - | Scoring Baseline | 0.500000 | 0.949712 ms |
| - | Reference Implementation | 0.281125 | 2.556783 ms |
