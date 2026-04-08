## Description

Backward pass for fused GQA QKV projection with RoPE and QK normalization. Computes gradients through RMS normalization, rotary position embedding, reshape/transpose operations, and linear projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_query | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| grad_key | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| grad_value | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| qkv_weight | [qkv_size, hidden_size] | bfloat16 |
| q_norm_weight | [head_dim] | bfloat16 |
| k_norm_weight | [head_dim] | bfloat16 |
| query_pre_norm | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| key_pre_norm | [batch_size, num_kv_heads, seq_len, head_dim] | bfloat16 |
| q_rstd | [batch_size, num_heads, seq_len, rstd_dim] | bfloat16 |
| k_rstd | [batch_size, num_kv_heads, seq_len, rstd_dim] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_cos | [batch_size, seq_len, head_dim] | bfloat16 |
| grad_sin | [batch_size, seq_len, head_dim] | bfloat16 |
| grad_qkv_weight | [qkv_size, hidden_size] | bfloat16 |
| grad_q_norm_weight | [head_dim] | bfloat16 |
| grad_k_norm_weight | [head_dim] | bfloat16 |

| # | hidden_size | num_heads | num_kv_heads | head_dim | num_kv_groups | qkv_size | rstd_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 32 | 256 | 3.9017 | 0.4557 |
| 2 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 16 | 128 | 1.3648 | 0.1142 |
| 3 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 1 | 131 | 0.7860 | 0.0144 |
| 4 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 1 | 1571 | 1.1298 | 0.0877 |
| 5 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 8 | 256 | 1.3633 | 0.1142 |
| 6 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 16 | 256 | 2.2666 | 0.2280 |
| 7 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 32 | 128 | 2.2748 | 0.2280 |
| 8 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 64 | 128 | 4.9282 | 0.4557 |
| 9 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 1 | 256 | 0.6118 | 0.0152 |
| 10 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 4 | 128 | 0.7330 | 0.0289 |
| 11 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 2 | 1024 | 1.3612 | 0.1142 |
| 12 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 1 | 128 | 0.8252 | 0.0144 |
| 13 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 2 | 2048 | 2.2738 | 0.2280 |
| 14 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 2 | 293 | 0.7525 | 0.0330 |
| 15 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 4 | 541 | 1.4081 | 0.1207 |
| 16 | 4096 | 32 | 8 | 128 | 4 | 6144 | 1 | 1 | 512 | 0.7155 | 0.0289 |

```python
import torch
import torch.nn.functional as F
from typing import Tuple

def _rms_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for RMS normalization."""
    input_dtype = grad_output.dtype
    grad_output_float = grad_output.to(torch.float32)
    x_float = x.to(torch.float32)
    weight_float = weight.to(torch.float32)
    rstd_float = rstd.to(torch.float32)
    
    # Normalized input
    x_normed = x_float * rstd_float
    
    # Gradient w.r.t. weight: sum over batch, heads, and sequence dimensions
    grad_weight = (grad_output_float * x_normed).sum(dim=(0, 1, 2))
    
    # Gradient w.r.t. input
    # Step 1: grad from direct path (weight * grad_output * rstd)
    grad_x_direct = grad_output_float * weight_float * rstd_float
    
    # Step 2: grad from rstd path
    grad_rstd = (grad_output_float * weight_float * x_float).sum(
        dim=-1, keepdim=True
    )
    
    # d(rstd)/d(x) = -rstd^3 * x / D
    D = x.shape[-1]
    grad_x_from_rstd = grad_rstd * (-rstd_float.pow(3) * x_float / D)
    
    grad_x = grad_x_direct + grad_x_from_rstd
    
    return grad_x.to(input_dtype), grad_weight.to(input_dtype)

def _rope_backward(
    grad_output: torch.Tensor,
    x_rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for RoPE."""
    # Split grad_output into two halves
    half_dim = grad_output.shape[-1] // 2
    grad_1 = grad_output[..., :half_dim]
    grad_2 = grad_output[..., half_dim:]
    
    # Gradient w.r.t. input x (inverse rotation)
    grad_rotated_inv = torch.cat((grad_2, -grad_1), dim=-1)
    grad_x = grad_output * cos + grad_rotated_inv * sin
    
    # Recover original x from rotated x
    x_rotated_1 = x_rotated[..., :half_dim]
    x_rotated_2 = x_rotated[..., half_dim:]
    x_rotated_inv = torch.cat((-x_rotated_2, x_rotated_1), dim=-1)

    x_original = x_rotated * cos + x_rotated_inv * sin
    
    # Gradient w.r.t. cos - sum over heads dimension (dim=1)
    grad_cos = (grad_output * x_original).sum(dim=1)
    
    # Gradient w.r.t. sin
    x_original_1 = x_original[..., :half_dim]
    x_original_2 = x_original[..., half_dim:]
    x_original_rotated = torch.cat((-x_original_2, x_original_1), dim=-1)
    
    # Sum over heads dimension (dim=1)
    grad_sin = (grad_output * x_original_rotated).sum(dim=1)
    
    return grad_x, grad_cos, grad_sin

@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_pre_norm: torch.Tensor,
    key_pre_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    eps: float
):
    """Backward pass for fused GQA QKV projection with RoPE and QK normalization."""
    # Constants
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_kv_groups = 4
    
    bsz, seq_len, hidden_size = hidden_states.shape
    qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim
    
    # Step 6 backward: RMS normalization gradients
    grad_query_pre_norm, grad_q_norm_weight = _rms_norm_backward(
        grad_query, query_pre_norm, q_norm_weight, q_rstd, num_heads
    )
    
    grad_key_pre_norm, grad_k_norm_weight = _rms_norm_backward(
        grad_key, key_pre_norm, k_norm_weight, k_rstd, num_kv_heads
    )
    
    # Step 5 backward: RoPE gradients
    # Expand cos/sin to match query/key shapes
    # cos, sin: [bsz, seq_len, head_dim] -> [bsz, 1, seq_len, head_dim]
    cos_expanded_q = cos.unsqueeze(1)  # [bsz, 1, seq_len, head_dim]
    sin_expanded_q = sin.unsqueeze(1)
    cos_expanded_k = cos.unsqueeze(1)
    sin_expanded_k = sin.unsqueeze(1)
    
    grad_query_pre_rope, grad_cos_q, grad_sin_q = _rope_backward(
        grad_query_pre_norm, query_pre_norm, cos_expanded_q, sin_expanded_q, num_heads
    )
    
    grad_key_pre_rope, grad_cos_k, grad_sin_k = _rope_backward(
        grad_key_pre_norm, key_pre_norm, cos_expanded_k, sin_expanded_k, num_kv_heads
    )
    
    # Combine cos/sin gradients from query and key paths
    # grad_cos_q: [bsz, seq_len, head_dim], grad_cos_k: [bsz, seq_len, head_dim]
    grad_cos = grad_cos_q + grad_cos_k
    grad_sin = grad_sin_q + grad_sin_k
    
    # Step 4 backward: Transpose and reshape gradients
    # grad_query_pre_rope: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, num_heads, head_dim]
    grad_query_reshaped = grad_query_pre_rope.transpose(1, 2)
    # grad_key_pre_rope: [bsz, num_kv_heads, seq_len, head_dim] -> [bsz, seq_len, num_kv_heads, head_dim]
    grad_key_reshaped = grad_key_pre_rope.transpose(1, 2)
    # grad_value: [bsz, num_kv_heads, seq_len, head_dim] -> [bsz, seq_len, num_kv_heads, head_dim]
    grad_value_reshaped = grad_value.transpose(1, 2)
    
    # Step 3+2 backward: Flatten Q, K, V and concatenate in standard [Q_all | K_all | V_all] layout
    grad_query_flat = grad_query_reshaped.reshape(bsz, seq_len, num_heads * head_dim)
    grad_key_flat = grad_key_reshaped.reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_value_flat = grad_value_reshaped.reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_qkv_states = torch.cat([grad_query_flat, grad_key_flat, grad_value_flat], dim=-1)
    
    # Step 1 backward: Linear projection gradients
    grad_qkv_flat = grad_qkv_states.reshape(-1, qkv_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    
    # Gradient w.r.t. hidden_states: [bsz*seq_len, qkv_size] @ [qkv_size, hidden_size] -> [bsz*seq_len, hidden_size]
    grad_hidden_states = F.linear(grad_qkv_flat, qkv_weight.t())
    grad_hidden_states = grad_hidden_states.reshape(bsz, seq_len, hidden_size)
    
    # Gradient w.r.t. qkv_weight: [qkv_size, bsz*seq_len] @ [bsz*seq_len, hidden_size] -> [qkv_size, hidden_size]
    grad_qkv_weight = torch.matmul(grad_qkv_flat.t(), hidden_flat)
    
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_qkv_weight.to(torch.bfloat16),
        grad_q_norm_weight.to(torch.bfloat16),
        grad_k_norm_weight.to(torch.bfloat16)
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.080907 ms |
| - | Scoring Baseline | 0.500000 | 1.363213 ms |
| - | Reference Implementation | 0.486687 | 1.429684 ms |
