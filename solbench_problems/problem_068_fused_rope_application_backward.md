## Description

Backward pass for fused rotary position embedding (RoPE) application. Computes gradients w.r.t. query, key, cosine, and sine tensors. The gradient computation mirrors the forward pass structure: grad_x = (grad_output * cos) + (rotate_half(grad_output) * sin).
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_q_embed | [batch_size, num_heads, seq_len, head_dim] | float32 |
| grad_k_embed | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |
| q | [batch_size, num_heads, seq_len, head_dim] | float32 |
| k | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |
| cos | [batch_size, seq_len, head_dim] | float32 |
| sin | [batch_size, seq_len, head_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_q | [batch_size, num_heads, seq_len, head_dim] | float32 |
| grad_k | [batch_size, num_kv_heads, seq_len, head_dim] | float32 |
| grad_cos | [batch_size, seq_len, head_dim] | float32 |
| grad_sin | [batch_size, seq_len, head_dim] | float32 |

| # | head_dim | half_head_dim | batch_size | num_heads | num_kv_heads | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 64 | 1 | 32 | 8 | 293 | 0.0913 | 0.0024 |
| 2 | 128 | 64 | 2 | 32 | 8 | 1024 | 0.1017 | 0.0143 |
| 3 | 128 | 64 | 1 | 64 | 8 | 2048 | 0.1996 | 0.0253 |
| 4 | 128 | 64 | 4 | 32 | 8 | 1571 | 0.1618 | 0.0432 |
| 5 | 128 | 64 | 1 | 64 | 8 | 1024 | 0.1132 | 0.0128 |
| 6 | 128 | 64 | 2 | 32 | 8 | 128 | 0.0913 | 0.0021 |
| 7 | 128 | 64 | 1 | 64 | 8 | 512 | 0.0927 | 0.0066 |
| 8 | 128 | 64 | 32 | 32 | 8 | 128 | 0.1535 | 0.0283 |
| 9 | 128 | 64 | 1 | 32 | 8 | 2053 | 0.1052 | 0.0144 |
| 10 | 128 | 64 | 1 | 32 | 32 | 1024 | 0.0862 | 0.0115 |
| 11 | 128 | 64 | 16 | 32 | 8 | 128 | 0.1036 | 0.0143 |
| 12 | 128 | 64 | 1 | 32 | 8 | 256 | 0.0919 | 0.0021 |
| 13 | 128 | 64 | 1 | 32 | 8 | 997 | 0.0943 | 0.0072 |
| 14 | 128 | 64 | 2 | 64 | 8 | 512 | 0.1035 | 0.0128 |
| 15 | 128 | 64 | 2 | 32 | 8 | 541 | 0.0913 | 0.0078 |
| 16 | 128 | 64 | 2 | 32 | 8 | 2048 | 0.1354 | 0.0283 |

```python
import torch

@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Backward pass for fused RoPE application.
    
    Mathematical derivation:
    Given forward: output = (x * cos) + (rotate_half(x) * sin)
    Where: rotate_half(x) = [-x2, x1] for x = [x1, x2]
    
    Gradient w.r.t. x:
        grad_x = grad_output * cos - rotate_half(grad_output) * sin
    
    Gradient w.r.t. cos:
        grad_cos = sum over heads of (grad_output * x)
    
    Gradient w.r.t. sin:
        grad_sin = sum over heads of (grad_output * rotate_half(x))
    """
    half_head_dim = 64
    unsqueeze_dim = 1
    
    # Unsqueeze cos and sin for broadcasting with q and k
    # cos, sin: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
    sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)
    
    # Helper function to rotate half
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half_head_dim]
        x2 = x[..., half_head_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
    # Compute gradient w.r.t. q
    # grad_q = grad_q_embed * cos - rotate_half(grad_q_embed) * sin
    grad_q = (grad_q_embed * cos_unsqueezed) - (rotate_half(grad_q_embed) * sin_unsqueezed)

    # Compute gradient w.r.t. k
    # grad_k = grad_k_embed * cos - rotate_half(grad_k_embed) * sin
    grad_k = (grad_k_embed * cos_unsqueezed) - (rotate_half(grad_k_embed) * sin_unsqueezed)
    
    # Compute gradient w.r.t. cos
    # grad_cos = (grad_q_embed * q) + (grad_k_embed * k), summed over heads
    grad_cos_from_q = grad_q_embed * q
    grad_cos_from_k = grad_k_embed * k
    # Sum over the head dimension (dim=1) to match original cos shape
    grad_cos = grad_cos_from_q.sum(dim=unsqueeze_dim) + grad_cos_from_k.sum(dim=unsqueeze_dim)
    
    # Compute gradient w.r.t. sin
    # grad_sin = (grad_q_embed * rotate_half(q)) + (grad_k_embed * rotate_half(k)), summed over heads
    q_rotated = rotate_half(q)
    k_rotated = rotate_half(k)
    grad_sin_from_q = grad_q_embed * q_rotated
    grad_sin_from_k = grad_k_embed * k_rotated
    # Sum over the head dimension (dim=1) to match original sin shape
    grad_sin = grad_sin_from_q.sum(dim=unsqueeze_dim) + grad_sin_from_k.sum(dim=unsqueeze_dim)
    
    return grad_q, grad_k, grad_cos, grad_sin
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.010411 ms |
| - | Scoring Baseline | 0.500000 | 0.110011 ms |
| - | Reference Implementation | 0.197631 | 0.434222 ms |
