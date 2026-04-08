## Description

Backward pass for rotary position embedding with attention scaling. Computes gradients with respect to position_ids through the RoPE embedding generation pipeline including inverse frequency expansion, frequency computation, and cos/sin with attention scaling.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_cos | [batch_size, seq_len, head_dim] | bfloat16 |
| grad_sin | [batch_size, seq_len, head_dim] | bfloat16 |
| emb | [batch_size, seq_len, head_dim] | float32 |
| inv_freq_expanded | [batch_size, half_dim, 1] | float32 |
| attention_scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_position_ids | [batch_size, seq_len] | float32 |

| # | head_dim | half_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 64 | 32 | 256 | 0.0799 | 0.0011 |
| 2 | 128 | 64 | 2 | 3637 | 0.0767 | 0.0010 |
| 3 | 128 | 64 | 1 | 128 | 0.0776 | 0.0004 |
| 4 | 128 | 64 | 8 | 1024 | 0.0766 | 0.0011 |
| 5 | 128 | 64 | 1 | 131 | 0.0763 | 0.0004 |
| 6 | 128 | 64 | 16 | 613 | 0.0772 | 0.0012 |
| 7 | 128 | 64 | 8 | 449 | 0.0758 | 0.0007 |
| 8 | 128 | 64 | 16 | 512 | 0.0763 | 0.0011 |
| 9 | 128 | 64 | 64 | 3011 | 0.0868 | 0.0165 |
| 10 | 128 | 64 | 4 | 293 | 0.0769 | 0.0005 |
| 11 | 128 | 64 | 8 | 4093 | 0.1077 | 0.0031 |
| 12 | 128 | 64 | 64 | 128 | 0.0811 | 0.0011 |
| 13 | 128 | 64 | 4 | 3967 | 0.0781 | 0.0017 |
| 14 | 128 | 64 | 1 | 1321 | 0.0768 | 0.0005 |
| 15 | 128 | 64 | 64 | 8192 | 0.1719 | 0.0443 |
| 16 | 128 | 64 | 1 | 2048 | 0.0766 | 0.0006 |

```python
import torch

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    emb: torch.Tensor,
    inv_freq_expanded: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    """
    Backward pass for RoPE embedding with attention scaling.
    
    Computes gradient w.r.t. position_ids through the chain rule:
    - Forward: emb = inv_freq @ position_ids, cos_out = cos(emb) * scaling, sin_out = sin(emb) * scaling
    - Backward: grad_position_ids = inv_freq.T @ (grad_cos * (-sin(emb) * scaling) + grad_sin * (cos(emb) * scaling))
    
    Args:
        grad_cos: Gradient w.r.t. cos output, shape (batch, seq_len, head_dim)
        grad_sin: Gradient w.r.t. sin output, shape (batch, seq_len, head_dim)
        emb: Saved embedding tensor from forward, shape (batch, seq_len, head_dim)
        inv_freq_expanded: Expanded inverse frequencies, shape (batch, head_dim/2, 1)
        attention_scaling: Scaling factor applied in forward pass
    
    Returns:
        grad_position_ids: Gradient w.r.t. position_ids, shape (batch, seq_len)
    """
    # Convert gradients to float32 for numerical stability
    grad_cos = grad_cos.float()
    grad_sin = grad_sin.float()
    
    # Get dimensions
    head_dim = emb.shape[-1]
    half_dim = head_dim // 2
    
    # Split gradients into two halves (corresponding to the concatenation in forward)
    # emb was created as cat(freqs, freqs), so gradients from both halves contribute
    grad_cos_first_half = grad_cos[..., :half_dim]
    grad_cos_second_half = grad_cos[..., half_dim:]
    grad_sin_first_half = grad_sin[..., :half_dim]
    grad_sin_second_half = grad_sin[..., half_dim:]
    
    # Sum gradients from both halves (since freqs was duplicated in forward)
    grad_cos_freqs = grad_cos_first_half + grad_cos_second_half
    grad_sin_freqs = grad_sin_first_half + grad_sin_second_half
    
    # Compute gradient w.r.t. emb (freqs)
    # d(cos(emb))/d(emb) = -sin(emb)
    # d(sin(emb))/d(emb) = cos(emb)
    # Both are scaled by attention_scaling
    emb_half = emb[..., :half_dim]
    grad_emb = (
        grad_cos_freqs * (-emb_half.sin()) * attention_scaling +
        grad_sin_freqs * emb_half.cos() * attention_scaling
    )
    
    # Compute gradient w.r.t. position_ids
    # Forward: freqs = inv_freq_expanded @ position_ids_expanded, then transpose(1, 2)
    # So: emb (after transpose) has shape (batch, seq_len, half_dim)
    # We need to transpose grad_emb back: (batch, seq_len, half_dim) -> (batch, half_dim, seq_len)
    grad_freqs = grad_emb.transpose(1, 2)
    
    # Gradient through matmul: freqs = inv_freq_expanded @ position_ids_expanded
    # grad_position_ids_expanded = inv_freq_expanded.T @ grad_freqs
    # Shape: (batch, 1, half_dim) @ (batch, half_dim, seq_len) -> (batch, 1, seq_len)
    grad_position_ids_expanded = inv_freq_expanded.transpose(-2, -1) @ grad_freqs
    
    # Squeeze to get final shape: (batch, seq_len)
    grad_position_ids = grad_position_ids_expanded.squeeze(1)
    
    return grad_position_ids
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001344 ms |
| - | Scoring Baseline | 0.500000 | 0.083620 ms |
| - | Reference Implementation | 0.376074 | 0.144748 ms |
