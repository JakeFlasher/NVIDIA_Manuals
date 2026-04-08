## Description

Backward pass for masked softmax with attention dropout. Computes gradients through dropout -> softmax -> masked_fill in reverse order.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, n_heads, seq_len, seq_len] | float32 |
| p_attn | [batch_size, n_heads, seq_len, seq_len] | float32 |
| mask | [batch_size, 1, seq_len, seq_len] | bool |
| dropout_mask | [batch_size, n_heads, seq_len, seq_len] | bool |
| p_dropout | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_scores | [batch_size, n_heads, seq_len, seq_len] | float32 |

| # | batch_size | n_heads | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 32 | 512 | 0.0840 | 0.0070 |
| 2 | 1 | 16 | 919 | 0.1157 | 0.0110 |
| 3 | 2 | 16 | 541 | 0.1201 | 0.0028 |
| 4 | 2 | 8 | 293 | 0.0690 | 0.0008 |
| 5 | 2 | 4 | 256 | 0.0560 | 0.0005 |
| 6 | 1 | 2 | 2048 | 0.0634 | 0.0026 |
| 7 | 4 | 4 | 4096 | 0.6503 | 0.0704 |
| 8 | 4 | 16 | 256 | 0.0781 | 0.0015 |
| 9 | 8 | 8 | 131 | 0.0510 | 0.0013 |
| 10 | 64 | 2 | 128 | 0.0759 | 0.0009 |
| 11 | 64 | 8 | 128 | 0.1010 | 0.0026 |
| 12 | 16 | 8 | 2048 | 1.2753 | 0.1404 |
| 13 | 8 | 4 | 613 | 0.1529 | 0.0035 |
| 14 | 32 | 8 | 256 | 0.1458 | 0.0048 |
| 15 | 8 | 16 | 373 | 0.1614 | 0.0050 |
| 16 | 4 | 8 | 512 | 0.1078 | 0.0026 |

```python
import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    """
    Backward pass for masked softmax with dropout.
    
    Computes gradients through three operations in reverse order:
    1. Dropout backward: scale by dropout mask and inverse probability
    2. Softmax backward: y * (grad - sum(y * grad))
    3. Masked fill backward: zero out gradients at masked positions
    
    Args:
        grad_output: Gradient w.r.t. output [B, H, T, T]
        p_attn: Softmax output (before dropout) [B, H, T, T]
        mask: Attention mask [B, 1, T, T] (True=unmasked, False=masked)
        dropout_mask: Dropout mask [B, H, T, T] (True=kept, False=dropped)
        p_dropout: Dropout probability
    
    Returns:
        grad_scores: Gradient w.r.t. input scores [B, H, T, T]
    """
    # Step 1: Gradient through dropout
    # Dropout backward: grad = grad_output * dropout_mask / (1 - p_dropout)
    if p_dropout > 0.0:
        # Scale by dropout mask and inverse keep probability
        grad_softmax_output = grad_output * dropout_mask.float() / (1.0 - p_dropout)
    else:
        # No dropout, gradient passes through unchanged
        grad_softmax_output = grad_output
    
    # Step 2: Gradient through softmax
    # Softmax backward formula: grad_input = y * (grad_output - sum(y * grad_output))
    # This is derived from the Jacobian of softmax:
    # J_ij = y_i * (delta_ij - y_j)
    
    # Compute sum of (y * grad) along the softmax dimension (dim=-1)
    # Shape: [B, H, T, T] -> [B, H, T, 1]
    sum_term = (p_attn * grad_softmax_output).sum(dim=-1, keepdim=True)
    
    # Compute softmax gradient: y * (grad - sum_term)
    # Shape: [B, H, T, T]
    grad_softmax_input = p_attn * (grad_softmax_output - sum_term)
    
    # Step 3: Gradient through masked_fill
    # Masked positions should have zero gradient
    # Only unmasked positions (mask == True) receive gradient flow
    # Zero out gradients at masked positions (mask == False)
    grad_scores = grad_softmax_input.masked_fill(~mask, 0.0)
    
    return grad_scores
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003837 ms |
| - | Scoring Baseline | 0.500000 | 0.123189 ms |
| - | Reference Implementation | 0.328738 | 0.266812 ms |
