## Description

Backward pass for attention output projection with reshape. Computes gradients w.r.t. attn_output and weight from grad_output. Forward: transpose (B,H,S,D)->(B,S,H,D), reshape (B,S,H,D)->(B,S,H*D), linear projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| reshaped | [batch_size, seq_len, hidden_size] | bfloat16 |
| weight | [hidden_size, hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_attn_output | [batch_size, num_heads, seq_len, head_dim] | bfloat16 |
| grad_weight | [hidden_size, hidden_size] | bfloat16 |

| # | num_heads | head_dim | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 64 | 2048 | 64 | 128 | 0.1894 | 0.0763 |
| 2 | 32 | 64 | 2048 | 16 | 128 | 0.0726 | 0.0194 |
| 3 | 32 | 64 | 2048 | 2 | 4096 | 0.1877 | 0.0763 |
| 4 | 32 | 64 | 2048 | 4 | 512 | 0.0723 | 0.0194 |
| 5 | 32 | 64 | 2048 | 16 | 256 | 0.1105 | 0.0383 |
| 6 | 32 | 64 | 2048 | 4 | 1024 | 0.1105 | 0.0383 |
| 7 | 32 | 64 | 2048 | 8 | 1024 | 0.1877 | 0.0763 |
| 8 | 32 | 64 | 2048 | 8 | 128 | 0.0524 | 0.0099 |
| 9 | 32 | 64 | 2048 | 1 | 256 | 0.0376 | 0.0030 |
| 10 | 32 | 64 | 2048 | 1 | 1024 | 0.0529 | 0.0099 |
| 11 | 32 | 64 | 2048 | 4 | 256 | 0.0521 | 0.0099 |
| 12 | 32 | 64 | 2048 | 32 | 256 | 0.1877 | 0.0763 |
| 13 | 32 | 64 | 2048 | 8 | 256 | 0.0838 | 0.0194 |
| 14 | 32 | 64 | 2048 | 4 | 128 | 0.0581 | 0.0051 |
| 15 | 32 | 64 | 2048 | 2 | 1024 | 0.0718 | 0.0194 |
| 16 | 32 | 64 | 2048 | 32 | 128 | 0.1106 | 0.0383 |

```python
import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for attention output projection with reshape.
    
    Forward was:
        1. Transpose: (B, H, S, D) -> (B, S, H, D)
        2. Reshape: (B, S, H, D) -> (B, S, H*D)
        3. Linear: y = x @ W^T
    
    Backward computes:
        - grad_weight = grad_output^T @ reshaped
        - grad_attn_output = (grad_output @ W^T) reshaped and transposed back
    """
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    head_dim = 64
    
    # Convert to float32 for numerical stability
    grad_output_f32 = grad_output.to(torch.float32)
    reshaped_f32 = reshaped.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    
    # Gradient w.r.t. weight
    # d_loss/d_W = grad_output^T @ reshaped
    # grad_output: (B, S, H*D), reshaped: (B, S, H*D)
    # Reshape to 2D: (B*S, H*D)
    grad_output_2d = grad_output_f32.reshape(-1, hidden_size)  # (B*S, H*D)
    reshaped_2d = reshaped_f32.reshape(-1, hidden_size)  # (B*S, H*D)
    
    # Matrix multiply: (H*D, B*S) @ (B*S, H*D) -> (H*D, H*D)
    grad_weight = grad_output_2d.t().mm(reshaped_2d)
    
    # Gradient w.r.t. input (attn_output)
    # d_loss/d_reshaped = grad_output @ W
    # grad_output shape: (B*S, H*D), weight shape: (H*D, H*D)
    grad_reshaped_2d = grad_output_2d.mm(weight_f32)  # (B*S, H*D)
    grad_reshaped = grad_reshaped_2d.reshape(batch_size, seq_len, hidden_size)
    
    # Backward through reshape: (B, S, H*D) -> (B, S, H, D)
    grad_transposed = grad_reshaped.reshape(batch_size, seq_len, num_heads, head_dim)
    
    # Backward through transpose: (B, S, H, D) -> (B, H, S, D)
    grad_attn_output = grad_transposed.transpose(1, 2).contiguous()
    
    return grad_attn_output.to(torch.bfloat16), grad_weight.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.022395 ms |
| 1st place | AKO4ALL_L1 | 0.532099 | 0.081428 ms |
| - | Scoring Baseline | 0.500000 | 0.089466 ms |
| 2nd place | Clever Scorpion | 0.419470 | 0.113584 ms |
| 3rd place | Quick Impala | 0.086774 | 0.768614 ms |
| - | Reference Implementation | 0.064894 | 1.065400 ms |
