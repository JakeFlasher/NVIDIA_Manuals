## Description

Backward pass for fused attention QK matmul with scaling, causal mask, and softmax. Computes gradients for query and key tensors through the chain: softmax_backward -> scale_backward -> matmul_backward.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, num_heads, seq_len_q, seq_len_k] | bfloat16 |
| query | [batch_size, num_heads, seq_len_q, head_dim] | bfloat16 |
| key | [batch_size, num_heads, seq_len_k, head_dim] | bfloat16 |
| attn_weights | [batch_size, num_heads, seq_len_q, seq_len_k] | bfloat16 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_query | [batch_size, num_heads, seq_len_q, head_dim] | bfloat16 |
| grad_key | [batch_size, num_heads, seq_len_k, head_dim] | bfloat16 |

| # | num_heads | head_dim | batch_size | seq_len_q | seq_len_k | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 128 | 1 | 211 | 211 | 0.1401 | 0.0052 |
| 2 | 128 | 128 | 1 | 2048 | 2048 | 1.4285 | 0.2980 |
| 3 | 128 | 128 | 1 | 512 | 512 | 0.2199 | 0.0223 |
| 4 | 128 | 128 | 1 | 1024 | 1024 | 0.4476 | 0.0792 |
| 5 | 128 | 128 | 16 | 256 | 256 | 0.4888 | 0.1054 |
| 6 | 128 | 128 | 4 | 256 | 256 | 0.2000 | 0.0267 |
| 7 | 128 | 128 | 1 | 256 | 512 | 0.1435 | 0.0124 |
| 8 | 128 | 128 | 8 | 512 | 512 | 0.7858 | 0.1754 |
| 9 | 128 | 128 | 8 | 128 | 128 | 0.1482 | 0.0179 |
| 10 | 128 | 128 | 2 | 449 | 449 | 0.5012 | 0.0350 |
| 11 | 128 | 128 | 2 | 128 | 256 | 0.1267 | 0.0081 |
| 12 | 128 | 128 | 32 | 128 | 128 | 0.3260 | 0.0704 |
| 13 | 128 | 128 | 2 | 512 | 512 | 0.3073 | 0.0442 |
| 14 | 128 | 128 | 2 | 256 | 512 | 0.1983 | 0.0245 |
| 15 | 128 | 128 | 1 | 4096 | 4096 | 5.5574 | 1.1557 |
| 16 | 128 | 128 | 8 | 256 | 256 | 0.3236 | 0.0529 |

```python
import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_weights: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for fused attention QK matmul + scale + mask + softmax.
    
    Gradient flow:
    grad_output -> softmax_backward -> scale_backward -> matmul_backward -> grad_Q, grad_K
    
    Args:
        grad_output: (batch, num_heads, seq_len_q, seq_len_k) gradient from downstream
        query: (batch, num_heads, seq_len_q, head_dim) saved from forward
        key: (batch, num_heads, seq_len_k, head_dim) saved from forward
        attn_weights: (batch, num_heads, seq_len_q, seq_len_k) softmax output saved from forward
        scaling: float, scaling factor (1/sqrt(head_dim))
    
    Returns:
        grad_query: (batch, num_heads, seq_len_q, head_dim)
        grad_key: (batch, num_heads, seq_len_k, head_dim)
    """
    # Convert to float32 for numerical stability in softmax backward
    grad_output_f32 = grad_output.to(torch.float32)
    attn_weights_f32 = attn_weights.to(torch.float32)
    
    # Step 1: Gradient through softmax
    # For softmax: d_softmax/d_logits = softmax * (grad_out - sum(grad_out * softmax))
    # This is the efficient formulation that avoids computing the full Jacobian
    sum_grad = torch.sum(grad_output_f32 * attn_weights_f32, dim=-1, keepdim=True)
    grad_attn_logits = attn_weights_f32 * (grad_output_f32 - sum_grad)
    
    # Convert back to original dtype
    grad_attn_logits = grad_attn_logits.to(query.dtype)
    
    # Step 2: Gradient through scaling
    # d(x * scaling)/dx = scaling
    grad_scaled_logits = grad_attn_logits * scaling
    
    # Step 3: Gradient through Q @ K^T matrix multiplication
    # Forward: attn_logits = Q @ K^T
    # Backward: grad_Q = grad_attn_logits @ K
    #           grad_K = grad_attn_logits^T @ Q
    
    # grad_Q = grad_scaled_logits @ K
    # Shape: (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, head_dim)
    #     -> (batch, num_heads, seq_len_q, head_dim)
    grad_query = torch.matmul(grad_scaled_logits, key)
    
    # grad_K = grad_scaled_logits^T @ Q
    # Shape: (batch, num_heads, seq_len_k, seq_len_q) @ (batch, num_heads, seq_len_q, head_dim)
    #     -> (batch, num_heads, seq_len_k, head_dim)
    grad_key = torch.matmul(grad_scaled_logits.transpose(-2, -1), query)
    
    return grad_query, grad_key
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.045959 ms |
| - | Scoring Baseline | 0.500000 | 0.358060 ms |
| - | Reference Implementation | 0.218565 | 1.204198 ms |
