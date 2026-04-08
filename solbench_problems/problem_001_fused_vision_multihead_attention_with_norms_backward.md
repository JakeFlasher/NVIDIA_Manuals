## Description

Backward pass for fused vision multi-head attention with pre-normalization and residual connection. Computes gradients through: residual connection, dropout, output projection, attention mechanism (softmax, matmul), head split/merge, QKV projection, and LayerNorm.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, embed_dim] | float32 |
| x | [batch_size, seq_len, embed_dim] | float32 |
| x_mean | [batch_size, seq_len, 1] | float32 |
| x_var | [batch_size, seq_len, 1] | float32 |
| x_norm | [batch_size, seq_len, embed_dim] | float32 |
| ln_weight | [embed_dim] | float32 |
| qkv_weight | [qkv_dim, embed_dim] | float32 |
| q | [batch_size, num_heads, seq_len, head_dim] | float32 |
| k | [batch_size, num_heads, seq_len, head_dim] | float32 |
| v | [batch_size, num_heads, seq_len, head_dim] | float32 |
| attn_weights | [batch_size, num_heads, seq_len, seq_len] | float32 |
| attn_output | [batch_size, seq_len, embed_dim] | float32 |
| out_weight | [embed_dim, embed_dim] | float32 |
| scale | scalar | float32 |
| norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_x | [batch_size, seq_len, embed_dim] | float32 |
| grad_qkv_weight | [qkv_dim, embed_dim] | float32 |
| grad_qkv_bias | [qkv_dim] | float32 |
| grad_out_weight | [embed_dim, embed_dim] | float32 |
| grad_out_bias | [embed_dim] | float32 |
| grad_ln_weight | [embed_dim] | float32 |
| grad_ln_bias | [embed_dim] | float32 |

| # | seq_len | embed_dim | num_heads | head_dim | qkv_dim | batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 577 | 1024 | 16 | 64 | 3072 | 10 | 3.8747 | 0.0689 |
| 2 | 577 | 1024 | 16 | 64 | 3072 | 5 | 2.2590 | 0.0347 |
| 3 | 577 | 1024 | 16 | 64 | 3072 | 14 | 5.2201 | 0.0963 |
| 4 | 577 | 1024 | 16 | 64 | 3072 | 3 | 1.5218 | 0.0210 |
| 5 | 577 | 1024 | 16 | 64 | 3072 | 2 | 1.3007 | 0.0141 |
| 6 | 577 | 1024 | 16 | 64 | 3072 | 32 | 11.4170 | 0.2196 |
| 7 | 577 | 1024 | 16 | 64 | 3072 | 47 | 16.4507 | 0.3224 |
| 8 | 577 | 1024 | 16 | 64 | 3072 | 28 | 10.0707 | 0.1922 |
| 9 | 577 | 1024 | 16 | 64 | 3072 | 23 | 8.4932 | 0.1580 |
| 10 | 577 | 1024 | 16 | 64 | 3072 | 41 | 14.3960 | 0.2813 |
| 11 | 577 | 1024 | 16 | 64 | 3072 | 1 | 0.7493 | 0.0073 |
| 12 | 577 | 1024 | 16 | 64 | 3072 | 12 | 4.5592 | 0.0826 |
| 13 | 577 | 1024 | 16 | 64 | 3072 | 7 | 2.9184 | 0.0484 |
| 14 | 577 | 1024 | 16 | 64 | 3072 | 56 | 19.5926 | 0.3840 |
| 15 | 577 | 1024 | 16 | 64 | 3072 | 37 | 13.1706 | 0.2539 |
| 16 | 577 | 1024 | 16 | 64 | 3072 | 53 | 18.4393 | 0.3635 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    x_mean: torch.Tensor,
    x_var: torch.Tensor,
    x_norm: torch.Tensor,
    ln_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    out_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
):
    """
    Backward pass for fused vision multi-head attention with norms.
    
    Computes gradients through:
    1. Residual connection
    2. Output projection
    3. Attention mechanism (softmax, matmul)
    4. Head split/merge
    5. QKV projection
    6. LayerNorm
    """
    batch_size, seq_len, embed_dim = x.shape
    num_heads = 16
    head_dim = 64
    
    # Gradient through residual connection
    # output = output_before_residual + residual
    grad_output_before_residual = grad_output
    grad_residual = grad_output
    
    # Gradient through output projection
    # output = F.linear(attn_output, out_weight, out_bias)
    grad_attn_output = F.linear(grad_output_before_residual, out_weight.t())
    grad_out_weight = torch.matmul(
        grad_output_before_residual.reshape(-1, embed_dim).t(),
        attn_output.reshape(-1, embed_dim)
    )
    grad_out_bias = grad_output_before_residual.sum(dim=(0, 1))
    
    # Gradient through head merging
    grad_attn_output_heads = grad_attn_output.view(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)
    
    # Gradient through attention application: attn_output = attn_weights @ v
    grad_v = torch.matmul(
        attn_weights.transpose(-2, -1),
        grad_attn_output_heads
    )
    
    grad_attn_weights = torch.matmul(
        grad_attn_output_heads,
        v.transpose(-2, -1)
    )
    
    # Gradient through softmax
    sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)
    
    # Gradient through scaled dot-product: attn_scores = (q * scale) @ k^T
    grad_q = torch.matmul(grad_attn_scores, k) * scale
    grad_k = torch.matmul(
        grad_attn_scores.transpose(-2, -1),
        q * scale
    )
    
    # Gradient through head splitting
    grad_q = grad_q.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_k = grad_k.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_v = grad_v.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    
    # Gradient through QKV chunking
    grad_qkv = torch.cat([grad_q, grad_k, grad_v], dim=-1)
    
    # Gradient through QKV projection
    grad_x_norm = F.linear(grad_qkv, qkv_weight.t())
    grad_qkv_weight = torch.matmul(
        grad_qkv.reshape(-1, 3 * embed_dim).t(),
        x_norm.reshape(-1, embed_dim)
    )
    grad_qkv_bias = grad_qkv.sum(dim=(0, 1))
    
    # Gradient through LayerNorm
    x_normalized = (x - x_mean) / torch.sqrt(x_var + norm_eps)
    grad_ln_weight = (grad_x_norm * x_normalized).sum(dim=(0, 1))
    grad_ln_bias = grad_x_norm.sum(dim=(0, 1))
    
    # Gradient through normalization
    grad_x_normalized = grad_x_norm * ln_weight
    
    # Gradient through standardization
    std = torch.sqrt(x_var + norm_eps)
    x_centered = x - x_mean
    
    grad_x_from_norm = grad_x_normalized / std
    mean_grad = grad_x_from_norm.mean(dim=-1, keepdim=True)
    mean_grad_x_centered = (grad_x_normalized * x_centered).mean(dim=-1, keepdim=True) / (std ** 2)
    
    grad_x_from_norm = grad_x_from_norm - mean_grad - x_centered * mean_grad_x_centered
    
    # Combine gradients from residual and normalization paths
    grad_x = grad_residual + grad_x_from_norm
    
    return (
        grad_x,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.094813 ms |
| 1st place | AKO4ALL_L2 | 0.814770 | 1.335636 ms |
| - | Scoring Baseline | 0.500000 | 5.620653 ms |
| - | Reference Implementation | 0.466183 | 6.443375 ms |
