## Description

Backward pass for vision rotary position embedding generation. Computes gradients through cos/sin operations, concatenation, indexing, and outer product to produce gradient for inverse frequencies.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_cos | [total_patches, head_dim] | float32 |
| grad_sin | [total_patches, head_dim] | float32 |
| pos_ids | [total_patches, 2] | int64 |
| inv_freq | [head_dim_quarter] | float32 |
| emb | [total_patches, head_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_inv_freq | [head_dim_quarter] | float32 |

| # | head_dim | head_dim_half | head_dim_quarter | num_images | total_patches | max_grid_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 72 | 36 | 18 | 32 | 12288 | 112 | 0.0932 | 0.0011 |
| 2 | 72 | 36 | 18 | 1 | 256 | 16 | 0.0862 | 0.0004 |
| 3 | 72 | 36 | 18 | 1 | 196 | 14 | 0.0848 | 0.0004 |
| 4 | 72 | 36 | 18 | 8 | 2048 | 48 | 0.0868 | 0.0005 |
| 5 | 72 | 36 | 18 | 2 | 541 | 24 | 0.0889 | 0.0004 |
| 6 | 72 | 36 | 18 | 1 | 131 | 12 | 0.0846 | 0.0004 |
| 7 | 72 | 36 | 18 | 16 | 6144 | 80 | 0.0873 | 0.0008 |
| 8 | 72 | 36 | 18 | 4 | 1024 | 32 | 0.0881 | 0.0005 |
| 9 | 72 | 36 | 18 | 32 | 8192 | 96 | 0.0870 | 0.0009 |
| 10 | 72 | 36 | 18 | 2 | 768 | 28 | 0.0880 | 0.0004 |
| 11 | 72 | 36 | 18 | 4 | 773 | 28 | 0.0879 | 0.0004 |
| 12 | 72 | 36 | 18 | 2 | 1163 | 36 | 0.0876 | 0.0005 |
| 13 | 72 | 36 | 18 | 4 | 613 | 26 | 0.0882 | 0.0004 |
| 14 | 72 | 36 | 18 | 16 | 1087 | 34 | 0.0876 | 0.0005 |
| 15 | 72 | 36 | 18 | 1 | 324 | 18 | 0.0884 | 0.0004 |
| 16 | 72 | 36 | 18 | 8 | 691 | 28 | 0.0881 | 0.0004 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    total_patches = axes_and_scalars['total_patches']
    head_dim = axes_and_scalars['head_dim']
    head_dim_quarter = axes_and_scalars['head_dim_quarter']
    max_grid_size = axes_and_scalars['max_grid_size']
    
    # Gradients from upstream
    grad_cos = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    grad_sin = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    
    # Position IDs - random valid indices into frequency table
    pos_ids = torch.randint(0, max_grid_size, (total_patches, 2), device=device, dtype=torch.int64)
    
    # Inverse frequencies (learnable parameter)
    inv_freq = torch.randn(head_dim_quarter, device=device, dtype=torch.float32)
    
    # Saved embedding from forward pass (random values representing actual embeddings)
    emb = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    
    return {
        'grad_cos': grad_cos,
        'grad_sin': grad_sin,
        'pos_ids': pos_ids,
        'inv_freq': inv_freq,
        'emb': emb
    }

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor
) -> torch.Tensor:
    """Backward pass for vision rotary position embedding generation.
    
    Computes gradient w.r.t. inverse frequencies through:
    1. Gradient through cos/sin: grad_emb = -grad_cos * sin(emb) + grad_sin * cos(emb)
    2. Gradient through concatenation: split and sum
    3. Gradient through indexing: scatter via index_add_
    4. Gradient through outer product: matrix multiply with seq
    
    Args:
        grad_cos: [total_patches, head_dim] gradient w.r.t. cosine output
        grad_sin: [total_patches, head_dim] gradient w.r.t. sine output
        pos_ids: [total_patches, 2] position indices for h and w
        inv_freq: [head_dim//4] inverse frequency tensor
        emb: [total_patches, head_dim] saved embedding from forward
        
    Returns:
        grad_inv_freq: [head_dim//4] gradient w.r.t. inverse frequencies
    """
    total_patches = grad_cos.shape[0]
    head_dim = grad_cos.shape[1]
    head_dim_quarter = inv_freq.shape[0]
    max_grid_size = pos_ids.max().item() + 1
    
    # Step 1: Gradient through cos and sin operations
    # d(cos(x))/dx = -sin(x), d(sin(x))/dx = cos(x)
    grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()
    
    # Step 2: Gradient through concatenation
    # Forward: emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    # Backward: split and sum
    grad_rotary_pos_emb = grad_emb[:, :head_dim//2] + grad_emb[:, head_dim//2:]
    
    # Step 3: Gradient through flatten operation
    # Forward: rotary_pos_emb = freqs[pos_ids].flatten(1)
    # Backward: unflatten
    grad_freqs_indexed = grad_rotary_pos_emb.reshape(total_patches, 2, head_dim_quarter)
    
    # Step 4: Gradient through indexing operation
    # Scatter gradients back to full freqs tensor
    grad_freqs = torch.zeros(
        max_grid_size, head_dim_quarter,
        device=inv_freq.device,
        dtype=inv_freq.dtype
    )
    
    # Scatter gradients for height positions
    grad_freqs.index_add_(0, pos_ids[:, 0], grad_freqs_indexed[:, 0, :])
    
    # Scatter gradients for width positions
    grad_freqs.index_add_(0, pos_ids[:, 1], grad_freqs_indexed[:, 1, :])
    
    # Step 5: Gradient through outer product
    # Forward: freqs = torch.outer(seq, inv_freq)
    # Backward: grad_inv_freq = seq^T @ grad_freqs
    seq = torch.arange(
        max_grid_size,
        device=inv_freq.device,
        dtype=inv_freq.dtype
    )
    grad_inv_freq = torch.matmul(seq, grad_freqs)
    
    return grad_inv_freq
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000495 ms |
| - | Scoring Baseline | 0.500000 | 0.087653 ms |
| - | Reference Implementation | 0.400772 | 0.131970 ms |
