## Description

Mamba SSM parameter discretization and segment sum computation for chunk-based parallel processing. Discretizes continuous-time SSM parameters using timestep dt, computes cumulative sums for decay factors, and performs segment sum operations for intra-chunk attention-like computations.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, num_chunks, chunk_size, num_heads, head_dim] | bfloat16 |
| dt | [batch_size, num_chunks, chunk_size, num_heads] | bfloat16 |
| A_log | [num_heads] | bfloat16 |
| B | [batch_size, num_chunks, chunk_size, n_groups, d_state] | bfloat16 |
| C | [batch_size, num_chunks, chunk_size, n_groups, d_state] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| L | [batch_size, num_heads, num_chunks, chunk_size, chunk_size] | bfloat16 |
| G | [batch_size, num_chunks, chunk_size, chunk_size, num_heads] | bfloat16 |
| M | [batch_size, num_chunks, chunk_size, chunk_size, num_heads] | bfloat16 |
| A_cumsum | [batch_size, num_heads, num_chunks, chunk_size] | float32 |
| decay_states | [batch_size, num_heads, num_chunks, chunk_size] | float32 |

| # | chunk_size | num_heads | head_dim | n_groups | d_state | batch_size | num_chunks | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 16 | 160 | 1 | 256 | 4 | 3 | 0.1751 | 0.0107 |
| 2 | 256 | 16 | 160 | 1 | 256 | 2 | 16 | 0.2437 | 0.0279 |
| 3 | 256 | 16 | 160 | 1 | 256 | 16 | 2 | 0.2404 | 0.0279 |
| 4 | 256 | 16 | 160 | 1 | 256 | 2 | 2 | 0.1950 | 0.0039 |
| 5 | 256 | 16 | 160 | 1 | 256 | 64 | 1 | 0.3433 | 0.0553 |
| 6 | 256 | 16 | 160 | 1 | 256 | 32 | 1 | 0.2379 | 0.0279 |
| 7 | 256 | 16 | 160 | 1 | 256 | 4 | 5 | 0.2093 | 0.0176 |
| 8 | 256 | 16 | 160 | 1 | 256 | 1 | 8 | 0.1752 | 0.0073 |
| 9 | 256 | 16 | 160 | 1 | 256 | 1 | 3 | 0.1740 | 0.0030 |
| 10 | 256 | 16 | 160 | 1 | 256 | 8 | 8 | 0.3384 | 0.0553 |
| 11 | 256 | 16 | 160 | 1 | 256 | 4 | 1 | 0.1666 | 0.0039 |
| 12 | 256 | 16 | 160 | 1 | 256 | 32 | 2 | 0.3391 | 0.0553 |
| 13 | 256 | 16 | 160 | 1 | 256 | 4 | 2 | 0.1762 | 0.0073 |
| 14 | 256 | 16 | 160 | 1 | 256 | 1 | 1 | 0.1575 | 0.0013 |
| 15 | 256 | 16 | 160 | 1 | 256 | 1 | 2 | 0.1717 | 0.0021 |
| 16 | 256 | 16 | 160 | 1 | 256 | 16 | 1 | 0.1807 | 0.0141 |

```python
import torch
import torch.nn.functional as F

def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Stable segment sum using cumulative sums and masking.
    
    Args:
        input_tensor: [..., chunk_size] tensor
    
    Returns:
        tensor_segsum: [..., chunk_size, chunk_size] cumulative sum matrix
    """
    chunk_size = input_tensor.size(-1)
    device = input_tensor.device
    
    # Expand to [..., chunk_size, chunk_size] and create lower triangular structure
    input_expanded = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    
    # Create lower triangular mask (diagonal=-1 excludes diagonal)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
        diagonal=-1
    )
    input_masked = input_expanded.masked_fill(~mask, 0)
    
    # Compute cumulative sum along second-to-last dimension
    tensor_segsum = torch.cumsum(input_masked, dim=-2)
    
    # Apply mask to keep only lower triangular part (including diagonal this time)
    mask_final = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
        diagonal=0
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask_final, float('-inf'))
    
    return tensor_segsum

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
):
    """
    Mamba SSM parameter discretization and segment sum computation.
    
    Args:
        hidden_states: [batch, chunks, chunk_size, num_heads, head_dim]
        dt: [batch, chunks, chunk_size, num_heads]
        A_log: [num_heads] - log-scale state transition parameter
        B: [batch, chunks, chunk_size, n_groups, d_state]
        C: [batch, chunks, chunk_size, n_groups, d_state]
    
    Returns:
        L: Decay matrix for intra-chunk computation
        G: Attention-like weights from B and C contraction
        M: Masked attention weights
        A_cumsum: Cumulative decay factors
        decay_states: State decay factors for inter-chunk recurrence
    """
    batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
    n_groups = B.shape[3]
    
    # 1. Discretize A parameter
    # A is in log scale, so exp(-A) gives the continuous-time decay
    A = -torch.exp(A_log.float())  # [num_heads]
    
    # Apply softplus to dt for numerical stability
    dt_soft = F.softplus(dt.float())  # [batch, chunks, chunk_size, num_heads]
    
    # Discretize: A_discrete = A * dt
    # Rearrange dt to [batch, num_heads, chunks, chunk_size]
    dt_permuted = dt_soft.permute(0, 3, 1, 2)  # [batch, num_heads, chunks, chunk_size]
    A_discrete = A[None, :, None, None] * dt_permuted  # [batch, num_heads, chunks, chunk_size]
    
    # 2. Compute cumulative sum of A (for decay factors)
    A_cumsum = torch.cumsum(A_discrete, dim=-1)  # [batch, num_heads, chunks, chunk_size]
    
    # 3. Compute L matrix (decay matrix for intra-chunk)
    # L[i,j] = exp(sum(A[j:i])) for i > j (lower triangular)
    L = torch.exp(segment_sum(A_discrete))  # [batch, num_heads, chunks, chunk_size, chunk_size]
    
    # 4. Repeat B and C across head groups
    # B, C: [batch, chunks, chunk_size, n_groups, d_state]
    # Need to expand to [batch, chunks, chunk_size, num_heads, d_state]
    repeats = num_heads // n_groups
    B_expanded = B.repeat_interleave(repeats, dim=3)  # [batch, chunks, chunk_size, num_heads, d_state]
    C_expanded = C.repeat_interleave(repeats, dim=3)  # [batch, chunks, chunk_size, num_heads, d_state]
    
    # 5. Compute G matrix (contraction of C and B)
    # G[i,j] = sum_n(C[i,n] * B[j,n])
    # Shape: [batch, chunks, chunk_size, chunk_size, num_heads]
    # C_expanded: [batch, chunks, chunk_size, num_heads, d_state]
    # B_expanded: [batch, chunks, chunk_size, num_heads, d_state]
    G = torch.einsum('bcihd,bcjhd->bcijh', C_expanded.float(), B_expanded.float())  # [batch, chunks, chunk_size, chunk_size, num_heads]
    
    # 6. Compute M matrix (masked attention weights)
    # M = G * L (element-wise, with proper broadcasting)
    # L: [batch, num_heads, chunks, chunk_size, chunk_size]
    # G: [batch, chunks, chunk_size, chunk_size, num_heads]
    # Need to align dimensions
    L_permuted = L.permute(0, 2, 3, 4, 1)  # [batch, chunks, chunk_size, chunk_size, num_heads]
    M = G * L_permuted  # [batch, chunks, chunk_size, chunk_size, num_heads]
    
    # 7. Compute decay factors for state updates
    # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)
    decay_states = torch.exp(
        A_cumsum[:, :, :, -1:] - A_cumsum
    )  # [batch, num_heads, chunks, chunk_size]
    
    return (
        L.to(torch.bfloat16),
        G.to(torch.bfloat16),
        M.to(torch.bfloat16),
        A_cumsum,
        decay_states
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.011238 ms |
| - | Scoring Baseline | 0.500000 | 0.212409 ms |
| - | Reference Implementation | 0.180314 | 1.040492 ms |
