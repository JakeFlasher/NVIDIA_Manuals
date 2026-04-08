## Description

Fused intra-chunk diagonal computation for Mamba2 SSD that combines segment_sum for causal masking, B/C matrix contraction to form attention-like weights G, and application of exponential decay mask L to compute Y_diag. This is the most memory-intensive part of the chunked parallel scan.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, num_chunks, chunk_size, num_heads, head_dim] | bfloat16 |
| A_cumsum | [batch_size, num_heads, num_chunks, chunk_size] | bfloat16 |
| B | [batch_size, num_chunks, chunk_size, n_groups, state_size] | bfloat16 |
| C | [batch_size, num_chunks, chunk_size, n_groups, state_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| Y_diag | [batch_size, num_chunks, chunk_size, num_heads, head_dim] | bfloat16 |

| # | chunk_size | num_heads | head_dim | n_groups | state_size | batch_size | num_chunks | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 32 | 128 | 8 | 128 | 4 | 7 | 0.1477 | 0.0100 |
| 2 | 128 | 32 | 128 | 8 | 128 | 1 | 8 | 0.1458 | 0.0032 |
| 3 | 128 | 32 | 128 | 8 | 128 | 4 | 1 | 0.1392 | 0.0018 |
| 4 | 128 | 32 | 128 | 8 | 128 | 2 | 8 | 0.3678 | 0.0059 |
| 5 | 128 | 32 | 128 | 8 | 128 | 1 | 4 | 0.1519 | 0.0018 |
| 6 | 128 | 32 | 128 | 8 | 128 | 1 | 2 | 0.1474 | 0.0011 |
| 7 | 128 | 32 | 128 | 8 | 128 | 1 | 3 | 0.1471 | 0.0014 |
| 8 | 128 | 32 | 128 | 8 | 128 | 16 | 1 | 0.1428 | 0.0059 |
| 9 | 128 | 32 | 128 | 8 | 128 | 2 | 2 | 0.1696 | 0.0018 |
| 10 | 128 | 32 | 128 | 8 | 128 | 4 | 16 | 0.2389 | 0.0224 |
| 11 | 128 | 32 | 128 | 8 | 128 | 1 | 32 | 0.1576 | 0.0114 |
| 12 | 128 | 32 | 128 | 8 | 128 | 8 | 4 | 0.1793 | 0.0114 |
| 13 | 128 | 32 | 128 | 8 | 128 | 4 | 4 | 0.1684 | 0.0059 |
| 14 | 128 | 32 | 128 | 8 | 128 | 2 | 4 | 0.1671 | 0.0032 |

```python
import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    """
    Compute intra-chunk diagonal output Y_diag for Mamba2 SSD.
    
    Steps:
    1. Compute segment_sum(A) to get cumulative decay factors
    2. Apply exp() to get causal mask L with exponential decay
    3. Contract B and C matrices to form attention weights G
    4. Apply mask: M = G * L (element-wise)
    5. Compute output: Y_diag = sum(M * hidden_states) over sequence dimension
    """
    # Constants
    CHUNK_SIZE = 128
    NUM_HEADS = 32
    N_GROUPS = 8
    
    batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
    
    # Step 1: Compute segment_sum with lower triangular masking for causality
    # A_cumsum: [batch, num_heads, num_chunks, chunk_size]
    
    # Expand to add target dimension
    # [batch, num_heads, num_chunks, chunk_size] -> [batch, num_heads, num_chunks, chunk_size, chunk_size]
    A_expanded = A_cumsum[..., None].expand(*A_cumsum.size(), CHUNK_SIZE).to(torch.float32)
    
    # Create lower triangular mask (diagonal = -1 means exclude diagonal)
    mask = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=A_cumsum.device, dtype=torch.bool),
        diagonal=-1
    )
    
    # Zero out upper triangular part
    A_masked = A_expanded.masked_fill(~mask, 0)
    
    # Compute cumulative sum along the source dimension
    A_cumsum_seg = torch.cumsum(A_masked, dim=-2)
    
    # Create mask including diagonal this time
    mask_with_diag = torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=A_cumsum.device, dtype=torch.bool),
        diagonal=0
    )
    
    # Mask out upper triangular part of cumsum result
    segment_sum_A = A_cumsum_seg.masked_fill(~mask_with_diag, -torch.inf)
    
    # Apply exponential to get causal decay mask L
    # L[i,j] represents decay from position j to position i (i >= j)
    L = torch.exp(segment_sum_A)
    # L: [batch, num_heads, num_chunks, chunk_size, chunk_size]
    
    # Step 2: Contract B and C to form attention-like weights G
    # B: [batch, num_chunks, chunk_size, n_groups, state_size]
    # C: [batch, num_chunks, chunk_size, n_groups, state_size]
    
    # Convert to float32 for computation
    B_f32 = B.to(torch.float32)
    C_f32 = C.to(torch.float32)
    
    # Expand B and C from n_groups to num_heads
    # [batch, num_chunks, chunk_size, n_groups, state_size] -> [batch, num_chunks, chunk_size, num_heads, state_size]
    B_expanded = B_f32.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
    C_expanded = C_f32.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
    
    # Compute G via outer product and sum over state dimension
    # G[i,j] = sum_n(C[i,n] * B[j,n])
    # C: [batch, num_chunks, chunk_size_i, num_heads, state_size]
    # B: [batch, num_chunks, chunk_size_j, num_heads, state_size]
    # Result: [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads]
    
    # Add dimension for broadcasting
    # C: [batch, num_chunks, chunk_size, 1, num_heads, state_size]
    # B: [batch, num_chunks, 1, chunk_size, num_heads, state_size]
    C_for_G = C_expanded[:, :, :, None, :, :]
    B_for_G = B_expanded[:, :, None, :, :, :]
    
    # Element-wise multiply and sum over state dimension
    # G: [batch, num_chunks, chunk_size, chunk_size, num_heads]
    G = (C_for_G * B_for_G).sum(dim=-1)
    
    # Step 3: Apply causal mask L to G to get M
    # L: [batch, num_heads, num_chunks, chunk_size, chunk_size]
    # G: [batch, num_chunks, chunk_size, chunk_size, num_heads]
    
    # Permute L to match G dimensions
    # L: [batch, num_heads, num_chunks, chunk_size, chunk_size] -> [batch, num_chunks, chunk_size, chunk_size, num_heads]
    L_permuted = L.permute(0, 2, 3, 4, 1)
    
    # Element-wise multiply
    M = G * L_permuted
    # M: [batch, num_chunks, chunk_size, chunk_size, num_heads]
    
    # Step 4: Compute Y_diag by applying M to hidden_states
    # M: [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads]
    # hidden_states: [batch, num_chunks, chunk_size_j, num_heads, head_dim]
    # Result: [batch, num_chunks, chunk_size_i, num_heads, head_dim]
    
    hidden_states_f32 = hidden_states.to(torch.float32)
    
    # Add dimension to M for head_dim
    # M: [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads, 1]
    M_expanded = M[..., None]
    
    # Add dimension to hidden_states for chunk_size_i
    # hidden_states: [batch, num_chunks, 1, chunk_size_j, num_heads, head_dim]
    hidden_states_expanded = hidden_states_f32[:, :, None, :, :, :]
    
    # Element-wise multiply and sum over chunk_size_j dimension
    Y_diag = (M_expanded * hidden_states_expanded).sum(dim=3)
    # Y_diag: [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    return Y_diag.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.004210 ms |
| - | Scoring Baseline | 0.500000 | 0.169893 ms |
| - | Reference Implementation | 0.033720 | 6.766481 ms |
