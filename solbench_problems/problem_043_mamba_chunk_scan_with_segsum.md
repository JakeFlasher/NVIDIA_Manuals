## Description

Mamba-2 chunk-based parallel scan with segment sum computation for intra-chunk and inter-chunk state propagation. This is the core parallel scanning algorithm that enables efficient SSM computation by breaking sequences into chunks (chunk_size=256), computing intra-chunk attention-like patterns using segment sums, maintaining inter-chunk state recurrence with exponential decay, and combining diagonal (intra-chunk) and off-diagonal (inter-chunk) contributions.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, num_heads, head_dim] | bfloat16 |
| A | [batch_size, num_heads, seq_len] | bfloat16 |
| B | [batch_size, seq_len, n_groups, state_size] | bfloat16 |
| C | [batch_size, seq_len, n_groups, state_size] | bfloat16 |
| D | [num_heads] | bfloat16 |
| initial_states | [batch_size, num_heads, head_dim, state_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_out] | bfloat16 |
| final_state | [batch_size, num_heads, head_dim, state_size] | bfloat16 |

| # | num_heads | head_dim | state_size | n_groups | chunk_size | hidden_out | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 64 | 256 | 1 | 256 | 1024 | 1 | 1024 | 0.4037 | 0.0013 |
| 2 | 16 | 64 | 256 | 1 | 256 | 1024 | 1 | 997 | 0.3892 | 0.0012 |
| 3 | 16 | 64 | 256 | 1 | 256 | 1024 | 16 | 512 | 1.2780 | 0.0081 |
| 4 | 16 | 64 | 256 | 1 | 256 | 1024 | 1 | 256 | 0.2813 | 0.0007 |
| 5 | 16 | 64 | 256 | 1 | 256 | 1024 | 4 | 541 | 0.6533 | 0.0024 |
| 6 | 16 | 64 | 256 | 1 | 256 | 1024 | 4 | 1879 | 1.2364 | 0.0060 |
| 7 | 16 | 64 | 256 | 1 | 256 | 1024 | 4 | 256 | 0.3958 | 0.0017 |
| 8 | 16 | 64 | 256 | 1 | 256 | 1024 | 2 | 512 | 0.3852 | 0.0014 |
| 9 | 16 | 64 | 256 | 1 | 256 | 1024 | 8 | 1024 | 1.1981 | 0.0070 |
| 10 | 16 | 64 | 256 | 1 | 256 | 1024 | 1 | 4096 | 0.8277 | 0.0033 |
| 11 | 16 | 64 | 256 | 1 | 256 | 1024 | 16 | 256 | 0.8445 | 0.0054 |
| 12 | 16 | 64 | 256 | 1 | 256 | 1024 | 32 | 256 | 1.4748 | 0.0103 |
| 13 | 16 | 64 | 256 | 1 | 256 | 1024 | 2 | 293 | 0.4400 | 0.0011 |
| 14 | 16 | 64 | 256 | 1 | 256 | 1024 | 4 | 2048 | 1.1619 | 0.0065 |
| 15 | 16 | 64 | 256 | 1 | 256 | 1024 | 2 | 1321 | 0.7079 | 0.0025 |
| 16 | 16 | 64 | 256 | 1 | 256 | 1024 | 4 | 1024 | 0.7338 | 0.0037 |

```python
import torch
import torch.nn.functional as F

def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute segment sum (cumulative sum with lower triangular masking)."""
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=0
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask, float('-inf'))
    return tensor_segsum

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pad tensor on seq_len dimension."""
    if pad_size == 0:
        return input_tensor
    if len(input_tensor.shape) == 4:
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0)
    else:
        pad_shape = (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode='constant', value=0)

def reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """Reshape tensor into chunks after padding."""
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_states: torch.Tensor,
):
    """Mamba-2 chunk-based parallel scan with segment sum."""
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    state_size = 256
    n_groups = 1
    chunk_size = 256
    
    # Compute padding size to make seq_len multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    
    # Convert to float32 for numerical stability
    hidden_states_f = hidden_states.to(torch.float32)
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_f = C.to(torch.float32)
    D_f = D.to(torch.float32)
    initial_states_f = initial_states.to(torch.float32)
    
    # Expand B and C to match num_heads (from n_groups=1 to num_heads=16)
    # [batch, seq_len, 1, state_size] -> [batch, seq_len, 16, state_size]
    B_expanded = B_f.expand(batch_size, seq_len, num_heads, state_size)
    C_expanded = C_f.expand(batch_size, seq_len, num_heads, state_size)
    
    # Apply D residual (before chunking)
    hidden_states_padded = pad_tensor_by_size(hidden_states_f, pad_size)
    D_residual = D_f[None, None, :, None] * hidden_states_padded  # [batch, seq_len_padded, num_heads, head_dim]
    
    # Reshape into chunks
    hidden_states_chunked = reshape_into_chunks(hidden_states_f, pad_size, chunk_size)
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    A_transposed = A_f.transpose(1, 2)  # [batch, seq_len, num_heads]
    A_chunked = reshape_into_chunks(A_transposed, pad_size, chunk_size)
    # [batch, num_chunks, chunk_size, num_heads]
    
    B_chunked = reshape_into_chunks(B_expanded, pad_size, chunk_size)
    # [batch, num_chunks, chunk_size, num_heads, state_size]
    
    C_chunked = reshape_into_chunks(C_expanded, pad_size, chunk_size)
    # [batch, num_chunks, chunk_size, num_heads, state_size]
    
    num_chunks = A_chunked.shape[1]
    
    # Permute A for cumsum: [batch, num_chunks, chunk_size, num_heads] -> [batch, num_heads, num_chunks, chunk_size]
    A_chunked_perm = A_chunked.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_chunked_perm, dim=-1)
    # [batch, num_heads, num_chunks, chunk_size]
    
    # 1. Compute intra-chunk outputs (diagonal blocks)
    # L matrix: exponential of segment sum of A
    L = torch.exp(segment_sum(A_chunked_perm))  # [batch, num_heads, num_chunks, chunk_size, chunk_size]
    
    # Compute G: contraction of C and B over state_size
    # C_chunked: [batch, num_chunks, chunk_size, num_heads, state_size]
    # B_chunked: [batch, num_chunks, chunk_size, num_heads, state_size]
    # G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
    G = torch.einsum('bcihs,bcjhs->bcijh', C_chunked, B_chunked)
    # [batch, num_chunks, chunk_size, chunk_size, num_heads]
    
    # Compute M: apply L (attention-like pattern) to G
    # L: [batch, num_heads, num_chunks, chunk_size, chunk_size]
    # G: [batch, num_chunks, chunk_size, chunk_size, num_heads]
    # Permute L to [batch, num_chunks, chunk_size, chunk_size, num_heads]
    L_perm = L.permute(0, 2, 3, 4, 1)  # [batch, num_chunks, chunk_size, chunk_size, num_heads]
    M = G * L_perm  # element-wise, [batch, num_chunks, chunk_size, chunk_size, num_heads]
    
    # Apply M to hidden_states (like attention to values)
    # M: [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads]
    # hidden_states_chunked: [batch, num_chunks, chunk_size_j, num_heads, head_dim]
    # Y_diag[b, nc, i, h, d] = sum_j M[b, nc, i, j, h] * hidden_states[b, nc, j, h, d]
    Y_diag = torch.einsum('bcijh,bcjhd->bcihd', M, hidden_states_chunked)
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    # 2. Compute states for each chunk (right term of factorization)
    # decay_states: exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # A_cumsum: [batch, num_heads, num_chunks, chunk_size]
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [batch, num_heads, num_chunks, chunk_size]
    # Permute to [batch, num_chunks, chunk_size, num_heads]
    decay_states_perm = decay_states.permute(0, 2, 3, 1)  # [batch, num_chunks, chunk_size, num_heads]
    
    # B_decay: [batch, num_chunks, chunk_size, num_heads, state_size]
    B_decay = B_chunked * decay_states_perm[..., None]
    
    # Compute states: sum over chunk_size dimension
    # states[b, nc, h, d, s] = sum_t B_decay[b, nc, t, h, s] * hidden_states[b, nc, t, h, d]
    states = torch.einsum('bcths,bcthd->bchds', B_decay, hidden_states_chunked)
    # [batch, num_chunks, num_heads, head_dim, state_size]
    
    # 3. Compute inter-chunk recurrence (middle term)
    # Prepend initial state
    # initial_states_f: [batch, num_heads, head_dim, state_size]
    # states: [batch, num_chunks, num_heads, head_dim, state_size]
    # Need to transpose states to [batch, num_chunks, num_heads, head_dim, state_size] -> already in this form
    # Prepend: [batch, 1, num_heads, head_dim, state_size]
    initial_states_expanded = initial_states_f[:, None, :, :, :]  # [batch, 1, num_heads, head_dim, state_size]
    states_with_init = torch.cat([initial_states_expanded, states], dim=1)
    # [batch, num_chunks+1, num_heads, head_dim, state_size]
    
    # Compute decay between chunks
    # A_cumsum[:, :, :, -1]: [batch, num_heads, num_chunks]
    A_chunk_ends = A_cumsum[:, :, :, -1]  # [batch, num_heads, num_chunks]
    A_chunk_ends_padded = F.pad(A_chunk_ends, (1, 0))  # [batch, num_heads, num_chunks+1]
    decay_chunk = torch.exp(segment_sum(A_chunk_ends_padded))  # [batch, num_heads, num_chunks+1, num_chunks+1]
    
    # Apply decay to propagate states across chunks
    # decay_chunk: [batch, num_heads, num_chunks+1, num_chunks+1]
    # states_with_init: [batch, num_chunks+1, num_heads, head_dim, state_size]
    # new_states[b, i, h, d, s] = sum_j decay_chunk[b, h, i, j] * states_with_init[b, j, h, d, s]
    new_states = torch.einsum('bhij,bjhds->bihds', decay_chunk, states_with_init)
    # [batch, num_chunks+1, num_heads, head_dim, state_size]
    
    states_out = new_states[:, :-1]  # [batch, num_chunks, num_heads, head_dim, state_size]
    final_state = new_states[:, -1]  # [batch, num_heads, head_dim, state_size]
    
    # 4. Compute state -> output conversion (left term)
    state_decay_out = torch.exp(A_cumsum)  # [batch, num_heads, num_chunks, chunk_size]
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)  # [batch, num_chunks, chunk_size, num_heads]
    
    # C_times_states: contract C with states over state_size
    # C_chunked: [batch, num_chunks, chunk_size, num_heads, state_size]
    # states_out: [batch, num_chunks, num_heads, head_dim, state_size]
    # Y_off[b, nc, t, h, d] = sum_s C[b, nc, t, h, s] * states[b, nc, h, d, s] * decay[b, nc, t, h]
    C_times_states = torch.einsum('bcths,bchds->bcthd', C_chunked, states_out)
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    Y_off = C_times_states * state_decay_out_perm[..., None]
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    # 5. Combine intra-chunk and inter-chunk outputs
    y = Y_diag + Y_off
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    # Reshape back: [batch, num_chunks, chunk_size, num_heads, head_dim] -> [batch, seq_len_padded, num_heads, head_dim]
    y = y.reshape(batch_size, -1, num_heads, head_dim)
    
    # Add D residual
    y = y + D_residual
    
    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    
    # Reshape to [batch, seq_len, num_heads * head_dim]
    output = y.reshape(batch_size, seq_len, num_heads * head_dim).to(torch.bfloat16)
    final_state = final_state.to(torch.bfloat16)
    
    return output, final_state
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002905 ms |
| - | Scoring Baseline | 0.500000 | 0.683750 ms |
| - | Reference Implementation | 0.308731 | 1.550087 ms |
