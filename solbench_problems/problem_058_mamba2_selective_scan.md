## Description

Mamba2 selective state space model core computation including convolution, selective scan with chunking, state space discretization, and gated normalization. This is the dominant computation in the hybrid architecture.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| in_proj_weight | [projection_size, hidden_size] | bfloat16 |
| conv1d_weight | [conv_dim, 1, conv_kernel_size] | bfloat16 |
| conv1d_bias | [conv_dim] | bfloat16 |
| dt_bias | [num_heads] | bfloat16 |
| A_log | [num_heads] | bfloat16 |
| D | [num_heads] | bfloat16 |
| norm_weight | [intermediate_size] | bfloat16 |
| out_proj_weight | [hidden_size, intermediate_size] | bfloat16 |
| time_step_limit_min | scalar | float32 |
| time_step_limit_max | scalar | float32 |
| layer_norm_epsilon | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_heads | head_dim | ssm_state_size | conv_kernel_size | n_groups | chunk_size | intermediate_size | groups_time_state_size | conv_dim | projection_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 2 | 2048 | 5.2293 | 1.9840 |
| 2 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 4 | 384 | 3.2908 | 0.7442 |
| 3 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 4 | 256 | 2.7401 | 0.4963 |
| 4 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 1 | 512 | 0.9711 | 0.2483 |
| 5 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 2 | 1024 | 3.1053 | 0.9922 |
| 6 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 2 | 512 | 2.0327 | 0.4963 |
| 7 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 1 | 4096 | 4.9700 | 1.9840 |
| 8 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 8 | 512 | 7.3637 | 1.9840 |
| 9 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 32 | 128 | 6.8347 | 1.9840 |
| 10 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 1 | 1024 | 1.3384 | 0.4963 |
| 11 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 16 | 384 | 12.4997 | 2.9758 |
| 12 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 2 | 1536 | 4.1836 | 1.4881 |
| 13 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 8 | 128 | 1.8821 | 0.4963 |
| 14 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 4 | 2048 | 10.0506 | 3.9675 |
| 15 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 16 | 128 | 3.5321 | 0.9922 |
| 16 | 8192 | 256 | 64 | 256 | 4 | 8 | 128 | 16384 | 2048 | 20480 | 37120 | 2 | 3072 | 7.5404 | 2.9758 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    in_proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    time_step_limit_min: float,
    time_step_limit_max: float,
    layer_norm_epsilon: float,
):
    # Constants
    hidden_size = 8192
    num_heads = 256
    head_dim = 64
    intermediate_size = 16384
    ssm_state_size = 256
    conv_kernel_size = 4
    n_groups = 8
    chunk_size = 128
    groups_time_state_size = n_groups * ssm_state_size
    conv_dim = intermediate_size + 2 * groups_time_state_size
    
    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device
    
    # 1. Input projection
    projected = torch.matmul(hidden_states, in_proj_weight.t())
    
    # Split projections
    gate_start = projected.shape[-1] - intermediate_size - conv_dim - num_heads
    gate = projected[..., gate_start:gate_start + intermediate_size]
    hidden_B_C = projected[..., gate_start + intermediate_size:gate_start + intermediate_size + conv_dim]
    dt = projected[..., -num_heads:]
    
    # 2. Causal Convolution
    hidden_B_C_t = hidden_B_C.transpose(1, 2)
    conv_out = F.conv1d(
        hidden_B_C_t,
        conv1d_weight,
        conv1d_bias,
        padding=conv_kernel_size - 1,
        groups=conv_dim
    )[..., :seq_len]
    hidden_B_C = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)  # silu
    
    # Split into hidden_states, B, C
    hidden_states_ssm = hidden_B_C[..., :intermediate_size]
    B = hidden_B_C[..., intermediate_size:intermediate_size + groups_time_state_size]
    C = hidden_B_C[..., intermediate_size + groups_time_state_size:]
    
    # 3. Selective State Space Model with Chunking
    
    # Discretize time step
    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit_min, time_step_limit_max)
    
    # Reshape for SSM computation
    hidden_states_ssm = hidden_states_ssm.view(batch_size, seq_len, num_heads, head_dim).float()
    B = B.view(batch_size, seq_len, n_groups, ssm_state_size).float()
    C = C.view(batch_size, seq_len, n_groups, ssm_state_size).float()
    
    # Repeat B and C for all heads in each group
    heads_per_group = num_heads // n_groups
    B = B.repeat(1, 1, heads_per_group, 1)  # (batch, seq, num_heads, ssm_state_size)
    C = C.repeat(1, 1, heads_per_group, 1)  # (batch, seq, num_heads, ssm_state_size)
    
    # Pad to chunk size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    
    def pad_tensor_4d(x, pad_size):
        if pad_size > 0:
            return F.pad(x, (0, 0, 0, 0, 0, pad_size, 0, 0))
        return x
    
    def pad_tensor_3d(x, pad_size):
        if pad_size > 0:
            return F.pad(x, (0, 0, 0, pad_size, 0, 0))
        return x
    
    # D residual (skip connection)
    D_residual = D.float()[..., None] * pad_tensor_4d(hidden_states_ssm, pad_size)
    
    # Discretize x and A
    hidden_states_ssm = hidden_states_ssm * dt[..., None]
    A = -torch.exp(A_log.float()) * dt
    
    # Pad tensors
    hidden_states_ssm_padded = pad_tensor_4d(hidden_states_ssm, pad_size)
    A_padded = pad_tensor_3d(A, pad_size)
    B_padded = pad_tensor_4d(B, pad_size)
    C_padded = pad_tensor_4d(C, pad_size)
    
    padded_seq_len = hidden_states_ssm_padded.shape[1]
    num_chunks = padded_seq_len // chunk_size
    
    # Reshape into chunks
    hidden_states_ssm_chunked = hidden_states_ssm_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)
    A_chunked = A_padded.reshape(batch_size, num_chunks, chunk_size, num_heads)
    B_chunked = B_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, ssm_state_size)
    C_chunked = C_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, ssm_state_size)
    
    # Permute A for computation: (batch, num_heads, num_chunks, chunk_size)
    A_perm = A_chunked.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_perm, dim=-1)
    
    # 3a. Intra-chunk computation (diagonal blocks)
    # Segment sum for causal mask
    def segment_sum(x):
        # x: (batch, num_heads, num_chunks, chunk_size)
        cs = x.size(-1)
        x_expanded = x[..., None].expand(*x.size(), cs)  # (batch, heads, chunks, cs, cs)
        mask = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=-1)
        x_masked = x_expanded.masked_fill(~mask, 0)
        tensor_segsum = torch.cumsum(x_masked, dim=-2)
        mask_diag = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=0)
        return tensor_segsum.masked_fill(~mask_diag, -torch.inf)
    
    L = torch.exp(segment_sum(A_perm))  # (batch, heads, chunks, cs, cs)
    
    # Attention-like weights: G = C^T B
    # C_chunked: (batch, chunks, cs, heads, state)
    # B_chunked: (batch, chunks, cs, heads, state)
    # G: (batch, chunks, cs_l, cs_s, heads)
    G = torch.einsum('bclhn,bcshn->bclsh', C_chunked, B_chunked)
    
    # L: (batch, heads, chunks, cs_l, cs_s) -> need (batch, chunks, cs_l, cs_s, heads)
    L_perm = L.permute(0, 2, 3, 4, 1)  # (batch, chunks, cs_l, cs_s, heads)
    
    # M = G * L (element-wise)
    M = G * L_perm  # (batch, chunks, cs_l, cs_s, heads)
    
    # Y_diag = einsum(M, hidden_states_ssm_chunked)
    # M: (batch, chunks, cs_l, cs_s, heads)
    # hidden_states_ssm_chunked: (batch, chunks, cs_s, heads, head_dim)
    Y_diag = torch.einsum('bclsh,bcshd->bclhd', M, hidden_states_ssm_chunked)
    
    # 3b. Compute states at chunk boundaries
    # decay_states: exp(A_cumsum[:,:,:,-1:] - A_cumsum)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (batch, heads, chunks, cs)
    # decay_states permuted: (batch, chunks, cs, heads)
    decay_states_perm = decay_states.permute(0, 2, 3, 1)
    
    # B_decay = B * decay_states[..., None]
    B_decay = B_chunked * decay_states_perm[..., None]  # (batch, chunks, cs, heads, state)
    
    # states = einsum(hidden_states_ssm_chunked, B_decay)
    # hidden_states_ssm_chunked: (batch, chunks, cs, heads, head_dim)
    # B_decay: (batch, chunks, cs, heads, state)
    # states: (batch, chunks, heads, head_dim, state)
    states = torch.einsum('bcshd,bcshn->bchdn', hidden_states_ssm_chunked, B_decay)
    
    # 3c. Inter-chunk recurrence
    previous_states = torch.zeros_like(states[:, :1])  # (batch, 1, heads, head_dim, state)
    states_with_prev = torch.cat([previous_states, states], dim=1)  # (batch, chunks+1, heads, head_dim, state)
    
    # decay_chunk computation
    # A_cumsum[:, :, :, -1]: (batch, heads, chunks)
    A_chunk_ends = A_cumsum[:, :, :, -1]  # (batch, heads, chunks)
    A_chunk_ends_padded = F.pad(A_chunk_ends, (1, 0))  # (batch, heads, chunks+1)
    
    # segment_sum for inter-chunk decay
    def segment_sum_1d(x):
        # x: (batch, heads, chunks+1)
        cs = x.size(-1)
        x_expanded = x[..., None].expand(*x.size(), cs)
        mask = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=-1)
        x_masked = x_expanded.masked_fill(~mask, 0)
        tensor_segsum = torch.cumsum(x_masked, dim=-2)
        mask_diag = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=0)
        return tensor_segsum.masked_fill(~mask_diag, -torch.inf)
    
    decay_chunk = torch.exp(segment_sum_1d(A_chunk_ends_padded))  # (batch, heads, chunks+1, chunks+1)
    
    # new_states = einsum(decay_chunk, states_with_prev)
    # decay_chunk: (batch, heads, chunks+1, chunks+1) -> need (batch, heads, c_out, c_in)
    # states_with_prev: (batch, chunks+1, heads, head_dim, state)
    # Permute states_with_prev: (batch, heads, chunks+1, head_dim, state)
    states_with_prev_perm = states_with_prev.permute(0, 2, 1, 3, 4)
    
    # new_states: (batch, heads, chunks+1, head_dim, state)
    new_states = torch.einsum('bhcd,bhdin->bhcin', decay_chunk, states_with_prev_perm)
    
    # Remove last chunk and permute back
    new_states = new_states[:, :, :-1, :, :]  # (batch, heads, chunks, head_dim, state)
    states_final = new_states.permute(0, 2, 1, 3, 4)  # (batch, chunks, heads, head_dim, state)
    
    # 3d. State to output (off-diagonal blocks)
    # state_decay_out = exp(A_cumsum)
    state_decay_out = torch.exp(A_cumsum)  # (batch, heads, chunks, cs)
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)  # (batch, chunks, cs, heads)
    
    # Y_off = einsum(C_chunked, states_final, state_decay_out_perm)
    # C_chunked: (batch, chunks, cs, heads, state)
    # states_final: (batch, chunks, heads, head_dim, state)
    # state_decay_out_perm: (batch, chunks, cs, heads)
    # Y_off: (batch, chunks, cs, heads, head_dim)
    Y_off = torch.einsum('bcshn,bchdn,bcsh->bcshd', C_chunked, states_final, state_decay_out_perm)
    
    # Combine intra and inter chunk outputs
    y = Y_diag + Y_off  # (batch, chunks, cs, heads, head_dim)
    y = y.reshape(batch_size, padded_seq_len, num_heads, head_dim)
    
    # Add skip connection
    y = y + D_residual
    
    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len]
    
    # Reshape and convert back to original dtype
    y = y.reshape(batch_size, seq_len, intermediate_size).to(dtype)
    
    # 4. Gated normalization
    group_size = intermediate_size // n_groups
    y_grouped = y.view(batch_size, seq_len, n_groups, group_size)
    variance = y_grouped.float().pow(2).mean(dim=-1, keepdim=True)
    y_normed = y_grouped * torch.rsqrt(variance + layer_norm_epsilon)
    y_normed = y_normed.view(batch_size, seq_len, intermediate_size).to(dtype)
    y_normed = y_normed * norm_weight
    y = y_normed * (gate * torch.sigmoid(gate))  # silu gate
    
    # 5. Output projection
    output = torch.matmul(y, out_proj_weight.t())
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 1.146728 ms |
| - | Scoring Baseline | 0.500000 | 3.871681 ms |
| - | Reference Implementation | 0.160728 | 15.533938 ms |
