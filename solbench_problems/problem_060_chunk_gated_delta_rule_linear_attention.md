## Description

Chunk-based Gated Delta Rule linear attention computation that processes sequences in chunks of 64 tokens. Implements delta rule update with exponential decay gating, chunk-wise attention with intra-chunk causal masking, inter-chunk recurrent state propagation, and cumulative decay tracking.
| Name | Shape | Dtype |
| --- | --- | --- |
| query | [batch_size, num_k_heads, seq_len, head_k_dim] | bfloat16 |
| key | [batch_size, num_k_heads, seq_len, head_k_dim] | bfloat16 |
| value | [batch_size, num_v_heads, seq_len, head_v_dim] | bfloat16 |
| g | [batch_size, num_v_heads, seq_len] | bfloat16 |
| beta | [batch_size, num_v_heads, seq_len] | bfloat16 |
| scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, num_v_heads, head_v_dim] | bfloat16 |

| # | num_k_heads | num_v_heads | head_k_dim | head_v_dim | chunk_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 16 | 128 | 128 | 64 | 8 | 373 | 0.6790 | 0.0111 |
| 2 | 4 | 16 | 128 | 128 | 64 | 2 | 512 | 0.4025 | 0.0040 |
| 3 | 4 | 16 | 128 | 128 | 64 | 1 | 131 | 0.2896 | 0.0010 |
| 4 | 4 | 16 | 128 | 128 | 64 | 2 | 541 | 0.4615 | 0.0043 |
| 5 | 4 | 16 | 128 | 128 | 64 | 4 | 293 | 0.4423 | 0.0048 |
| 6 | 4 | 16 | 128 | 128 | 64 | 2 | 2048 | 1.0424 | 0.0142 |
| 7 | 4 | 16 | 128 | 128 | 64 | 4 | 613 | 0.6840 | 0.0091 |
| 8 | 4 | 16 | 128 | 128 | 64 | 64 | 128 | 1.2365 | 0.0322 |
| 9 | 4 | 16 | 128 | 128 | 64 | 2 | 1024 | 0.6063 | 0.0074 |
| 10 | 4 | 16 | 128 | 128 | 64 | 16 | 256 | 0.7784 | 0.0152 |
| 11 | 4 | 16 | 128 | 128 | 64 | 8 | 256 | 0.5179 | 0.0078 |
| 12 | 4 | 16 | 128 | 128 | 64 | 1 | 512 | 0.3545 | 0.0022 |
| 13 | 4 | 16 | 128 | 128 | 64 | 16 | 128 | 0.4976 | 0.0083 |
| 14 | 4 | 16 | 128 | 128 | 64 | 8 | 128 | 0.3577 | 0.0044 |
| 15 | 4 | 16 | 128 | 128 | 64 | 8 | 512 | 0.9212 | 0.0147 |
| 16 | 4 | 16 | 128 | 128 | 64 | 8 | 691 | 1.0816 | 0.0196 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
):
    """
    Chunk-based Gated Delta Rule linear attention.
    
    Args:
        query: [batch, num_k_heads, seq_len, head_k_dim]
        key: [batch, num_k_heads, seq_len, head_k_dim]
        value: [batch, num_v_heads, seq_len, head_v_dim]
        g: [batch, num_v_heads, seq_len] - decay gates
        beta: [batch, num_v_heads, seq_len] - update gates
        scale: attention scale factor
    """
    initial_dtype = query.dtype
    batch_size, num_k_heads, sequence_length, k_head_dim = key.shape
    num_v_heads = value.shape[1]
    v_head_dim = value.shape[-1]
    chunk_size = 64
    
    # L2 normalization
    def l2norm(x, dim=-1, eps=1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm
    
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    
    # Convert to float32 for numerical stability
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    beta = beta.to(torch.float32)
    g = g.to(torch.float32)
    
    # Apply scaling to queries
    query = query * scale
    
    # Pad sequence to multiple of chunk_size
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    
    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    
    # Compute beta-weighted values and keys
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key.unsqueeze(1).expand(-1, num_v_heads // num_k_heads, -1, -1, -1)
    k_beta = k_beta.reshape(batch_size, num_v_heads, total_sequence_length, k_head_dim)
    k_beta = k_beta * beta.unsqueeze(-1)
    
    # Reshape to chunks
    query_chunks = query.unsqueeze(1).expand(-1, num_v_heads // num_k_heads, -1, -1, -1)
    query_chunks = query_chunks.reshape(batch_size, num_v_heads, total_sequence_length, k_head_dim)
    query_chunks = query_chunks.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_head_dim)
    
    key_chunks = key.unsqueeze(1).expand(-1, num_v_heads // num_k_heads, -1, -1, -1)
    key_chunks = key_chunks.reshape(batch_size, num_v_heads, total_sequence_length, k_head_dim)
    key_chunks = key_chunks.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_head_dim)
    
    value_chunks = value.reshape(batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim)
    k_beta_chunks = k_beta.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_head_dim)
    v_beta_chunks = v_beta.reshape(batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim)
    g_chunks = g.reshape(batch_size, num_v_heads, num_chunks, chunk_size)
    
    # Compute cumulative decay within chunks
    g_cumsum = g_chunks.cumsum(dim=-1)
    
    # Create decay mask
    decay_mask = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).exp()
    decay_mask = decay_mask.tril()
    
    # Create causal mask for attention
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0
    )
    
    # Compute intra-chunk attention
    attn = -(k_beta_chunks @ key_chunks.transpose(-1, -2)) * decay_mask
    attn = attn.masked_fill(causal_mask, 0)
    
    # Compute cumulative attention through iterative updates
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    
    # Add identity
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    
    # Transform values and compute cumulative decay for keys
    value_transformed = attn @ v_beta_chunks
    k_cumdecay = attn @ (k_beta_chunks * g_chunks.exp().unsqueeze(-1))
    
    # Initialize recurrent state
    recurrent_state = torch.zeros(
        batch_size, num_v_heads, k_head_dim, v_head_dim,
        dtype=torch.float32, device=value.device
    )
    
    # Output tensor
    output = torch.zeros(
        batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim,
        dtype=torch.float32, device=value.device
    )
    
    # Create upper triangular mask
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1
    )
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        q_chunk = query_chunks[:, :, chunk_idx]
        k_chunk = key_chunks[:, :, chunk_idx]
        v_transformed_chunk = value_transformed[:, :, chunk_idx]
        k_cumdecay_chunk = k_cumdecay[:, :, chunk_idx]
        g_chunk = g_chunks[:, :, chunk_idx]
        decay_mask_chunk = decay_mask[:, :, chunk_idx]
        
        # Intra-chunk attention
        attn_intra = (q_chunk @ k_chunk.transpose(-1, -2)) * decay_mask_chunk
        attn_intra = attn_intra.masked_fill_(upper_mask, 0)
        
        # Value correction from recurrent state
        v_prime = k_cumdecay_chunk @ recurrent_state
        v_new = v_transformed_chunk - v_prime
        
        # Inter-chunk attention
        attn_inter = (q_chunk * g_chunk.exp().unsqueeze(-1)) @ recurrent_state
        
        # Combine
        output[:, :, chunk_idx] = attn_inter + attn_intra @ v_new
        
        # Update recurrent state
        chunk_decay = g_chunk[:, :, -1].exp().unsqueeze(-1).unsqueeze(-1)
        recurrent_state = recurrent_state * chunk_decay
        
        token_decay = (g_chunk[:, :, -1].unsqueeze(-1) - g_chunk).exp().unsqueeze(-1)
        recurrent_state = recurrent_state + (k_chunk * token_decay).transpose(-1, -2) @ v_new
    
    # Reshape output and remove padding
    output = output.reshape(batch_size, num_v_heads, total_sequence_length, v_head_dim)
    output = output[:, :, :sequence_length]
    
    # Transpose to [batch, seq_len, num_v_heads, head_v_dim]
    output = output.transpose(1, 2).contiguous().to(initial_dtype)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.007380 ms |
| - | Scoring Baseline | 0.500000 | 0.590556 ms |
| - | Reference Implementation | 0.126961 | 4.192764 ms |
