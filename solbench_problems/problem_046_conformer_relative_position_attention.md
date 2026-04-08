## Description

Conformer attention with Shaw's relative positional embeddings. Processes sequences in fixed-size context blocks (512 tokens), computes scaled dot-product attention within each block, and adds relative positional bias via learned embeddings. Handles dynamic padding and masking for incomplete blocks.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_dim] | bfloat16 |
| pre_norm_weight | [hidden_dim] | bfloat16 |
| pre_norm_bias | [hidden_dim] | bfloat16 |
| to_q_weight | [inner_dim, hidden_dim] | bfloat16 |
| to_kv_weight | [inner_dim_x2, hidden_dim] | bfloat16 |
| to_out_weight | [hidden_dim, inner_dim] | bfloat16 |
| to_out_bias | [hidden_dim] | bfloat16 |
| rel_pos_emb_weight | [num_rel_pos, dim_head] | bfloat16 |
| scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_dim] | bfloat16 |

| # | hidden_dim | num_heads | dim_head | max_pos_emb | context_size | inner_dim | inner_dim_x2 | num_rel_pos | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 8 | 512 | 0.2819 | 0.0200 |
| 2 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 1 | 8192 | 0.3775 | 0.0395 |
| 3 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 2 | 2048 | 0.2714 | 0.0200 |
| 4 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 8 | 256 | 0.2694 | 0.0176 |
| 5 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 2 | 256 | 0.2449 | 0.0047 |
| 6 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 64 | 256 | 0.5586 | 0.1380 |
| 7 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 1 | 4096 | 0.2738 | 0.0200 |
| 8 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 2 | 1024 | 0.2408 | 0.0102 |
| 9 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 1 | 128 | 0.1595 | 0.0024 |
| 10 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 1 | 131 | 0.1824 | 0.0024 |
| 11 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 16 | 2048 | 0.9511 | 0.1570 |
| 12 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 4 | 541 | 0.3225 | 0.0177 |
| 13 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 16 | 512 | 0.3776 | 0.0395 |
| 14 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 2 | 512 | 0.2513 | 0.0053 |
| 15 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 8 | 773 | 0.4513 | 0.0372 |
| 16 | 1024 | 8 | 128 | 512 | 512 | 1024 | 2048 | 1025 | 4 | 2048 | 0.3763 | 0.0395 |

```python
import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    pre_norm_weight: torch.Tensor,
    pre_norm_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_kv_weight: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    rel_pos_emb_weight: torch.Tensor,
    scale: float,
):
    # Constants
    hidden_dim = 1024
    num_heads = 8
    dim_head = 128
    max_pos_emb = 512
    context_size = 512
    inner_dim = 1024
    
    bsz, num_features, _ = hidden_states.shape
    
    # Pre-normalization (LayerNorm)
    hidden_states = F.layer_norm(hidden_states, (hidden_dim,), pre_norm_weight, pre_norm_bias)
    
    # Calculate blocking parameters
    num_blocks = math.ceil(num_features / context_size)
    remainder = num_features % context_size
    
    # Pad to block boundary if needed
    if remainder > 0:
        pad_amount = context_size - remainder
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_amount), mode='constant', value=0)
    
    # Project to Q, K, V
    query_states = F.linear(hidden_states, to_q_weight)  # (B, S_padded, inner_dim)
    key_value_states = F.linear(hidden_states, to_kv_weight)  # (B, S_padded, inner_dim*2)
    key_states, value_states = key_value_states.chunk(2, dim=-1)
    
    # Reshape into blocks: (B, num_blocks, context_size, num_heads, dim_head)
    query_states = query_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)  # (B, num_blocks, num_heads, context_size, dim_head)
    
    key_states = key_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)
    
    value_states = value_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)
    
    # Compute relative position distances for context window
    device = hidden_states.device
    seq = torch.arange(context_size, device=device)
    relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
    attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + max_pos_emb
    
    # Compute relative positional attention bias
    # rel_pos_emb: (context_size, context_size, dim_head)
    rel_pos_emb = F.embedding(attention_dists, rel_pos_emb_weight)
    
    # Einsum: query @ rel_pos_emb -> positional attention scores
    # (B, M, Nh, C, D) @ (C, C, D) -> (B, M, Nh, C, C)
    pos_attn = torch.einsum(
        'bmhcd,crd->bmhcr', 
        query_states.float(), 
        rel_pos_emb.float()
    ) * scale
    
    # Apply masking for incomplete final block
    if remainder > 0:
        mask = torch.ones(
            context_size, context_size, 
            dtype=torch.bool, 
            device=device
        )
        mask[:remainder, :remainder] = False
        mask_value = -torch.finfo(pos_attn.dtype).max
        pos_attn[:, -1, :].masked_fill_(mask, mask_value)
    
    pos_attn = pos_attn.to(query_states.dtype)
    
    # Scaled dot-product attention with positional bias
    # Force MATH backend to ensure pos_attn bias is applied correctly
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=pos_attn,  # Relative position bias
            scale=scale
        )
    
    # Reshape back: (B, M, Nh, C, D) -> (B, S_padded, inner_dim)
    out = out.transpose(2, 3).reshape(bsz, num_blocks * context_size, inner_dim)
    
    # Remove padding and project to output
    out = out[:, :num_features, :]
    out = F.linear(out, to_out_weight, to_out_bias)
    
    return out
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.018456 ms |
| - | Scoring Baseline | 0.500000 | 0.316401 ms |
| - | Reference Implementation | 0.149224 | 1.781086 ms |
