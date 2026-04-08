## Description

Variable-length vision attention using cumulative sequence lengths (cu_seqlens). Processes multiple images/videos in a single batch by splitting Q/K/V according to cu_seqlens boundaries, applying rotary position embeddings, performing separate attention computations for each sequence, then concatenating results. Used in vision transformer layers for efficient batched inference with mixed image/video inputs.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [total_seq_len, embed_dim] | bfloat16 |
| cu_seqlens | [num_sequences] | int64 |
| cos | [total_seq_len, num_heads, head_dim] | bfloat16 |
| sin | [total_seq_len, num_heads, head_dim] | bfloat16 |
| qkv_weight | [qkv_dim, embed_dim] | bfloat16 |
| qkv_bias | [qkv_dim] | bfloat16 |
| proj_weight | [embed_dim, embed_dim] | bfloat16 |
| proj_bias | [embed_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [total_seq_len, embed_dim] | bfloat16 |

| # | embed_dim | num_heads | head_dim | qkv_dim | total_seq_len | num_sequences | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1152 | 16 | 72 | 3456 | 4096 | 64 | 0.1962 | 0.0251 |
| 2 | 1152 | 16 | 72 | 3456 | 3719 | 30 | 0.2173 | 0.0234 |
| 3 | 1152 | 16 | 72 | 3456 | 512 | 8 | 0.2040 | 0.0035 |
| 4 | 1152 | 16 | 72 | 3456 | 373 | 4 | 0.1687 | 0.0027 |
| 5 | 1152 | 16 | 72 | 3456 | 7168 | 28 | 0.2332 | 0.0471 |
| 6 | 1152 | 16 | 72 | 3456 | 3089 | 25 | 0.2274 | 0.0195 |
| 7 | 1152 | 16 | 72 | 3456 | 1801 | 15 | 0.2190 | 0.0115 |
| 8 | 1152 | 16 | 72 | 3456 | 1152 | 6 | 0.1985 | 0.0077 |
| 9 | 1152 | 16 | 72 | 3456 | 4093 | 35 | 0.2241 | 0.0256 |
| 10 | 1152 | 16 | 72 | 3456 | 1024 | 4 | 0.1947 | 0.0071 |
| 11 | 1152 | 16 | 72 | 3456 | 293 | 3 | 0.2725 | 0.0022 |
| 12 | 1152 | 16 | 72 | 3456 | 2447 | 20 | 0.2220 | 0.0155 |
| 13 | 1152 | 16 | 72 | 3456 | 4096 | 16 | 0.2849 | 0.0271 |
| 14 | 1152 | 16 | 72 | 3456 | 541 | 5 | 0.2730 | 0.0037 |
| 15 | 1152 | 16 | 72 | 3456 | 211 | 2 | 0.2594 | 0.0017 |
| 16 | 1152 | 16 | 72 | 3456 | 2048 | 8 | 0.1863 | 0.0137 |

```python
import torch
import torch.nn.functional as F
from typing import Tuple

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid cu_seqlens that sum to total_seq_len."""
    total_seq_len = axes_and_scalars["total_seq_len"]
    num_sequences = axes_and_scalars["num_sequences"]
    embed_dim = axes_and_scalars["embed_dim"]
    num_heads = axes_and_scalars["num_heads"]
    head_dim = axes_and_scalars["head_dim"]
    qkv_dim = embed_dim * 3
    
    # Generate random sequence lengths that sum to total_seq_len
    # Ensure each sequence has at least 1 token
    if num_sequences >= total_seq_len:
        # Edge case: more sequences than tokens, give 1 token each to first total_seq_len sequences
        lengths = [1] * total_seq_len + [0] * (num_sequences - total_seq_len)
    else:
        # Distribute tokens among sequences
        base_len = total_seq_len // num_sequences
        remainder = total_seq_len % num_sequences
        lengths = [base_len] * num_sequences
        for i in range(remainder):
            lengths[i] += 1
    
    # Create cumulative sequence lengths (excluding initial 0)
    cu_seqlens = torch.cumsum(torch.tensor(lengths, dtype=torch.int64, device=device), dim=0)
    
    # Generate other inputs
    hidden_states = torch.randn(total_seq_len, embed_dim, device=device, dtype=torch.bfloat16)
    cos = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    sin = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    qkv_weight = torch.randn(qkv_dim, embed_dim, device=device, dtype=torch.bfloat16) * 0.02
    qkv_bias = torch.zeros(qkv_dim, device=device, dtype=torch.bfloat16)
    proj_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=torch.bfloat16) * 0.02
    proj_bias = torch.zeros(embed_dim, device=device, dtype=torch.bfloat16)
    
    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens,
        "cos": cos,
        "sin": sin,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "proj_weight": proj_weight,
        "proj_bias": proj_bias,
    }

def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
) -> torch.Tensor:
    """Variable-length vision attention with cu_seqlens."""
    embed_dim = 1152
    num_heads = 16
    head_dim = 72
    scaling = head_dim ** -0.5
    
    seq_length = hidden_states.shape[0]
    
    # QKV projection: [seq_len, embed_dim] -> [seq_len, 3 * embed_dim]
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
    # Each is now [seq_len, num_heads, head_dim]
    
    # Apply rotary position embeddings
    query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    
    # Build cu_seqlens with prepended 0 for splitting
    cu_seqlens_with_zero = torch.cat([torch.zeros(1, dtype=torch.int64, device=cu_seqlens.device), cu_seqlens])
    
    # Compute lengths for each sequence
    lengths = (cu_seqlens_with_zero[1:] - cu_seqlens_with_zero[:-1]).tolist()
    
    # Filter out zero-length sequences
    lengths = [l for l in lengths if l > 0]
    
    if len(lengths) == 0:
        # No valid sequences, return zeros
        return torch.zeros(seq_length, embed_dim, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Split Q, K, V by sequence lengths
    query_splits = torch.split(query_states, lengths, dim=0)
    key_splits = torch.split(key_states, lengths, dim=0)
    value_splits = torch.split(value_states, lengths, dim=0)
    
    # Process each sequence independently
    attn_outputs = []
    for q, k, v in zip(query_splits, key_splits, value_splits):
        # q, k, v: [seq_len_i, num_heads, head_dim]
        # Reshape for batch processing: [1, num_heads, seq_len_i, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        
        # Compute attention: [1, num_heads, seq_len_i, seq_len_i]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply attention to values: [1, num_heads, seq_len_i, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: [seq_len_i, num_heads, head_dim]
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_outputs.append(attn_output)
    
    # Concatenate all sequences: [total_seq_len, num_heads, head_dim]
    attn_output = torch.cat(attn_outputs, dim=0)
    
    # Reshape and project: [total_seq_len, embed_dim]
    attn_output = attn_output.reshape(seq_length, embed_dim)
    attn_output = F.linear(attn_output, proj_weight, proj_bias)
    
    return attn_output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.009737 ms |
| - | Scoring Baseline | 0.500000 | 0.221497 ms |
| - | Reference Implementation | 0.202975 | 1.014590 ms |
