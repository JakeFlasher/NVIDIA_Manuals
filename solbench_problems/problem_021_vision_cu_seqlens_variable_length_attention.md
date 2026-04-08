## Description

Vision encoder attention with variable-length sequence support using cumulative sequence lengths (cu_seqlens). Processes multiple images/videos with different resolutions in a single batch by using cu_seqlens to define sequence boundaries. Includes QKV projection, rotary position embeddings, scaled dot-product attention per sequence, and output projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [total_seq_len, hidden_size] | float32 |
| cu_seqlens | [num_seqs] | int64 |
| cos | [total_seq_len, head_dim] | float32 |
| sin | [total_seq_len, head_dim] | float32 |
| qkv_weight | [qkv_out_size, hidden_size] | float32 |
| qkv_bias | [qkv_out_size] | float32 |
| proj_weight | [hidden_size, hidden_size] | float32 |
| proj_bias | [hidden_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [total_seq_len, hidden_size] | float32 |

| # | hidden_size | num_heads | head_dim | qkv_out_size | total_seq_len | num_seqs | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1280 | 16 | 80 | 3840 | 512 | 4 | 0.4370 | 0.0044 |
| 2 | 1280 | 16 | 80 | 3840 | 691 | 6 | 0.5372 | 0.0058 |
| 3 | 1280 | 16 | 80 | 3840 | 1152 | 10 | 0.8487 | 0.0095 |
| 4 | 1280 | 16 | 80 | 3840 | 2304 | 13 | 1.4295 | 0.0191 |
| 5 | 1280 | 16 | 80 | 3840 | 293 | 4 | 0.3681 | 0.0027 |
| 6 | 1280 | 16 | 80 | 3840 | 997 | 8 | 0.7153 | 0.0085 |
| 7 | 1280 | 16 | 80 | 3840 | 3072 | 13 | 1.8296 | 0.0273 |
| 8 | 1280 | 16 | 80 | 3840 | 211 | 4 | 0.2947 | 0.0020 |
| 9 | 1280 | 16 | 80 | 3840 | 449 | 5 | 0.4747 | 0.0038 |
| 10 | 1280 | 16 | 80 | 3840 | 768 | 7 | 0.6157 | 0.0064 |
| 11 | 1280 | 16 | 80 | 3840 | 2851 | 15 | 1.8556 | 0.0234 |
| 12 | 1280 | 16 | 80 | 3840 | 1721 | 11 | 1.0432 | 0.0152 |
| 13 | 1280 | 16 | 80 | 3840 | 4608 | 19 | 2.6301 | 0.0388 |
| 14 | 1280 | 16 | 80 | 3840 | 1571 | 10 | 1.0061 | 0.0132 |
| 15 | 1280 | 16 | 80 | 3840 | 128 | 2 | 0.1945 | 0.0014 |
| 16 | 1280 | 16 | 80 | 3840 | 192 | 3 | 0.2704 | 0.0018 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs with valid cu_seqlens structure."""
    total_seq_len = axes_and_scalars['total_seq_len']
    num_seqs = axes_and_scalars['num_seqs']
    hidden_size = 1280
    head_dim = 80
    qkv_out_size = hidden_size * 3
    
    # Generate random hidden states
    hidden_states = torch.randn(total_seq_len, hidden_size, device=device, dtype=torch.float32)
    
    # Generate valid cu_seqlens: must start with 0 and end with total_seq_len
    # num_seqs includes the final boundary, so we have (num_seqs - 1) actual sequences
    if num_seqs <= 1:
        cu_seqlens = torch.tensor([0], dtype=torch.int64, device=device)
    else:
        # Generate (num_seqs - 1) random split points, then sort
        num_boundaries = num_seqs - 2  # internal boundaries
        if num_boundaries > 0 and total_seq_len > 1:
            # Generate random boundaries between 1 and total_seq_len-1
            boundaries = torch.randint(1, max(2, total_seq_len), (num_boundaries,), device=device, dtype=torch.int64)
            boundaries = torch.unique(boundaries)
            boundaries = torch.sort(boundaries)[0]
            cu_seqlens = torch.cat([
                torch.tensor([0], dtype=torch.int64, device=device),
                boundaries,
                torch.tensor([total_seq_len], dtype=torch.int64, device=device)
            ])
        else:
            cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int64, device=device)
        
        # Ensure we have exactly num_seqs elements
        while cu_seqlens.shape[0] < num_seqs:
            # Add more boundaries if needed
            mid = (cu_seqlens[-2] + cu_seqlens[-1]) // 2
            if mid > cu_seqlens[-2] and mid < cu_seqlens[-1]:
                cu_seqlens = torch.cat([cu_seqlens[:-1], torch.tensor([mid], dtype=torch.int64, device=device), cu_seqlens[-1:]])
            else:
                break
        cu_seqlens = cu_seqlens[:num_seqs]
        # Ensure last element is total_seq_len
        cu_seqlens[-1] = total_seq_len
    
    # Generate rotary embeddings
    cos = torch.randn(total_seq_len, head_dim, device=device, dtype=torch.float32)
    sin = torch.randn(total_seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Generate projection weights
    qkv_weight = torch.randn(qkv_out_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    qkv_bias = torch.zeros(qkv_out_size, device=device, dtype=torch.float32)
    proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    proj_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    return {
        'hidden_states': hidden_states,
        'cu_seqlens': cu_seqlens,
        'cos': cos,
        'sin': sin,
        'qkv_weight': qkv_weight,
        'qkv_bias': qkv_bias,
        'proj_weight': proj_weight,
        'proj_bias': proj_bias,
    }

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
):
    """
    Vision attention with variable-length sequences via cu_seqlens.
    
    Args:
        hidden_states: (total_seq_len, hidden_size)
        cu_seqlens: (num_seqs,) cumulative sequence lengths, first element is 0
        cos: (total_seq_len, head_dim) rotary cosine embeddings
        sin: (total_seq_len, head_dim) rotary sine embeddings
        qkv_weight: (3 * hidden_size, hidden_size)
        qkv_bias: (3 * hidden_size,)
        proj_weight: (hidden_size, hidden_size)
        proj_bias: (hidden_size,)
    """
    hidden_size = 1280
    num_heads = 16
    head_dim = 80
    scaling = head_dim ** -0.5
    
    total_seq_len = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # QKV projection: (total_seq_len, hidden_size) -> (total_seq_len, 3 * hidden_size)
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    
    # Reshape to (total_seq_len, 3, num_heads, head_dim)
    qkv = qkv.reshape(total_seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(1, 0, 2, 3)  # (3, total_seq_len, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(0)
    
    # Apply rotary position embeddings
    # query_states, key_states: (total_seq_len, num_heads, head_dim)
    # cos, sin: (total_seq_len, head_dim)
    
    q_float = query_states.float()
    k_float = key_states.float()
    cos_expanded = cos.unsqueeze(1).float()  # (total_seq_len, 1, head_dim)
    sin_expanded = sin.unsqueeze(1).float()  # (total_seq_len, 1, head_dim)
    
    # Split and rotate for query
    q1 = q_float[..., :head_dim // 2]
    q2 = q_float[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    q_embed = (q_float * cos_expanded) + (q_rotated * sin_expanded)
    query_states = q_embed.to(dtype)
    
    # Split and rotate for key
    k1 = k_float[..., :head_dim // 2]
    k2 = k_float[..., head_dim // 2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    k_embed = (k_float * cos_expanded) + (k_rotated * sin_expanded)
    key_states = k_embed.to(dtype)
    
    # Reshape for attention: (total_seq_len, num_heads, head_dim) -> (1, num_heads, total_seq_len, head_dim)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    
    # Compute sequence lengths from cu_seqlens
    num_seqs = cu_seqlens.shape[0]
    
    # Process each sequence separately
    attn_outputs = []
    for i in range(num_seqs - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        
        if seq_len == 0:
            continue
        
        # Extract sequence
        q_seq = query_states[:, :, start:end, :]  # (1, num_heads, seq_len, head_dim)
        k_seq = key_states[:, :, start:end, :]
        v_seq = value_states[:, :, start:end, :]
        
        # Compute attention scores
        attn_weights = torch.matmul(q_seq, k_seq.transpose(2, 3)) * scaling
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v_seq)  # (1, num_heads, seq_len, head_dim)
        attn_out = attn_out.transpose(1, 2)  # (1, seq_len, num_heads, head_dim)
        attn_outputs.append(attn_out)
    
    # Concatenate outputs: (1, total_seq_len, num_heads, head_dim)
    if len(attn_outputs) > 0:
        attn_output = torch.cat(attn_outputs, dim=1)
    else:
        attn_output = torch.zeros(1, 0, num_heads, head_dim, device=device, dtype=dtype)
    
    # Reshape: (total_seq_len, hidden_size)
    attn_output = attn_output.reshape(total_seq_len, hidden_size).contiguous()
    
    # Output projection
    output = F.linear(attn_output, proj_weight, proj_bias)
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.007298 ms |
| - | Scoring Baseline | 0.500000 | 0.698139 ms |
| - | Reference Implementation | 0.389168 | 1.093260 ms |
