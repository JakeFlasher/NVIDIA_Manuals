## Description

Backward pass for fused vision attention with cu_seqlens and 2D rotary position embeddings. Computes gradients for hidden_states, QKV projection weights/biases, and output projection weights/biases.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [total_seq_len, hidden_size] | float32 |
| hidden_states | [total_seq_len, hidden_size] | float32 |
| qkv_weight | [qkv_out_size, hidden_size] | float32 |
| qkv_bias | [qkv_out_size] | float32 |
| proj_weight | [hidden_size, hidden_size] | float32 |
| proj_bias | [hidden_size] | float32 |
| cos | [total_seq_len, head_dim] | float32 |
| sin | [total_seq_len, head_dim] | float32 |
| cu_seqlens | [num_seqlens] | int32 |
| attention_dropout | scalar | float32 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [total_seq_len, hidden_size] | float32 |
| grad_qkv_weight | [qkv_out_size, hidden_size] | float32 |
| grad_qkv_bias | [qkv_out_size] | float32 |
| grad_proj_weight | [hidden_size, hidden_size] | float32 |
| grad_proj_bias | [hidden_size] | float32 |

| # | hidden_size | num_heads | head_dim | qkv_out_size | total_seq_len | num_sequences | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3584 | 28 | 128 | 10752 | 392 | 2 | 0.6786 | 0.0634 |
| 2 | 3584 | 28 | 128 | 10752 | 211 | 2 | 0.7789 | 0.0338 |
| 3 | 3584 | 28 | 128 | 10752 | 5880 | 30 | 3.3883 | 0.9452 |
| 4 | 3584 | 28 | 128 | 10752 | 773 | 5 | 1.6185 | 0.1238 |
| 5 | 3584 | 28 | 128 | 10752 | 4704 | 24 | 2.7654 | 0.7562 |
| 6 | 3584 | 28 | 128 | 10752 | 613 | 4 | 1.3169 | 0.0983 |
| 7 | 3584 | 28 | 128 | 10752 | 2048 | 8 | 1.5113 | 0.3324 |
| 8 | 3584 | 28 | 128 | 10752 | 919 | 6 | 1.8494 | 0.1471 |
| 9 | 3584 | 28 | 128 | 10752 | 131 | 1 | 0.5270 | 0.0212 |
| 10 | 3584 | 28 | 128 | 10752 | 1176 | 6 | 1.0791 | 0.1894 |
| 11 | 3584 | 28 | 128 | 10752 | 8192 | 32 | 4.7482 | 1.3283 |
| 12 | 3584 | 28 | 128 | 10752 | 196 | 1 | 0.6864 | 0.0319 |
| 13 | 3584 | 28 | 128 | 10752 | 373 | 3 | 1.0697 | 0.0597 |
| 14 | 3584 | 28 | 128 | 10752 | 1024 | 4 | 1.3344 | 0.1664 |
| 15 | 3584 | 28 | 128 | 10752 | 4096 | 16 | 4.1343 | 0.6644 |
| 16 | 3584 | 28 | 128 | 10752 | 2940 | 15 | 3.0409 | 0.4728 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    total_seq_len = axes_and_scalars["total_seq_len"]
    num_sequences = axes_and_scalars["num_sequences"]
    hidden_size = 3584
    num_heads = 28
    head_dim = 128
    qkv_out_size = hidden_size * 3
    
    # Generate grad_output
    grad_output = torch.randn(total_seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Generate hidden_states
    hidden_states = torch.randn(total_seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Generate QKV weights and biases
    qkv_weight = torch.randn(qkv_out_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    qkv_bias = torch.randn(qkv_out_size, dtype=torch.float32, device=device) * 0.01
    
    # Generate projection weights and biases
    proj_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    proj_bias = torch.randn(hidden_size, dtype=torch.float32, device=device) * 0.01
    
    # Generate RoPE embeddings
    cos = torch.randn(total_seq_len, head_dim, dtype=torch.float32, device=device)
    sin = torch.randn(total_seq_len, head_dim, dtype=torch.float32, device=device)
    
    # Generate cu_seqlens with shape (num_sequences + 1,)
    # Distribute total_seq_len across num_sequences
    if num_sequences <= 1:
        cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)
    else:
        base_len = total_seq_len // num_sequences
        remainder = total_seq_len % num_sequences
        lengths = [base_len] * num_sequences
        for i in range(remainder):
            lengths[i] += 1
        cu_seqlens_list = [0]
        for l in lengths:
            cu_seqlens_list.append(cu_seqlens_list[-1] + l)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    
    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "proj_weight": proj_weight,
        "proj_bias": proj_bias,
        "cos": cos,
        "sin": sin,
        "cu_seqlens": cu_seqlens,
        "attention_dropout": 0.0,
        "scaling": head_dim ** -0.5,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    attention_dropout: float,
    scaling: float,
):
    """Backward pass for fused vision attention with cu_seqlens and 2D RoPE.
    
    Computes gradients for:
    - hidden_states
    - qkv_weight, qkv_bias
    - proj_weight, proj_bias
    """
    seq_length = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_heads = 28
    head_dim = 128
    
    # ============ Forward recomputation ============
    # QKV projection
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
    
    # Apply 2D RoPE
    cos_expanded = cos.unsqueeze(-2).float()
    sin_expanded = sin.unsqueeze(-2).float()
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def rotate_half_inverse(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((x2, -x1), dim=-1)
    
    query_states_float = query_states.float()
    key_states_float = key_states.float()
    
    query_states_rope = (query_states_float * cos_expanded + rotate_half(query_states_float) * sin_expanded).to(query_states.dtype)
    key_states_rope = (key_states_float * cos_expanded + rotate_half(key_states_float) * sin_expanded).to(key_states.dtype)
    
    # Reshape for attention
    query_states_attn = query_states_rope.transpose(0, 1).unsqueeze(0)
    key_states_attn = key_states_rope.transpose(0, 1).unsqueeze(0)
    value_states_attn = value_states.transpose(0, 1).unsqueeze(0)
    
    # Variable-length attention using cu_seqlens
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    
    query_splits = torch.split(query_states_attn, lengths.tolist(), dim=2)
    key_splits = torch.split(key_states_attn, lengths.tolist(), dim=2)
    value_splits = torch.split(value_states_attn, lengths.tolist(), dim=2)
    
    # Compute attention and save weights
    attn_outputs = []
    attn_weights_list = []
    
    for q, k, v in zip(query_splits, key_splits, value_splits):
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_outputs.append(attn_output)
        attn_weights_list.append(attn_weights)
    
    attn_output = torch.cat(attn_outputs, dim=2)
    attn_output_reshaped = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, hidden_size).contiguous()
    
    # ============ Backward computation ============
    
    # Gradient through output projection
    grad_proj_bias = grad_output.sum(dim=0)
    grad_proj_weight = grad_output.t() @ attn_output_reshaped
    grad_attn_output = grad_output @ proj_weight
    
    # Reshape gradient
    grad_attn_output_reshaped = grad_attn_output.reshape(seq_length, num_heads, head_dim)
    grad_attn_output_reshaped = grad_attn_output_reshaped.transpose(0, 1).unsqueeze(0)
    
    # Split gradient by sequence lengths
    grad_attn_output_splits = torch.split(grad_attn_output_reshaped, lengths.tolist(), dim=2)
    
    # Backward through attention for each sequence
    grad_query_splits = []
    grad_key_splits = []
    grad_value_splits = []
    
    for grad_attn_out, attn_weights, q, k, v in zip(
        grad_attn_output_splits, attn_weights_list, query_splits, key_splits, value_splits
    ):
        # Gradient through attention output
        grad_attn_weights = torch.matmul(grad_attn_out, v.transpose(2, 3))
        grad_v = torch.matmul(attn_weights.transpose(2, 3), grad_attn_out)
        
        # Gradient through softmax
        sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
        grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)
        grad_attn_scores = grad_attn_scores * scaling
        
        # Gradient through Q @ K^T
        grad_q = torch.matmul(grad_attn_scores, k)
        grad_k = torch.matmul(grad_attn_scores.transpose(2, 3), q)
        
        grad_query_splits.append(grad_q)
        grad_key_splits.append(grad_k)
        grad_value_splits.append(grad_v)
    
    # Concatenate gradients
    grad_query_states = torch.cat(grad_query_splits, dim=2)
    grad_key_states = torch.cat(grad_key_splits, dim=2)
    grad_value_states = torch.cat(grad_value_splits, dim=2)
    
    # Reshape
    grad_query_states = grad_query_states.squeeze(0).transpose(0, 1)
    grad_key_states = grad_key_states.squeeze(0).transpose(0, 1)
    grad_value_states = grad_value_states.squeeze(0).transpose(0, 1)
    
    # Gradient through 2D RoPE
    grad_query_states_float = grad_query_states.float()
    grad_key_states_float = grad_key_states.float()
    
    grad_query_pre_rope = (
        grad_query_states_float * cos_expanded +
        rotate_half_inverse(grad_query_states_float * sin_expanded)
    ).to(grad_query_states.dtype)
    
    grad_key_pre_rope = (
        grad_key_states_float * cos_expanded +
        rotate_half_inverse(grad_key_states_float * sin_expanded)
    ).to(grad_key_states.dtype)
    
    # Stack gradients
    grad_qkv = torch.stack([grad_query_pre_rope, grad_key_pre_rope, grad_value_states], dim=1)
    grad_qkv = grad_qkv.reshape(seq_length, 3 * hidden_size)
    
    # Gradient through QKV projection
    grad_hidden_states = grad_qkv @ qkv_weight
    grad_qkv_weight = grad_qkv.t() @ hidden_states
    grad_qkv_bias = grad_qkv.sum(dim=0)
    
    return grad_hidden_states, grad_qkv_weight, grad_qkv_bias, grad_proj_weight, grad_proj_bias
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.169941 ms |
| - | Scoring Baseline | 0.500000 | 1.538113 ms |
| - | Reference Implementation | 0.397468 | 2.229789 ms |
