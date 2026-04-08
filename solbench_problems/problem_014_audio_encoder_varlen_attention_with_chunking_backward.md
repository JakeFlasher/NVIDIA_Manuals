## Description

Backward pass for variable-length audio encoder attention with chunking. Computes gradients for hidden_states and all projection weights/biases through the attention mechanism, respecting chunk boundaries defined by cu_seqlens.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [total_seq_len, d_model] | bfloat16 |
| hidden_states | [total_seq_len, d_model] | bfloat16 |
| query_states | [1, num_heads, total_seq_len, head_dim] | bfloat16 |
| key_states | [1, num_heads, total_seq_len, head_dim] | bfloat16 |
| value_states | [1, num_heads, total_seq_len, head_dim] | bfloat16 |
| cu_seqlens | [num_chunks_plus_one] | int32 |
| q_weight | [qkv_dim, d_model] | bfloat16 |
| k_weight | [qkv_dim, d_model] | bfloat16 |
| v_weight | [qkv_dim, d_model] | bfloat16 |
| out_weight | [d_model, qkv_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [total_seq_len, d_model] | bfloat16 |
| grad_q_weight | [qkv_dim, d_model] | bfloat16 |
| grad_q_bias | [qkv_dim] | bfloat16 |
| grad_k_weight | [qkv_dim, d_model] | bfloat16 |
| grad_k_bias | [qkv_dim] | bfloat16 |
| grad_v_weight | [qkv_dim, d_model] | bfloat16 |
| grad_v_bias | [qkv_dim] | bfloat16 |
| grad_out_weight | [d_model, qkv_dim] | bfloat16 |
| grad_out_bias | [d_model] | bfloat16 |

| # | d_model | num_heads | head_dim | qkv_dim | total_seq_len | num_chunks | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1280 | 20 | 64 | 1280 | 1280 | 20 | 0.4203 | 0.0197 |
| 2 | 1280 | 20 | 64 | 1280 | 373 | 5 | 0.4635 | 0.0061 |
| 3 | 1280 | 20 | 64 | 1280 | 997 | 14 | 0.6100 | 0.0155 |
| 4 | 1280 | 20 | 64 | 1280 | 1249 | 18 | 0.6466 | 0.0193 |
| 5 | 1280 | 20 | 64 | 1280 | 448 | 7 | 0.3558 | 0.0072 |
| 6 | 1280 | 20 | 64 | 1280 | 1163 | 17 | 0.6509 | 0.0180 |
| 7 | 1280 | 20 | 64 | 1280 | 449 | 6 | 0.5209 | 0.0072 |
| 8 | 1280 | 20 | 64 | 1280 | 1024 | 16 | 0.4289 | 0.0159 |
| 9 | 1280 | 20 | 64 | 1280 | 256 | 4 | 0.3376 | 0.0043 |
| 10 | 1280 | 20 | 64 | 1280 | 640 | 10 | 0.3889 | 0.0101 |
| 11 | 1280 | 20 | 64 | 1280 | 1087 | 15 | 0.6200 | 0.0169 |
| 12 | 1280 | 20 | 64 | 1280 | 211 | 3 | 0.4368 | 0.0042 |
| 13 | 1280 | 20 | 64 | 1280 | 853 | 12 | 0.5932 | 0.0133 |
| 14 | 1280 | 20 | 64 | 1280 | 128 | 1 | 0.3124 | 0.0041 |
| 15 | 1280 | 20 | 64 | 1280 | 613 | 8 | 0.5777 | 0.0097 |
| 16 | 1280 | 20 | 64 | 1280 | 919 | 13 | 0.6429 | 0.0143 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs with valid cu_seqlens for variable-length attention backward."""
    total_seq_len = axes_and_scalars["total_seq_len"]
    num_chunks = axes_and_scalars["num_chunks"]
    d_model = axes_and_scalars["d_model"]
    num_heads = axes_and_scalars["num_heads"]
    head_dim = axes_and_scalars["head_dim"]
    qkv_dim = num_heads * head_dim
    scaling = head_dim ** -0.5
    
    # Generate valid cu_seqlens that sum to total_seq_len
    if num_chunks == 1:
        chunk_lengths = [total_seq_len]
    else:
        # Distribute total_seq_len across chunks
        base_len = total_seq_len // num_chunks
        remainder = total_seq_len % num_chunks
        chunk_lengths = [base_len + (1 if i < remainder else 0) for i in range(num_chunks)]
    
    cu_seqlens = torch.zeros(num_chunks + 1, dtype=torch.int32, device=device)
    for i in range(num_chunks):
        cu_seqlens[i + 1] = cu_seqlens[i] + chunk_lengths[i]
    
    # Generate random tensors
    grad_output = torch.randn(total_seq_len, d_model, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(total_seq_len, d_model, dtype=torch.bfloat16, device=device)
    query_states = torch.randn(1, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key_states = torch.randn(1, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    value_states = torch.randn(1, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(qkv_dim, d_model, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(qkv_dim, d_model, dtype=torch.bfloat16, device=device)
    v_weight = torch.randn(qkv_dim, d_model, dtype=torch.bfloat16, device=device)
    out_weight = torch.randn(d_model, qkv_dim, dtype=torch.bfloat16, device=device)
    
    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "query_states": query_states,
        "key_states": key_states,
        "value_states": value_states,
        "cu_seqlens": cu_seqlens,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "out_weight": out_weight,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
):
    """
    Backward pass for variable-length audio encoder attention.
    
    Computes gradients through:
    1. Output projection
    2. Attention mechanism (per chunk)
    3. QKV projections
    
    Args:
        grad_output: Gradient from upstream (total_seq_len, d_model)
        hidden_states: Original input (total_seq_len, d_model)
        query_states: Saved queries (1, num_heads, total_seq_len, head_dim)
        key_states: Saved keys (1, num_heads, total_seq_len, head_dim)
        value_states: Saved values (1, num_heads, total_seq_len, head_dim)
        cu_seqlens: Cumulative sequence lengths (num_chunks + 1,)
        q_weight: Query projection weight (qkv_dim, d_model)
        k_weight: Key projection weight (qkv_dim, d_model)
        v_weight: Value projection weight (qkv_dim, d_model)
        out_weight: Output projection weight (d_model, qkv_dim)
    
    Returns:
        Tuple of gradients for all differentiable inputs
    """
    total_seq_len = hidden_states.shape[0]
    d_model = hidden_states.shape[1]
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[3]
    scaling = head_dim ** -0.5
    
    # Convert to float32 for numerical stability
    grad_output_f32 = grad_output.to(torch.float32)
    hidden_states_f32 = hidden_states.to(torch.float32)
    query_states_f32 = query_states.to(torch.float32)
    key_states_f32 = key_states.to(torch.float32)
    value_states_f32 = value_states.to(torch.float32)
    q_weight_f32 = q_weight.to(torch.float32)
    k_weight_f32 = k_weight.to(torch.float32)
    v_weight_f32 = v_weight.to(torch.float32)
    out_weight_f32 = out_weight.to(torch.float32)
    
    # Compute chunk lengths from cu_seqlens
    cu_seqlens_cpu = cu_seqlens.cpu()
    chunk_lengths = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).tolist()
    
    # Step 1: Gradient w.r.t. output projection
    # grad_attn_output = grad_output @ out_weight
    grad_attn_output = torch.matmul(grad_output_f32, out_weight_f32)  # (total_seq_len, qkv_dim)
    
    # Recompute attention outputs for grad_out_weight
    query_chunks = query_states_f32.split(chunk_lengths, dim=2)
    key_chunks = key_states_f32.split(chunk_lengths, dim=2)
    value_chunks = value_states_f32.split(chunk_lengths, dim=2)
    
    attn_outputs_recompute = []
    for q, k, v in zip(query_chunks, key_chunks, value_chunks):
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # Compute attention output
        out = torch.matmul(attn_weights, v)
        attn_outputs_recompute.append(out)
    
    attn_output_recompute = torch.cat(attn_outputs_recompute, dim=2)
    attn_output_recompute = attn_output_recompute.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    
    # grad_out_weight = grad_output.T @ attn_output
    grad_out_weight = torch.matmul(grad_output_f32.t(), attn_output_recompute)
    grad_out_bias = grad_output_f32.sum(dim=0)
    
    # Step 2: Reshape grad_attn_output for attention backward
    # (total_seq_len, qkv_dim) -> (1, num_heads, total_seq_len, head_dim)
    grad_attn_output = grad_attn_output.reshape(total_seq_len, num_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(0, 1).unsqueeze(0)
    
    # Split grad_attn_output by chunks
    grad_attn_chunks = grad_attn_output.split(chunk_lengths, dim=2)
    
    # Step 3: Compute gradients for Q, K, V per chunk
    grad_query_chunks = []
    grad_key_chunks = []
    grad_value_chunks = []
    
    for grad_attn, q, k, v in zip(grad_attn_chunks, query_chunks, key_chunks, value_chunks):
        # Recompute attention weights for this chunk
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        # Gradient w.r.t. V: attn_weights^T @ grad_attn
        grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_attn)
        
        # Gradient w.r.t. attn_weights: grad_attn @ V^T
        grad_attn_weights = torch.matmul(grad_attn, v.transpose(-2, -1))
        
        # Backprop through softmax: d_softmax = softmax * (grad - sum(softmax * grad))
        grad_attn_weights = attn_weights * (grad_attn_weights - (attn_weights * grad_attn_weights).sum(dim=-1, keepdim=True))
        
        # Gradient w.r.t. Q and K
        grad_q = torch.matmul(grad_attn_weights, k) * scaling
        grad_k = torch.matmul(grad_attn_weights.transpose(-2, -1), q) * scaling
        
        grad_query_chunks.append(grad_q)
        grad_key_chunks.append(grad_k)
        grad_value_chunks.append(grad_v)
    
    # Concatenate gradients
    grad_query = torch.cat(grad_query_chunks, dim=2)
    grad_key = torch.cat(grad_key_chunks, dim=2)
    grad_value = torch.cat(grad_value_chunks, dim=2)
    
    # Reshape: (1, num_heads, total_seq_len, head_dim) -> (total_seq_len, qkv_dim)
    grad_query = grad_query.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    grad_key = grad_key.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    grad_value = grad_value.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    
    # Step 4: Gradients w.r.t. QKV projection weights and biases
    grad_q_weight = torch.matmul(grad_query.t(), hidden_states_f32)
    grad_q_bias = grad_query.sum(dim=0)
    
    grad_k_weight = torch.matmul(grad_key.t(), hidden_states_f32)
    grad_k_bias = grad_key.sum(dim=0)
    
    grad_v_weight = torch.matmul(grad_value.t(), hidden_states_f32)
    grad_v_bias = grad_value.sum(dim=0)
    
    # Step 5: Gradient w.r.t. hidden_states (sum from Q, K, V projections)
    grad_hidden_states = torch.matmul(grad_query, q_weight_f32) + \
                        torch.matmul(grad_key, k_weight_f32) + \
                        torch.matmul(grad_value, v_weight_f32)
    
    # Convert back to bfloat16
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_q_bias.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_k_bias.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_v_bias.to(torch.bfloat16),
        grad_out_weight.to(torch.bfloat16),
        grad_out_bias.to(torch.bfloat16),
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.010137 ms |
| - | Scoring Baseline | 0.500000 | 0.486128 ms |
| - | Reference Implementation | 0.165309 | 2.515036 ms |
