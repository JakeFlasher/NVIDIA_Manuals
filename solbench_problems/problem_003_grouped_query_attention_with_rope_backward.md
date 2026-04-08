## Description

Backward pass for Grouped Query Attention with Rotary Position Embeddings. Computes gradients through output projection, attention computation, RoPE, and Q/K/V projections.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| cos | [batch_size, seq_len, head_dim] | bfloat16 |
| sin | [batch_size, seq_len, head_dim] | bfloat16 |
| q_weight | [q_proj_size, hidden_size] | bfloat16 |
| k_weight | [kv_proj_size, hidden_size] | bfloat16 |
| v_weight | [kv_proj_size, hidden_size] | bfloat16 |
| o_weight | [hidden_size, q_proj_size] | bfloat16 |
| query_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| value_states | [batch_size, num_attention_heads, seq_len, head_dim] | bfloat16 |
| attn_weights | [batch_size, num_attention_heads, seq_len, seq_len] | bfloat16 |
| attn_output | [batch_size, seq_len, q_proj_size] | bfloat16 |
| scaling | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_q_weight | [q_proj_size, hidden_size] | bfloat16 |
| grad_k_weight | [kv_proj_size, hidden_size] | bfloat16 |
| grad_v_weight | [kv_proj_size, hidden_size] | bfloat16 |
| grad_o_weight | [hidden_size, q_proj_size] | bfloat16 |

| # | hidden_size | num_attention_heads | num_key_value_heads | head_dim | q_proj_size | kv_proj_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 691 | 0.6895 | 0.0890 |
| 2 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 2 | 449 | 0.5627 | 0.1117 |
| 3 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 2 | 128 | 0.3251 | 0.0306 |
| 4 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 16 | 256 | 1.2737 | 0.4936 |
| 5 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 1024 | 0.5190 | 0.1379 |
| 6 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 131 | 0.3271 | 0.0212 |
| 7 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 2048 | 1.0716 | 0.3134 |
| 8 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 293 | 0.3786 | 0.0359 |
| 9 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 16 | 128 | 0.6946 | 0.2423 |
| 10 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 4 | 1024 | 1.5647 | 0.5505 |
| 11 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 8 | 384 | 1.0305 | 0.3774 |
| 12 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 128 | 0.3001 | 0.0211 |
| 13 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 512 | 0.3979 | 0.0644 |
| 14 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 1 | 256 | 0.3121 | 0.0312 |
| 15 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 2 | 1024 | 0.8574 | 0.2755 |
| 16 | 5120 | 32 | 8 | 128 | 4096 | 1024 | 4 | 512 | 0.7355 | 0.2565 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for GQA with RoPE.
    
    Computes gradients through:
    1. Output projection
    2. Attention output (softmax @ V)
    3. Attention scores (Q @ K^T)
    4. Repeat KV aggregation
    5. RoPE
    6. Q/K/V projections
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    kv_seq_len = key_states.shape[2]
    
    # 1. Gradient through output projection
    # output = attn_output @ o_weight^T
    # grad_attn_output = grad_output @ o_weight
    # grad_o_weight = grad_output^T @ attn_output
    grad_attn_output = torch.matmul(grad_output, o_weight)
    grad_o_weight = torch.matmul(
        grad_output.reshape(-1, grad_output.shape[-1]).t(),
        attn_output.reshape(-1, attn_output.shape[-1])
    )
    
    # 2. Gradient through reshape and transpose
    # [batch, seq_len, 4096] -> [batch, seq_len, 32, 128] -> [batch, 32, seq_len, 128]
    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, num_attention_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(1, 2)
    
    # 3. Gradient through attention: attn_output = attn_weights @ value_states
    # grad_attn_weights = grad_attn_output @ value_states^T
    # grad_value_states = attn_weights^T @ grad_attn_output
    grad_attn_weights = torch.matmul(grad_attn_output, value_states.transpose(2, 3))
    grad_value_states = torch.matmul(attn_weights.transpose(2, 3), grad_attn_output)
    
    # 4. Gradient through softmax
    # For softmax: grad_input = softmax * (grad_output - sum(grad_output * softmax))
    attn_weights_fp32 = attn_weights.to(torch.float32)
    grad_attn_weights_fp32 = grad_attn_weights.to(torch.float32)
    sum_grad = (grad_attn_weights_fp32 * attn_weights_fp32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights_fp32 * (grad_attn_weights_fp32 - sum_grad)
    grad_attn_scores = grad_attn_scores.to(query_states.dtype)
    
    # 5. Gradient through attention scores: attn_scores = (Q @ K^T) * scaling
    grad_attn_scores = grad_attn_scores * scaling
    
    # 6. Gradient through Q @ K^T
    # grad_Q = grad_attn_scores @ K
    # grad_K = grad_attn_scores^T @ Q
    grad_query_states = torch.matmul(grad_attn_scores, key_states)
    grad_key_states = torch.matmul(grad_attn_scores.transpose(2, 3), query_states)
    
    # 7. Gradient through repeat_kv (aggregate gradients for GQA)
    # Sum over the repeated dimension
    if num_key_value_groups != 1:
        grad_key_states = grad_key_states.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        ).sum(dim=2)
        
        grad_value_states = grad_value_states.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        ).sum(dim=2)
    
    # 8. Gradient through RoPE for queries
    # query_states = query_pre_rope * cos + rotate_half(query_pre_rope) * sin
    # grad_query_pre_rope = grad_query_states * cos + rotate_half_inverse(grad_query_states * sin)
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)
    
    # Query gradient through RoPE
    grad_q_cos = grad_query_states * cos_expanded
    grad_q_sin = grad_query_states * sin_expanded
    
    # Rotate grad_q_sin back (rotate_half_inverse)
    grad_q_sin_1 = grad_q_sin[..., : head_dim // 2]
    grad_q_sin_2 = grad_q_sin[..., head_dim // 2 :]
    grad_q_sin_rotated = torch.cat((grad_q_sin_2, -grad_q_sin_1), dim=-1)
    
    grad_query_states_pre_rope = grad_q_cos + grad_q_sin_rotated
    
    # Key gradient through RoPE
    grad_k_cos = grad_key_states * cos_expanded
    grad_k_sin = grad_key_states * sin_expanded
    
    grad_k_sin_1 = grad_k_sin[..., : head_dim // 2]
    grad_k_sin_2 = grad_k_sin[..., head_dim // 2 :]
    grad_k_sin_rotated = torch.cat((grad_k_sin_2, -grad_k_sin_1), dim=-1)
    
    grad_key_states_pre_rope = grad_k_cos + grad_k_sin_rotated
    
    # Value gradient (no RoPE applied to values)
    grad_value_states_pre_rope = grad_value_states
    
    # 9. Gradient through transpose and reshape for Q/K/V
    # [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim] -> [batch, seq_len, proj_size]
    grad_query_states_pre_rope = grad_query_states_pre_rope.transpose(1, 2).contiguous()
    grad_query_proj = grad_query_states_pre_rope.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    grad_key_states_pre_rope = grad_key_states_pre_rope.transpose(1, 2).contiguous()
    grad_key_proj = grad_key_states_pre_rope.reshape(batch_size, seq_len, num_key_value_heads * head_dim)
    
    grad_value_states_pre_rope = grad_value_states_pre_rope.transpose(1, 2).contiguous()
    grad_value_proj = grad_value_states_pre_rope.reshape(batch_size, seq_len, num_key_value_heads * head_dim)
    
    # 10. Gradient through Q/K/V projections
    # Q = hidden_states @ q_weight^T
    # grad_hidden_states_q = grad_Q @ q_weight
    # grad_q_weight = grad_Q^T @ hidden_states
    grad_hidden_states_q = torch.matmul(grad_query_proj, q_weight)
    grad_q_weight = torch.matmul(
        grad_query_proj.reshape(-1, grad_query_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    
    grad_hidden_states_k = torch.matmul(grad_key_proj, k_weight)
    grad_k_weight = torch.matmul(
        grad_key_proj.reshape(-1, grad_key_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    
    grad_hidden_states_v = torch.matmul(grad_value_proj, v_weight)
    grad_v_weight = torch.matmul(
        grad_value_proj.reshape(-1, grad_value_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )
    
    # Sum gradients for hidden_states from all three projection branches
    grad_hidden_states = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v
    
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_o_weight.to(torch.bfloat16),
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.113372 ms |
| - | Scoring Baseline | 0.500000 | 0.601202 ms |
| 1st place | AKO4ALL_L2 | 0.452743 | 0.684525 ms |
| - | Reference Implementation | 0.356818 | 0.977963 ms |
