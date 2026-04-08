## Description

Complete vision encoder layer combining multi-head self-attention with pre-LayerNorm, MLP with GELU activation, and residual connections. Performs: pre-norm -> attention -> residual -> pre-norm -> MLP -> residual.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| layer_norm1_weight | [hidden_size] | float32 |
| layer_norm1_bias | [hidden_size] | float32 |
| q_proj_weight | [hidden_size, hidden_size] | float32 |
| q_proj_bias | [hidden_size] | float32 |
| k_proj_weight | [hidden_size, hidden_size] | float32 |
| k_proj_bias | [hidden_size] | float32 |
| v_proj_weight | [hidden_size, hidden_size] | float32 |
| v_proj_bias | [hidden_size] | float32 |
| out_proj_weight | [hidden_size, hidden_size] | float32 |
| out_proj_bias | [hidden_size] | float32 |
| layer_norm2_weight | [hidden_size] | float32 |
| layer_norm2_bias | [hidden_size] | float32 |
| fc1_weight | [intermediate_size, hidden_size] | float32 |
| fc1_bias | [intermediate_size] | float32 |
| fc2_weight | [hidden_size, intermediate_size] | float32 |
| fc2_bias | [hidden_size] | float32 |
| layer_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | float32 |

| # | hidden_size | num_heads | intermediate_size | head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 768 | 12 | 3072 | 64 | 2 | 197 | 0.1632 | 0.0036 |
| 2 | 768 | 12 | 3072 | 64 | 1 | 197 | 0.1567 | 0.0023 |
| 3 | 768 | 12 | 3072 | 64 | 64 | 197 | 0.4390 | 0.1032 |
| 4 | 768 | 12 | 3072 | 64 | 4 | 512 | 0.1978 | 0.0182 |
| 5 | 768 | 12 | 3072 | 64 | 16 | 449 | 0.3232 | 0.0620 |
| 6 | 768 | 12 | 3072 | 64 | 32 | 1423 | 1.3998 | 0.4662 |
| 7 | 768 | 12 | 3072 | 64 | 8 | 919 | 0.3436 | 0.0693 |
| 8 | 768 | 12 | 3072 | 64 | 1 | 2048 | 0.2189 | 0.0235 |
| 9 | 768 | 12 | 3072 | 64 | 8 | 1249 | 0.4309 | 0.0997 |
| 10 | 768 | 12 | 3072 | 64 | 2 | 211 | 0.1635 | 0.0038 |
| 11 | 768 | 12 | 3072 | 64 | 8 | 197 | 0.1867 | 0.0132 |
| 12 | 768 | 12 | 3072 | 64 | 16 | 197 | 0.2311 | 0.0261 |
| 13 | 768 | 12 | 3072 | 64 | 16 | 613 | 0.3944 | 0.0873 |
| 14 | 768 | 12 | 3072 | 64 | 2 | 4096 | 0.4874 | 0.1213 |
| 15 | 768 | 12 | 3072 | 64 | 32 | 541 | 0.5855 | 0.1516 |
| 16 | 768 | 12 | 3072 | 64 | 32 | 197 | 0.3080 | 0.0518 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 12
    head_dim = hidden_size // num_heads
    scale = head_dim ** -0.5
    
    # ===== Attention Block =====
    residual = hidden_states
    
    # LayerNorm1
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm1_weight + layer_norm1_bias
    
    # Q, K, V projections
    queries = torch.matmul(hidden_states, q_proj_weight.t()) + q_proj_bias
    keys = torch.matmul(hidden_states, k_proj_weight.t()) + k_proj_bias
    values = torch.matmul(hidden_states, v_proj_weight.t()) + v_proj_bias
    
    # Reshape for multi-head attention
    queries = queries.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    keys = keys.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    values = values.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention computation
    attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    
    attn_output = torch.matmul(attn_weights, values)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # Output projection
    attn_output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    hidden_states = residual + attn_output
    
    # ===== MLP Block =====
    residual = hidden_states
    
    # LayerNorm2
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm2_weight + layer_norm2_bias
    
    # MLP: fc1 -> quick_gelu -> fc2
    hidden_states = torch.matmul(hidden_states, fc1_weight.t()) + fc1_bias
    # Quick GELU: x * sigmoid(1.702 * x)
    hidden_states = hidden_states * torch.sigmoid(1.702 * hidden_states)
    hidden_states = torch.matmul(hidden_states, fc2_weight.t()) + fc2_bias
    
    # Residual connection
    output = residual + hidden_states
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.035996 ms |
| - | Scoring Baseline | 0.500000 | 0.311769 ms |
| - | Reference Implementation | 0.176075 | 1.343982 ms |
