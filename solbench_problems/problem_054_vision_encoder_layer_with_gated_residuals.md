## Description

Complete vision encoder layer with self-attention, MLP, layer norms, and tanh-gated residual connections. This is the dominant computational pattern in the vision encoder, appearing in 8 global vision layers with gating. The gated residuals use tanh activation on learnable gate parameters to modulate the contribution of attention and MLP outputs.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_state | [batch_size, seq_len, hidden_size] | bfloat16 |
| input_layernorm_weight | [hidden_size] | bfloat16 |
| input_layernorm_bias | [hidden_size] | bfloat16 |
| q_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| k_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| v_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| o_proj_weight | [hidden_size, hidden_size] | bfloat16 |
| post_attention_layernorm_weight | [hidden_size] | bfloat16 |
| post_attention_layernorm_bias | [hidden_size] | bfloat16 |
| fc1_weight | [intermediate_size, hidden_size] | bfloat16 |
| fc1_bias | [intermediate_size] | bfloat16 |
| fc2_weight | [hidden_size, intermediate_size] | bfloat16 |
| fc2_bias | [hidden_size] | bfloat16 |
| gate_attn | [1] | bfloat16 |
| gate_ffn | [1] | bfloat16 |
| norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, hidden_size] | bfloat16 |

| # | hidden_size | num_heads | intermediate_size | head_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1280 | 16 | 5120 | 80 | 1 | 4096 | 0.4639 | 0.1367 |
| 2 | 1280 | 16 | 5120 | 80 | 2 | 4100 | 0.7418 | 0.2735 |
| 3 | 1280 | 16 | 5120 | 80 | 1 | 1571 | 0.2623 | 0.0415 |
| 4 | 1280 | 16 | 5120 | 80 | 4 | 4100 | 1.3869 | 0.5465 |
| 5 | 1280 | 16 | 5120 | 80 | 1 | 4100 | 0.4732 | 0.1369 |
| 6 | 1280 | 16 | 5120 | 80 | 4 | 1024 | 0.3646 | 0.1012 |
| 7 | 1280 | 16 | 5120 | 80 | 16 | 256 | 0.3459 | 0.0923 |
| 8 | 1280 | 16 | 5120 | 80 | 1 | 131 | 0.2273 | 0.0056 |
| 9 | 1280 | 16 | 5120 | 80 | 16 | 512 | 0.5199 | 0.1901 |
| 10 | 1280 | 16 | 5120 | 80 | 1 | 128 | 0.2367 | 0.0056 |
| 11 | 1280 | 16 | 5120 | 80 | 1 | 3089 | 0.3979 | 0.0944 |
| 12 | 1280 | 16 | 5120 | 80 | 4 | 256 | 0.2268 | 0.0234 |
| 13 | 1280 | 16 | 5120 | 80 | 1 | 997 | 0.2304 | 0.0249 |
| 14 | 1280 | 16 | 5120 | 80 | 1 | 1024 | 0.2316 | 0.0256 |
| 15 | 1280 | 16 | 5120 | 80 | 4 | 512 | 0.2512 | 0.0478 |
| 16 | 1280 | 16 | 5120 | 80 | 2 | 2048 | 0.4015 | 0.1130 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_state: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    input_layernorm_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    post_attention_layernorm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    gate_attn: torch.Tensor,
    gate_ffn: torch.Tensor,
    norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_state.shape
    num_heads = 16
    head_dim = hidden_size // num_heads
    scaling = head_dim ** -0.5
    
    # Self-attention block with gated residual
    residual = hidden_state
    
    # Input layer norm
    hidden_state_fp32 = hidden_state.to(torch.float32)
    mean = hidden_state_fp32.mean(dim=-1, keepdim=True)
    var = ((hidden_state_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_state_normed = (hidden_state_fp32 - mean) / torch.sqrt(var + norm_eps)
    hidden_state_normed = hidden_state_normed.to(hidden_state.dtype)
    hidden_state = hidden_state_normed * input_layernorm_weight + input_layernorm_bias
    
    # Compute Q, K, V
    query = torch.matmul(hidden_state, q_proj_weight.t())
    key = torch.matmul(hidden_state, k_proj_weight.t())
    value = torch.matmul(hidden_state, v_proj_weight.t())
    
    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    
    # Reshape and project
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, o_proj_weight.t())
    
    # Gated residual connection for attention
    hidden_state = residual + torch.tanh(gate_attn) * attn_output
    
    # MLP block with gated residual
    residual = hidden_state
    
    # Post-attention layer norm
    hidden_state_fp32 = hidden_state.to(torch.float32)
    mean = hidden_state_fp32.mean(dim=-1, keepdim=True)
    var = ((hidden_state_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    hidden_state_normed = (hidden_state_fp32 - mean) / torch.sqrt(var + norm_eps)
    hidden_state_normed = hidden_state_normed.to(hidden_state.dtype)
    hidden_state = hidden_state_normed * post_attention_layernorm_weight + post_attention_layernorm_bias
    
    # MLP with GELU activation
    hidden_state = torch.matmul(hidden_state, fc1_weight.t()) + fc1_bias
    hidden_state = F.gelu(hidden_state)
    hidden_state = torch.matmul(hidden_state, fc2_weight.t()) + fc2_bias
    
    # Gated residual connection for MLP
    output = residual + torch.tanh(gate_ffn) * hidden_state
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.062119 ms |
| - | Scoring Baseline | 0.500000 | 0.365515 ms |
| - | Reference Implementation | 0.201608 | 1.446345 ms |
