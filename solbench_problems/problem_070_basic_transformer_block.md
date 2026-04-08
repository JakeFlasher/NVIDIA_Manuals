## Description

Complete BasicTransformerBlock combining self-attention, cross-attention, and feed-forward network with GEGLU activation. This is the fundamental building block in SDXL Refiner's UNet architecture, processing spatial features through self-attention (spatial coherence), cross-attention (text/image conditioning), and MLP with residual connections.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, spatial_seq_len, dim] | float32 |
| encoder_hidden_states | [batch_size, encoder_seq_len, cross_attention_dim] | float32 |
| norm1_weight | [dim] | float32 |
| norm1_bias | [dim] | float32 |
| attn1_to_q_weight | [inner_dim, dim] | float32 |
| attn1_to_k_weight | [inner_dim, dim] | float32 |
| attn1_to_v_weight | [inner_dim, dim] | float32 |
| attn1_to_out_weight | [dim, inner_dim] | float32 |
| attn1_to_out_bias | [dim] | float32 |
| norm2_weight | [dim] | float32 |
| norm2_bias | [dim] | float32 |
| attn2_to_q_weight | [inner_dim, dim] | float32 |
| attn2_to_k_weight | [inner_dim, cross_attention_dim] | float32 |
| attn2_to_v_weight | [inner_dim, cross_attention_dim] | float32 |
| attn2_to_out_weight | [dim, inner_dim] | float32 |
| attn2_to_out_bias | [dim] | float32 |
| norm3_weight | [dim] | float32 |
| norm3_bias | [dim] | float32 |
| ff_linear1_weight | [ff_gate_dim, dim] | float32 |
| ff_linear1_bias | [ff_gate_dim] | float32 |
| ff_linear2_weight | [dim, ff_intermediate] | float32 |
| ff_linear2_bias | [dim] | float32 |
| norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, spatial_seq_len, dim] | float32 |

| # | dim | num_attention_heads | attention_head_dim | cross_attention_dim | inner_dim | ff_intermediate | ff_gate_dim | batch_size | spatial_seq_len | encoder_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 2 | 613 | 77 | 0.5325 | 0.0758 |
| 2 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 2 | 4093 | 77 | 3.3267 | 0.7358 |
| 3 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 16 | 256 | 77 | 1.0670 | 0.2476 |
| 4 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 1 | 4096 | 77 | 1.7991 | 0.3685 |
| 5 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 4 | 256 | 77 | 0.4586 | 0.0622 |
| 6 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 2 | 512 | 77 | 0.4700 | 0.0628 |
| 7 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 8 | 1024 | 77 | 1.8711 | 0.5282 |
| 8 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 32 | 256 | 77 | 1.5698 | 0.4949 |
| 9 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 2 | 449 | 77 | 0.4830 | 0.0548 |
| 10 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 1 | 997 | 77 | 0.5137 | 0.0644 |
| 11 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 1 | 2048 | 77 | 0.8182 | 0.1493 |
| 12 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 4 | 1024 | 77 | 1.1024 | 0.2643 |
| 13 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 4 | 1879 | 77 | 2.2733 | 0.5363 |
| 14 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 2 | 1024 | 77 | 0.6986 | 0.1323 |
| 15 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 64 | 128 | 77 | 1.6295 | 0.5127 |
| 16 | 1280 | 160 | 24 | 1280 | 3840 | 5120 | 10240 | 4 | 512 | 77 | 0.6384 | 0.1251 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    attn1_to_q_weight: torch.Tensor,
    attn1_to_k_weight: torch.Tensor,
    attn1_to_v_weight: torch.Tensor,
    attn1_to_out_weight: torch.Tensor,
    attn1_to_out_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    attn2_to_q_weight: torch.Tensor,
    attn2_to_k_weight: torch.Tensor,
    attn2_to_v_weight: torch.Tensor,
    attn2_to_out_weight: torch.Tensor,
    attn2_to_out_bias: torch.Tensor,
    norm3_weight: torch.Tensor,
    norm3_bias: torch.Tensor,
    ff_linear1_weight: torch.Tensor,
    ff_linear1_bias: torch.Tensor,
    ff_linear2_weight: torch.Tensor,
    ff_linear2_bias: torch.Tensor,
    norm_eps: float,
):
    # Constants
    num_attention_heads = 160
    attention_head_dim = 24
    inner_dim = num_attention_heads * attention_head_dim
    scale = attention_head_dim ** -0.5
    
    batch_size = hidden_states.shape[0]
    spatial_seq_len = hidden_states.shape[1]
    encoder_seq_len = encoder_hidden_states.shape[1]
    
    # ============ Self-Attention Block ============
    # LayerNorm1
    mean1 = hidden_states.mean(dim=-1, keepdim=True)
    var1 = ((hidden_states - mean1) ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = (hidden_states - mean1) / torch.sqrt(var1 + norm_eps)
    norm_hidden_states = norm_hidden_states * norm1_weight + norm1_bias
    
    # QKV projections for self-attention
    query = F.linear(norm_hidden_states, attn1_to_q_weight)
    key = F.linear(norm_hidden_states, attn1_to_k_weight)
    value = F.linear(norm_hidden_states, attn1_to_v_weight)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query = query.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    key = key.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    value = value.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    
    # Compute attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    
    # Apply attention to values
    attn_output = torch.matmul(attention_probs, value)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, inner_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, attn1_to_out_weight, attn1_to_out_bias)
    
    # Residual connection
    hidden_states = hidden_states + attn_output
    
    # ============ Cross-Attention Block ============
    # LayerNorm2
    mean2 = hidden_states.mean(dim=-1, keepdim=True)
    var2 = ((hidden_states - mean2) ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = (hidden_states - mean2) / torch.sqrt(var2 + norm_eps)
    norm_hidden_states = norm_hidden_states * norm2_weight + norm2_bias
    
    # QKV projections for cross-attention
    query = F.linear(norm_hidden_states, attn2_to_q_weight)
    key = F.linear(encoder_hidden_states, attn2_to_k_weight)
    value = F.linear(encoder_hidden_states, attn2_to_v_weight)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query = query.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    key = key.view(batch_size, encoder_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    value = value.view(batch_size, encoder_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    
    # Compute attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    
    # Apply attention to values
    attn_output = torch.matmul(attention_probs, value)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, inner_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, attn2_to_out_weight, attn2_to_out_bias)
    
    # Residual connection
    hidden_states = hidden_states + attn_output
    
    # ============ Feed-Forward Block ============
    # LayerNorm3
    mean3 = hidden_states.mean(dim=-1, keepdim=True)
    var3 = ((hidden_states - mean3) ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = (hidden_states - mean3) / torch.sqrt(var3 + norm_eps)
    norm_hidden_states = norm_hidden_states * norm3_weight + norm3_bias
    
    # First linear (produces 2x intermediate for GEGLU)
    ff_output = F.linear(norm_hidden_states, ff_linear1_weight, ff_linear1_bias)
    
    # GEGLU activation: split and apply gelu to gate
    x, gate = ff_output.chunk(2, dim=-1)
    ff_output = x * F.gelu(gate, approximate='tanh')
    
    # Second linear
    ff_output = F.linear(ff_output, ff_linear2_weight, ff_linear2_bias)
    
    # Residual connection
    output = hidden_states + ff_output
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.191580 ms |
| - | Scoring Baseline | 0.500000 | 0.985838 ms |
| - | Reference Implementation | 0.062388 | 12.568239 ms |
