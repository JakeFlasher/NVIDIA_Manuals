## Description

VAE decoder mid-block combining ResNet blocks with self-attention at the bottleneck. Processes compressed latent representation through: ResNet -> Attention -> ResNet. The attention operates on spatial features with GroupNorm, QKV projections, multi-head self-attention, and residual connections. Uses 512 channels at the bottleneck with 4 groups for GroupNorm.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, in_channels, height, width] | float32 |
| temb | [batch_size, temb_channels] | float32 |
| resnet1_norm1_weight | [in_channels] | float32 |
| resnet1_norm1_bias | [in_channels] | float32 |
| resnet1_conv1_weight | [in_channels, in_channels, 3, 3] | float32 |
| resnet1_conv1_bias | [in_channels] | float32 |
| resnet1_time_emb_proj_weight | [in_channels, temb_channels] | float32 |
| resnet1_time_emb_proj_bias | [in_channels] | float32 |
| resnet1_norm2_weight | [in_channels] | float32 |
| resnet1_norm2_bias | [in_channels] | float32 |
| resnet1_conv2_weight | [in_channels, in_channels, 3, 3] | float32 |
| resnet1_conv2_bias | [in_channels] | float32 |
| attn_group_norm_weight | [in_channels] | float32 |
| attn_group_norm_bias | [in_channels] | float32 |
| attn_to_q_weight | [in_channels, in_channels] | float32 |
| attn_to_q_bias | [in_channels] | float32 |
| attn_to_k_weight | [in_channels, in_channels] | float32 |
| attn_to_k_bias | [in_channels] | float32 |
| attn_to_v_weight | [in_channels, in_channels] | float32 |
| attn_to_v_bias | [in_channels] | float32 |
| attn_to_out_weight | [in_channels, in_channels] | float32 |
| attn_to_out_bias | [in_channels] | float32 |
| resnet2_norm1_weight | [in_channels] | float32 |
| resnet2_norm1_bias | [in_channels] | float32 |
| resnet2_conv1_weight | [in_channels, in_channels, 3, 3] | float32 |
| resnet2_conv1_bias | [in_channels] | float32 |
| resnet2_time_emb_proj_weight | [in_channels, temb_channels] | float32 |
| resnet2_time_emb_proj_bias | [in_channels] | float32 |
| resnet2_norm2_weight | [in_channels] | float32 |
| resnet2_norm2_bias | [in_channels] | float32 |
| resnet2_conv2_weight | [in_channels, in_channels, 3, 3] | float32 |
| resnet2_conv2_bias | [in_channels] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, in_channels, height, width] | float32 |

| # | in_channels | temb_channels | num_groups | attention_head_dim | num_heads | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 512 | 32 | 1 | 512 | 1 | 32 | 32 | 0.2821 | 0.0134 |
| 2 | 512 | 512 | 32 | 1 | 512 | 1 | 61 | 61 | 0.6831 | 0.0591 |
| 3 | 512 | 512 | 32 | 1 | 512 | 2 | 64 | 64 | 0.9346 | 0.1332 |
| 4 | 512 | 512 | 32 | 1 | 512 | 1 | 48 | 48 | 0.5200 | 0.0331 |
| 5 | 512 | 512 | 32 | 1 | 512 | 16 | 32 | 32 | 1.3223 | 0.2091 |
| 6 | 512 | 512 | 32 | 1 | 512 | 4 | 16 | 16 | 0.4258 | 0.0126 |
| 7 | 512 | 512 | 32 | 1 | 512 | 8 | 32 | 32 | 0.8208 | 0.1047 |
| 8 | 512 | 512 | 32 | 1 | 512 | 32 | 32 | 32 | 2.3514 | 0.4178 |
| 9 | 512 | 512 | 32 | 1 | 512 | 4 | 48 | 48 | 0.9459 | 0.1311 |
| 10 | 512 | 512 | 32 | 1 | 512 | 2 | 41 | 41 | 0.7365 | 0.0457 |
| 11 | 512 | 512 | 32 | 1 | 512 | 1 | 16 | 16 | 0.3658 | 0.0034 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    resnet1_norm1_weight: torch.Tensor,
    resnet1_norm1_bias: torch.Tensor,
    resnet1_conv1_weight: torch.Tensor,
    resnet1_conv1_bias: torch.Tensor,
    resnet1_time_emb_proj_weight: torch.Tensor,
    resnet1_time_emb_proj_bias: torch.Tensor,
    resnet1_norm2_weight: torch.Tensor,
    resnet1_norm2_bias: torch.Tensor,
    resnet1_conv2_weight: torch.Tensor,
    resnet1_conv2_bias: torch.Tensor,
    attn_group_norm_weight: torch.Tensor,
    attn_group_norm_bias: torch.Tensor,
    attn_to_q_weight: torch.Tensor,
    attn_to_q_bias: torch.Tensor,
    attn_to_k_weight: torch.Tensor,
    attn_to_k_bias: torch.Tensor,
    attn_to_v_weight: torch.Tensor,
    attn_to_v_bias: torch.Tensor,
    attn_to_out_weight: torch.Tensor,
    attn_to_out_bias: torch.Tensor,
    resnet2_norm1_weight: torch.Tensor,
    resnet2_norm1_bias: torch.Tensor,
    resnet2_conv1_weight: torch.Tensor,
    resnet2_conv1_bias: torch.Tensor,
    resnet2_time_emb_proj_weight: torch.Tensor,
    resnet2_time_emb_proj_bias: torch.Tensor,
    resnet2_norm2_weight: torch.Tensor,
    resnet2_norm2_bias: torch.Tensor,
    resnet2_conv2_weight: torch.Tensor,
    resnet2_conv2_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1  # Single-head attention at VAE bottleneck
    head_dim = channels  # head_dim equals channels when num_heads=1
    scale = head_dim ** -0.5
    
    # ============ ResNet Block 1 ============
    residual1 = hidden_states
    
    # GroupNorm1 + SiLU + Conv1
    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)
    
    # Add time embedding
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    
    # GroupNorm2 + SiLU + Conv2
    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)
    
    # Residual connection
    hidden_states = h + residual1
    
    # ============ Attention Block ============
    attn_residual = hidden_states
    
    # GroupNorm
    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    
    # Reshape to [B, H*W, C]
    h = h.view(batch, channels, height * width).transpose(1, 2)
    
    # QKV projections
    query = F.linear(h, attn_to_q_weight, attn_to_q_bias)
    key = F.linear(h, attn_to_k_weight, attn_to_k_bias)
    value = F.linear(h, attn_to_v_weight, attn_to_v_bias)
    
    # Reshape for multi-head attention [B, num_heads, H*W, head_dim]
    seq_len = height * width
    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention scores [B, num_heads, H*W, H*W]
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    
    # Apply attention to values [B, num_heads, H*W, head_dim]
    h = torch.matmul(attention_probs, value)
    
    # Reshape back [B, H*W, C]
    h = h.transpose(1, 2).reshape(batch, seq_len, channels)
    
    # Output projection
    h = F.linear(h, attn_to_out_weight, attn_to_out_bias)
    
    # Reshape to [B, C, H, W]
    h = h.transpose(1, 2).view(batch, channels, height, width)
    
    # Residual connection
    hidden_states = h + attn_residual
    
    # ============ ResNet Block 2 ============
    residual2 = hidden_states
    
    # GroupNorm1 + SiLU + Conv1
    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)
    
    # Add time embedding
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    
    # GroupNorm2 + SiLU + Conv2
    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)
    
    # Residual connection
    output = h + residual2
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.052411 ms |
| - | Scoring Baseline | 0.500000 | 0.717484 ms |
| - | Reference Implementation | 0.335365 | 1.375200 ms |
