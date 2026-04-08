## Description

Complete input processing pipeline for SD3.5 that converts spatial image latents through patch embedding with positional encoding, processes timestep and pooled projections into temporal conditioning, and projects context embeddings. Combines three embedding pathways: image patches, temporal conditioning, and text context.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, in_channels, sample_size, sample_size] | float32 |
| encoder_hidden_states | [batch_size, seq_len_context, joint_attention_dim] | float32 |
| pooled_projections | [batch_size, pooled_projection_dim] | float32 |
| timestep | [batch_size] | float32 |
| proj_weight | [inner_dim, in_channels, patch_size, patch_size] | float32 |
| proj_bias | [inner_dim] | float32 |
| pos_embed | [1, num_patches, inner_dim] | float32 |
| timestep_linear1_weight | [inner_dim, sinusoidal_dim] | float32 |
| timestep_linear1_bias | [inner_dim] | float32 |
| timestep_linear2_weight | [inner_dim, inner_dim] | float32 |
| timestep_linear2_bias | [inner_dim] | float32 |
| pooled_linear1_weight | [inner_dim, pooled_projection_dim] | float32 |
| pooled_linear1_bias | [inner_dim] | float32 |
| pooled_linear2_weight | [inner_dim, inner_dim] | float32 |
| pooled_linear2_bias | [inner_dim] | float32 |
| context_embedder_weight | [caption_projection_dim, joint_attention_dim] | float32 |
| context_embedder_bias | [caption_projection_dim] | float32 |
| freqs | [sinusoidal_half_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output_hidden_states | [batch_size, num_patches, inner_dim] | float32 |
| temb | [batch_size, inner_dim] | float32 |
| output_encoder_hidden_states | [batch_size, seq_len_context, caption_projection_dim] | float32 |

| # | sample_size | patch_size | in_channels | num_attention_heads | attention_head_dim | inner_dim | joint_attention_dim | caption_projection_dim | pooled_projection_dim | pos_embed_max_size | sinusoidal_dim | sinusoidal_half_dim | num_patches | batch_size | seq_len_context | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 32 | 77 | 1.0192 | 0.0927 |
| 2 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 2 | 211 | 0.2505 | 0.0091 |
| 3 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 64 | 1489 | 4.1609 | 1.0949 |
| 4 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 8 | 1249 | 0.6830 | 0.1161 |
| 5 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 4 | 1163 | 0.4527 | 0.0545 |
| 6 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 1 | 77 | 0.1812 | 0.0058 |
| 7 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 1 | 131 | 0.1828 | 0.0059 |
| 8 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 2 | 1087 | 0.3072 | 0.0258 |
| 9 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 16 | 449 | 0.7735 | 0.0910 |
| 10 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 8 | 77 | 0.4199 | 0.0255 |
| 11 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 4 | 1024 | 0.4338 | 0.0483 |
| 12 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 1 | 691 | 0.1958 | 0.0087 |
| 13 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 8 | 373 | 0.4914 | 0.0390 |
| 14 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 32 | 2048 | 2.6080 | 0.7444 |
| 15 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 32 | 541 | 1.4120 | 0.2140 |
| 16 | 128 | 2 | 16 | 38 | 64 | 2432 | 4096 | 2432 | 2048 | 192 | 256 | 128 | 4096 | 2 | 773 | 0.2808 | 0.0189 |

```python
import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    pos_embed: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    pooled_linear1_weight: torch.Tensor,
    pooled_linear1_bias: torch.Tensor,
    pooled_linear2_weight: torch.Tensor,
    pooled_linear2_bias: torch.Tensor,
    context_embedder_weight: torch.Tensor,
    context_embedder_bias: torch.Tensor,
    freqs: torch.Tensor,
):
    batch_size = hidden_states.shape[0]
    
    # 1. Patch Embedding: Conv2d for patch extraction
    # (B, C, H, W) -> (B, inner_dim, H//patch_size, W//patch_size)
    patch_embedded = F.conv2d(hidden_states, proj_weight, proj_bias, stride=2)
    
    # Flatten to sequence: (B, inner_dim, h, w) -> (B, h*w, inner_dim)
    patch_embedded = patch_embedded.flatten(2).transpose(1, 2)
    
    # 2. Add Positional Embeddings
    num_patches = patch_embedded.shape[1]
    pos_embed_slice = pos_embed[:, :num_patches, :]
    output_hidden_states = patch_embedded + pos_embed_slice
    
    # 3. Timestep Embedding: sinusoidal encoding
    # timesteps: (B,) -> (B, 1)
    timestep_expanded = timestep.unsqueeze(-1)
    
    # Compute sinusoidal features: (B, 128)
    args = timestep_expanded * freqs.unsqueeze(0)
    timestep_sinusoidal = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 256)
    
    # Timestep MLP: Linear -> SiLU -> Linear
    timestep_embed = F.linear(timestep_sinusoidal, timestep_linear1_weight, timestep_linear1_bias)
    timestep_embed = F.silu(timestep_embed)
    timestep_embed = F.linear(timestep_embed, timestep_linear2_weight, timestep_linear2_bias)
    
    # 4. Pooled Projection Embedding: Linear -> SiLU -> Linear
    pooled_embed = F.linear(pooled_projections, pooled_linear1_weight, pooled_linear1_bias)
    pooled_embed = F.silu(pooled_embed)
    pooled_embed = F.linear(pooled_embed, pooled_linear2_weight, pooled_linear2_bias)
    
    # 5. Combine temporal embeddings
    temb = timestep_embed + pooled_embed
    
    # 6. Context Embedder: Linear projection
    output_encoder_hidden_states = F.linear(encoder_hidden_states, context_embedder_weight, context_embedder_bias)
    
    return output_hidden_states, temb, output_encoder_hidden_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.047121 ms |
| - | Scoring Baseline | 0.500000 | 0.536331 ms |
| - | Reference Implementation | 0.314261 | 1.120547 ms |
