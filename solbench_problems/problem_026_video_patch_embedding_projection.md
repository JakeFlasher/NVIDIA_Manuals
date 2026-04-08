## Description

Video patch embedding projection that converts raw video frames into patch tokens. Applies Conv3D patch projection, adds spatial and temporal positional embeddings, and applies layer normalization. Input video shape is (batch, frames, channels, height, width), output is (batch, num_patches, hidden_size) where num_patches = frames * (height/patch_size) * (width/patch_size).
| Name | Shape | Dtype |
| --- | --- | --- |
| video | [batch_size, num_frames, in_channels, height, width] | float16 |
| patch_projection_weight | [hidden_size, in_channels, 1, patch_size, patch_size] | float16 |
| patch_projection_bias | [hidden_size] | float16 |
| spatial_pos_embedding | [1, num_spatial_patches, hidden_size] | float16 |
| temporal_pos_embedding | [1, num_frames, 1, hidden_size] | float16 |
| norm_weight | [hidden_size] | float16 |
| norm_bias | [hidden_size] | float16 |
| eps | scalar | float16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, total_patches, hidden_size] | float16 |

| # | num_frames | in_channels | hidden_size | patch_size | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 81 | 16 | 5120 | 2 | 1 | 224 | 224 | 12.3115 | 1.3911 |
| 2 | 81 | 16 | 5120 | 2 | 1 | 192 | 192 | 9.0827 | 1.0221 |
| 3 | 81 | 16 | 5120 | 2 | 1 | 128 | 128 | 3.7296 | 0.4545 |
| 4 | 81 | 16 | 5120 | 2 | 2 | 64 | 64 | 2.0917 | 0.2261 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    video: torch.Tensor,
    patch_projection_weight: torch.Tensor,
    patch_projection_bias: torch.Tensor,
    spatial_pos_embedding: torch.Tensor,
    temporal_pos_embedding: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
):
    """
    Video patch embedding projection.
    
    Args:
        video: (batch, frames, channels, height, width)
        patch_projection_weight: (hidden_size, in_channels, 1, patch_size, patch_size)
        patch_projection_bias: (hidden_size,)
        spatial_pos_embedding: (1, num_spatial_patches, hidden_size)
        temporal_pos_embedding: (1, num_frames, 1, hidden_size)
        norm_weight: (hidden_size,)
        norm_bias: (hidden_size,)
        eps: layer norm epsilon
    
    Returns:
        output: (batch, total_patches, hidden_size)
    """
    batch_size, frames, channels, height, width = video.shape
    hidden_size = patch_projection_weight.shape[0]
    patch_size = patch_projection_weight.shape[3]
    
    # Rearrange to (batch, channels, frames, height, width) for Conv3D
    video = video.permute(0, 2, 1, 3, 4)
    
    # Apply patch projection using Conv3D
    # (B, C, F, H, W) -> (B, hidden_size, F, H', W')
    patches = F.conv3d(
        video,
        patch_projection_weight,
        patch_projection_bias,
        stride=(1, patch_size, patch_size),
        padding=(0, 0, 0)
    )
    
    # Rearrange to (batch, frames, hidden_size, num_patches_h, num_patches_w)
    patches = patches.permute(0, 2, 1, 3, 4)
    
    # Get spatial dimensions
    batch_size, frames, hidden_size, num_h, num_w = patches.shape
    
    # Flatten spatial dimensions: (B, F, hidden_size, H', W') -> (B, F, H'*W', hidden_size)
    patches = patches.reshape(batch_size, frames, hidden_size, num_h * num_w)
    patches = patches.permute(0, 1, 3, 2)  # (B, F, num_spatial_patches, hidden_size)
    
    # Add spatial positional embeddings (broadcast across frames)
    # spatial_pos_embedding: (1, num_spatial_patches, hidden_size) -> broadcast to (B, F, S, H)
    patches = patches + spatial_pos_embedding.unsqueeze(1)
    
    # Add temporal positional embeddings (broadcast across spatial patches)
    # temporal_pos_embedding: (1, num_frames, 1, hidden_size) -> broadcast to (B, F, S, H)
    patches = patches + temporal_pos_embedding
    
    # Flatten temporal and spatial dimensions: (B, F, S, H) -> (B, F*S, H)
    patches = patches.reshape(batch_size, frames * num_h * num_w, hidden_size)
    
    # Apply layer normalization
    # Compute mean and variance along last dimension
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True, unbiased=False)
    patches_normalized = (patches - mean) / torch.sqrt(var + eps)
    output = patches_normalized * norm_weight + norm_bias
    
    return output
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.618261 ms |
| - | Scoring Baseline | 0.500000 | 5.434677 ms |
| - | Reference Implementation | 0.056199 | 86.497237 ms |
