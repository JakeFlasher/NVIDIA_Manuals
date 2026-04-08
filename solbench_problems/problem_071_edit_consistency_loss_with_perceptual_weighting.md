## Description

Edit consistency loss combining pixel reconstruction, perceptual features, and edit-region-aware spatial weighting for image editing models. Computes multi-scale perceptual features, applies spatial weighting based on edit masks, and combines pixel loss, perceptual loss, and preservation loss.
| Name | Shape | Dtype |
| --- | --- | --- |
| predicted | [batch_size, channels, height, width] | float32 |
| target | [batch_size, channels, height, width] | float32 |
| source | [batch_size, channels, height, width] | float32 |
| edit_mask | [batch_size, 1, height, width] | float32 |
| extractor0_conv1_weight | [feat_dim_0, channels, 3, 3] | float32 |
| extractor0_conv1_bias | [feat_dim_0] | float32 |
| extractor0_conv2_weight | [feat_dim_0, feat_dim_0, 3, 3] | float32 |
| extractor0_conv2_bias | [feat_dim_0] | float32 |
| extractor1_conv1_weight | [feat_dim_1, feat_dim_0, 3, 3] | float32 |
| extractor1_conv1_bias | [feat_dim_1] | float32 |
| extractor1_conv2_weight | [feat_dim_1, feat_dim_1, 3, 3] | float32 |
| extractor1_conv2_bias | [feat_dim_1] | float32 |
| extractor2_conv1_weight | [feat_dim_2, feat_dim_1, 3, 3] | float32 |
| extractor2_conv1_bias | [feat_dim_2] | float32 |
| extractor2_conv2_weight | [feat_dim_2, feat_dim_2, 3, 3] | float32 |
| extractor2_conv2_bias | [feat_dim_2] | float32 |
| scale_weights | [num_scales] | float32 |
| pixel_loss_weight | scalar | float32 |
| perceptual_loss_weight | scalar | float32 |
| edit_region_weight_multiplier | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| total_loss | [] | float32 |

| # | channels | feat_dim_0 | feat_dim_1 | feat_dim_2 | num_scales | batch_size | height | width | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 64 | 128 | 256 | 3 | 1 | 1024 | 1024 | 2.6631 | 0.3459 |
| 2 | 3 | 64 | 128 | 256 | 3 | 2 | 768 | 768 | 3.2030 | 0.3890 |
| 3 | 3 | 64 | 128 | 256 | 3 | 2 | 640 | 640 | 2.4649 | 0.2703 |
| 4 | 3 | 64 | 128 | 256 | 3 | 1 | 448 | 448 | 0.8894 | 0.0665 |
| 5 | 3 | 64 | 128 | 256 | 3 | 8 | 256 | 256 | 1.5268 | 0.1731 |
| 6 | 3 | 64 | 128 | 256 | 3 | 64 | 128 | 128 | 2.5093 | 0.3459 |
| 7 | 3 | 64 | 128 | 256 | 3 | 4 | 512 | 512 | 2.8182 | 0.3459 |
| 8 | 3 | 64 | 128 | 256 | 3 | 4 | 256 | 256 | 0.9444 | 0.0868 |
| 9 | 3 | 64 | 128 | 256 | 3 | 1 | 320 | 320 | 0.6194 | 0.0341 |
| 10 | 3 | 64 | 128 | 256 | 3 | 1 | 512 | 256 | 0.7633 | 0.0436 |
| 11 | 3 | 64 | 128 | 256 | 3 | 32 | 128 | 128 | 1.3971 | 0.1731 |
| 12 | 3 | 64 | 128 | 256 | 3 | 2 | 256 | 256 | 0.6291 | 0.0436 |
| 13 | 3 | 64 | 128 | 256 | 3 | 1 | 640 | 640 | 1.5323 | 0.1353 |
| 14 | 3 | 64 | 128 | 256 | 3 | 8 | 384 | 384 | 2.9639 | 0.3890 |
| 15 | 3 | 64 | 128 | 256 | 3 | 16 | 128 | 128 | 0.8561 | 0.0868 |
| 16 | 3 | 64 | 128 | 256 | 3 | 16 | 224 | 224 | 1.9972 | 0.2649 |

```python
import torch
import torch.nn.functional as F

def get_inputs(
    axes_and_scalars: dict, device: torch.device
) -> dict:
    """Generate inputs for the edit consistency loss."""
    batch_size = axes_and_scalars["batch_size"]
    height = axes_and_scalars["height"]
    width = axes_and_scalars["width"]
    channels = 3
    feat_dim_0 = 64
    feat_dim_1 = 128
    feat_dim_2 = 256
    num_scales = 3
    
    # Image tensors
    predicted = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    target = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    source = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)
    
    # Edit mask - binary values
    edit_mask = (torch.rand(batch_size, 1, height, width, device=device) > 0.5).float()
    
    # Extractor weights with proper initialization
    extractor0_conv1_weight = torch.randn(feat_dim_0, channels, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor0_conv1_bias = torch.zeros(feat_dim_0, dtype=torch.float32, device=device)
    extractor0_conv2_weight = torch.randn(feat_dim_0, feat_dim_0, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor0_conv2_bias = torch.zeros(feat_dim_0, dtype=torch.float32, device=device)
    
    extractor1_conv1_weight = torch.randn(feat_dim_1, feat_dim_0, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor1_conv1_bias = torch.zeros(feat_dim_1, dtype=torch.float32, device=device)
    extractor1_conv2_weight = torch.randn(feat_dim_1, feat_dim_1, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor1_conv2_bias = torch.zeros(feat_dim_1, dtype=torch.float32, device=device)
    
    extractor2_conv1_weight = torch.randn(feat_dim_2, feat_dim_1, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor2_conv1_bias = torch.zeros(feat_dim_2, dtype=torch.float32, device=device)
    extractor2_conv2_weight = torch.randn(feat_dim_2, feat_dim_2, 3, 3, dtype=torch.float32, device=device) * 0.1
    extractor2_conv2_bias = torch.zeros(feat_dim_2, dtype=torch.float32, device=device)
    
    # Scale weights
    scale_weights = torch.ones(num_scales, dtype=torch.float32, device=device) / num_scales
    
    # Scalar values
    pixel_loss_weight = 1.0
    perceptual_loss_weight = 0.5
    edit_region_weight_multiplier = 2.0
    
    return {
        "predicted": predicted,
        "target": target,
        "source": source,
        "edit_mask": edit_mask,
        "extractor0_conv1_weight": extractor0_conv1_weight,
        "extractor0_conv1_bias": extractor0_conv1_bias,
        "extractor0_conv2_weight": extractor0_conv2_weight,
        "extractor0_conv2_bias": extractor0_conv2_bias,
        "extractor1_conv1_weight": extractor1_conv1_weight,
        "extractor1_conv1_bias": extractor1_conv1_bias,
        "extractor1_conv2_weight": extractor1_conv2_weight,
        "extractor1_conv2_bias": extractor1_conv2_bias,
        "extractor2_conv1_weight": extractor2_conv1_weight,
        "extractor2_conv1_bias": extractor2_conv1_bias,
        "extractor2_conv2_weight": extractor2_conv2_weight,
        "extractor2_conv2_bias": extractor2_conv2_bias,
        "scale_weights": scale_weights,
        "pixel_loss_weight": pixel_loss_weight,
        "perceptual_loss_weight": perceptual_loss_weight,
        "edit_region_weight_multiplier": edit_region_weight_multiplier,
    }

@torch.no_grad()
def run(
    predicted: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    edit_mask: torch.Tensor,
    extractor0_conv1_weight: torch.Tensor,
    extractor0_conv1_bias: torch.Tensor,
    extractor0_conv2_weight: torch.Tensor,
    extractor0_conv2_bias: torch.Tensor,
    extractor1_conv1_weight: torch.Tensor,
    extractor1_conv1_bias: torch.Tensor,
    extractor1_conv2_weight: torch.Tensor,
    extractor1_conv2_bias: torch.Tensor,
    extractor2_conv1_weight: torch.Tensor,
    extractor2_conv1_bias: torch.Tensor,
    extractor2_conv2_weight: torch.Tensor,
    extractor2_conv2_bias: torch.Tensor,
    scale_weights: torch.Tensor,
    pixel_loss_weight: float,
    perceptual_loss_weight: float,
    edit_region_weight_multiplier: float,
):
    # Sobel kernels for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=edit_mask.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=edit_mask.device).view(1, 1, 3, 3)
    
    # Compute edge map from edit mask
    edges_x = F.conv2d(edit_mask, sobel_x, padding=1)
    edges_y = F.conv2d(edit_mask, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
    
    # Compute spatial weights
    spatial_weights = torch.ones_like(edit_mask)
    spatial_weights = spatial_weights + (edit_region_weight_multiplier - 1.0) * edit_mask
    spatial_weights = spatial_weights + 0.5 * edge_magnitude
    
    # 1. Pixel-level reconstruction loss (L1)
    pixel_diff = torch.abs(predicted - target)
    weighted_pixel_loss = pixel_diff * spatial_weights
    pixel_loss = weighted_pixel_loss.mean()
    
    # 2. Extract perceptual features for predicted and target
    def extract_features(x, conv1_w, conv1_b, conv2_w, conv2_b):
        x = F.conv2d(x, conv1_w, conv1_b, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, conv2_w, conv2_b, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x
    
    # Scale 0
    pred_feat0 = extract_features(predicted, extractor0_conv1_weight, extractor0_conv1_bias, extractor0_conv2_weight, extractor0_conv2_bias)
    target_feat0 = extract_features(target, extractor0_conv1_weight, extractor0_conv1_bias, extractor0_conv2_weight, extractor0_conv2_bias)
    
    # Scale 1
    pred_feat1 = extract_features(pred_feat0, extractor1_conv1_weight, extractor1_conv1_bias, extractor1_conv2_weight, extractor1_conv2_bias)
    target_feat1 = extract_features(target_feat0, extractor1_conv1_weight, extractor1_conv1_bias, extractor1_conv2_weight, extractor1_conv2_bias)
    
    # Scale 2
    pred_feat2 = extract_features(pred_feat1, extractor2_conv1_weight, extractor2_conv1_bias, extractor2_conv2_weight, extractor2_conv2_bias)
    target_feat2 = extract_features(target_feat1, extractor2_conv1_weight, extractor2_conv1_bias, extractor2_conv2_weight, extractor2_conv2_bias)
    
    pred_features = [pred_feat0, pred_feat1, pred_feat2]
    target_features = [target_feat0, target_feat1, target_feat2]
    
    # Compute perceptual loss across scales
    perceptual_loss = torch.tensor(0.0, dtype=torch.float32, device=predicted.device)
    
    for scale_idx in range(3):
        pred_feat = pred_features[scale_idx]
        target_feat = target_features[scale_idx]
        
        # Downsample spatial weights to match feature resolution
        downsampled_weights = F.interpolate(
            spatial_weights,
            size=(pred_feat.shape[2], pred_feat.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # Compute weighted perceptual difference
        feat_diff = torch.abs(pred_feat - target_feat)
        weighted_feat_diff = feat_diff * downsampled_weights
        scale_loss = weighted_feat_diff.mean()
        
        # Apply learnable scale weight
        scale_loss = scale_loss * torch.abs(scale_weights[scale_idx])
        perceptual_loss = perceptual_loss + scale_loss
    
    # 3. Preservation loss for non-edited regions
    inverse_mask = 1.0 - edit_mask
    preservation_diff = torch.abs(predicted - source) * inverse_mask
    preservation_loss = preservation_diff.mean()
    
    # Combine all losses
    total_loss = (
        pixel_loss_weight * pixel_loss +
        perceptual_loss_weight * perceptual_loss +
        0.3 * preservation_loss
    )
    
    return total_loss
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.149004 ms |
| - | Scoring Baseline | 0.500000 | 1.493878 ms |
| - | Reference Implementation | 0.260288 | 3.973370 ms |
