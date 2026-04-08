## Description

Residual coupling block for normalizing flow with affine coupling transformations. Splits channels in half, transforms one half conditioned on the other, and combines. Supports forward and reverse passes for flow-based generative modeling.
| Name | Shape | Dtype |
| --- | --- | --- |
| x | [batch_size, channels, time] | float32 |
| x_mask | [batch_size, 1, time] | float32 |
| reverse | scalar | bool |
| transform_0_conv0_weight | [hidden_channels, half_channels, kernel_size] | float32 |
| transform_0_conv0_bias | [hidden_channels] | float32 |
| transform_0_conv1_weight | [hidden_channels, hidden_channels, kernel_size] | float32 |
| transform_0_conv1_bias | [hidden_channels] | float32 |
| transform_0_conv2_weight | [half_channels, hidden_channels, kernel_size] | float32 |
| transform_0_conv2_bias | [half_channels] | float32 |
| transform_1_conv0_weight | [hidden_channels, half_channels, kernel_size] | float32 |
| transform_1_conv0_bias | [hidden_channels] | float32 |
| transform_1_conv1_weight | [hidden_channels, hidden_channels, kernel_size] | float32 |
| transform_1_conv1_bias | [hidden_channels] | float32 |
| transform_1_conv2_weight | [half_channels, hidden_channels, kernel_size] | float32 |
| transform_1_conv2_bias | [half_channels] | float32 |
| transform_2_conv0_weight | [hidden_channels, half_channels, kernel_size] | float32 |
| transform_2_conv0_bias | [hidden_channels] | float32 |
| transform_2_conv1_weight | [hidden_channels, hidden_channels, kernel_size] | float32 |
| transform_2_conv1_bias | [hidden_channels] | float32 |
| transform_2_conv2_weight | [half_channels, hidden_channels, kernel_size] | float32 |
| transform_2_conv2_bias | [half_channels] | float32 |
| transform_3_conv0_weight | [hidden_channels, half_channels, kernel_size] | float32 |
| transform_3_conv0_bias | [hidden_channels] | float32 |
| transform_3_conv1_weight | [hidden_channels, hidden_channels, kernel_size] | float32 |
| transform_3_conv1_bias | [hidden_channels] | float32 |
| transform_3_conv2_weight | [half_channels, hidden_channels, kernel_size] | float32 |
| transform_3_conv2_bias | [half_channels] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, channels, time] | float32 |

| # | channels | hidden_channels | half_channels | kernel_size | n_layers | batch_size | time | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 192 | 192 | 96 | 5 | 4 | 2 | 293 | 0.4537 | 0.0014 |
| 2 | 192 | 192 | 96 | 5 | 4 | 8 | 768 | 0.5290 | 0.0104 |
| 3 | 192 | 192 | 96 | 5 | 4 | 16 | 2447 | 1.0085 | 0.0641 |
| 4 | 192 | 192 | 96 | 5 | 4 | 2 | 256 | 0.4695 | 0.0012 |
| 5 | 192 | 192 | 96 | 5 | 4 | 64 | 8192 | 8.5576 | 0.8541 |
| 6 | 192 | 192 | 96 | 5 | 4 | 64 | 3637 | 4.1897 | 0.3794 |
| 7 | 192 | 192 | 96 | 5 | 4 | 16 | 2048 | 0.9921 | 0.0538 |
| 8 | 192 | 192 | 96 | 5 | 4 | 1 | 128 | 0.4333 | 0.0006 |
| 9 | 192 | 192 | 96 | 5 | 4 | 4 | 1087 | 0.4889 | 0.0075 |
| 10 | 192 | 192 | 96 | 5 | 4 | 16 | 1536 | 0.7746 | 0.0404 |
| 11 | 192 | 192 | 96 | 5 | 4 | 2 | 691 | 0.4385 | 0.0027 |
| 12 | 192 | 192 | 96 | 5 | 4 | 4 | 384 | 0.5034 | 0.0029 |
| 13 | 192 | 192 | 96 | 5 | 4 | 8 | 1721 | 0.6133 | 0.0228 |
| 14 | 192 | 192 | 96 | 5 | 4 | 8 | 1024 | 0.5476 | 0.0137 |
| 15 | 192 | 192 | 96 | 5 | 4 | 4 | 2689 | 0.6294 | 0.0179 |
| 16 | 192 | 192 | 96 | 5 | 4 | 4 | 541 | 0.4831 | 0.0039 |

```python
import math

import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time = axes_and_scalars["time"]
    channels = 192
    hidden_channels = 192
    half_channels = 96
    kernel_size = 5

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv1d(out_c, in_c, k):
        fan_in = in_c * k
        return torch.randn(out_c, in_c, k, device=device, generator=g) * math.sqrt(2.0 / fan_in)

    inputs = {
        "x": torch.randn(batch_size, channels, time, device=device, generator=g),
        # Binary mask
        "x_mask": torch.ones(batch_size, 1, time, device=device),
        "reverse": False,
    }

    # 4 transforms x 3 convs each
    for i in range(4):
        # conv0: hidden_channels out, half_channels in
        inputs[f"transform_{i}_conv0_weight"] = kaiming_conv1d(hidden_channels, half_channels, kernel_size)
        inputs[f"transform_{i}_conv0_bias"] = torch.randn(hidden_channels, device=device, generator=g)
        # conv1: hidden_channels out, hidden_channels in
        inputs[f"transform_{i}_conv1_weight"] = kaiming_conv1d(hidden_channels, hidden_channels, kernel_size)
        inputs[f"transform_{i}_conv1_bias"] = torch.randn(hidden_channels, device=device, generator=g)
        # conv2: half_channels out, hidden_channels in
        inputs[f"transform_{i}_conv2_weight"] = kaiming_conv1d(half_channels, hidden_channels, kernel_size)
        inputs[f"transform_{i}_conv2_bias"] = torch.randn(half_channels, device=device, generator=g)

    return inputs

def apply_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
    """Apply a single transform: Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d"""
    # Conv1d with padding
    padding = conv0_w.shape[2] // 2
    h = F.conv1d(x0, conv0_w, conv0_b, padding=padding)
    h = F.relu(h)
    h = F.conv1d(h, conv1_w, conv1_b, padding=padding)
    h = F.relu(h)
    h = F.conv1d(h, conv2_w, conv2_b, padding=padding)
    return h

@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
):
    """
    Residual coupling flow block.
    
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    half_channels = x.shape[1] // 2
    
    # Collect all transform weights
    transforms = [
        (transform_0_conv0_weight, transform_0_conv0_bias,
         transform_0_conv1_weight, transform_0_conv1_bias,
         transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias,
         transform_1_conv1_weight, transform_1_conv1_bias,
         transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias,
         transform_2_conv1_weight, transform_2_conv1_bias,
         transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias,
         transform_3_conv1_weight, transform_3_conv1_bias,
         transform_3_conv2_weight, transform_3_conv2_bias),
    ]
    
    if not reverse:
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]
            
            # Compute transformation conditioned on x0
            h = apply_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
            
            # Apply mask
            h = h * x_mask
            
            # Affine coupling: x1 = x1 + h
            x1 = x1 + h
            
            # Concatenate back
            x = torch.cat([x0, x1], dim=1)
            
            # Apply mask to output
            x = x * x_mask
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]
            
            # Compute transformation conditioned on x0
            h = apply_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
            
            # Apply mask
            h = h * x_mask
            
            # Inverse affine coupling: x1 = x1 - h
            x1 = x1 - h
            
            # Concatenate back
            x = torch.cat([x0, x1], dim=1)
            
            # Apply mask to output
            x = x * x_mask
    
    return x
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.012924 ms |
| - | Scoring Baseline | 0.500000 | 0.768992 ms |
| - | Reference Implementation | 0.400013 | 1.141080 ms |
