## Description

Backward pass for ConvNextV2 layer with NHWC persistence. Computes gradients through: residual addition, drop path, permutations, linear projections, GRN (Global Response Normalization), GELU activation, LayerNorm, and depthwise 7x7 convolution.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [B, C, H, W] | float32 |
| residual | [B, C, H, W] | float32 |
| x_dwconv | [B, C, H, W] | float32 |
| x_nhwc | [B, H, W, C] | float32 |
| mean | [B, H, W, 1] | float32 |
| var | [B, H, W, 1] | float32 |
| x_normalized | [B, H, W, C] | float32 |
| x_ln | [B, H, W, C] | float32 |
| x_expanded | [B, H, W, C4] | float32 |
| x_gelu | [B, H, W, C4] | float32 |
| global_features | [B, 1, 1, C4] | float32 |
| gf_mean | [B, 1, 1, 1] | float32 |
| norm_features | [B, 1, 1, C4] | float32 |
| x_grn_scaled | [B, H, W, C4] | float32 |
| x_grn | [B, H, W, C4] | float32 |
| dwconv_weight | [C, 1, 7, 7] | float32 |
| layernorm_weight | [C] | float32 |
| pwconv1_weight | [C4, C] | float32 |
| grn_weight | [1, 1, 1, C4] | float32 |
| pwconv2_weight | [C, C4] | float32 |
| drop_mask | [B, 1, 1, 1] | float32 |
| drop_path_prob | scalar | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_x | [B, C, H, W] | float32 |
| grad_dwconv_weight | [C, 1, 7, 7] | float32 |
| grad_dwconv_bias | [C] | float32 |
| grad_layernorm_weight | [C] | float32 |
| grad_layernorm_bias | [C] | float32 |
| grad_pwconv1_weight | [C4, C] | float32 |
| grad_pwconv1_bias | [C4] | float32 |
| grad_grn_weight | [1, 1, 1, C4] | float32 |
| grad_grn_bias | [1, 1, 1, C4] | float32 |
| grad_pwconv2_weight | [C, C4] | float32 |
| grad_pwconv2_bias | [C] | float32 |

| # | C | C4 | B | H | W | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | 512 | 16 | 14 | 14 | 0.3148 | 0.0028 |
| 2 | 128 | 512 | 8 | 28 | 28 | 0.3761 | 0.0051 |
| 3 | 128 | 512 | 8 | 14 | 14 | 0.3985 | 0.0016 |
| 4 | 128 | 512 | 1 | 14 | 14 | 0.3115 | 0.0006 |
| 5 | 128 | 512 | 4 | 56 | 56 | 0.5579 | 0.0097 |
| 6 | 128 | 512 | 32 | 28 | 28 | 0.8209 | 0.0189 |
| 7 | 128 | 512 | 1 | 56 | 56 | 0.2931 | 0.0028 |
| 8 | 128 | 512 | 8 | 56 | 56 | 0.7970 | 0.0189 |
| 9 | 128 | 512 | 1 | 28 | 28 | 0.3379 | 0.0011 |
| 10 | 128 | 512 | 64 | 14 | 14 | 0.8821 | 0.0097 |
| 11 | 128 | 512 | 4 | 28 | 28 | 0.4531 | 0.0028 |
| 12 | 128 | 512 | 32 | 56 | 56 | 2.4916 | 0.0743 |
| 13 | 128 | 512 | 4 | 56 | 56 | 1.3524 | 0.0374 |
| 14 | 128 | 512 | 16 | 56 | 56 | 0.3913 | 0.0016 |
| 15 | 128 | 512 | 2 | 28 | 28 | - | - |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    B = axes_and_scalars["B"]
    H = axes_and_scalars["H"]
    W = axes_and_scalars["W"]
    C = 128
    C4 = C * 4
    eps = 1e-6
    drop_path_prob = 0.1

    # Realistic weight initialization
    dwconv_weight = torch.randn(C, 1, 7, 7, device=device) * (1.0 / 49) ** 0.5
    layernorm_weight = torch.ones(C, device=device) + torch.randn(C, device=device) * 0.01
    pwconv1_weight = torch.randn(C4, C, device=device) * (2.0 / C) ** 0.5
    grn_weight = torch.zeros(1, 1, 1, C4, device=device) + torch.randn(1, 1, 1, C4, device=device) * 0.01
    pwconv2_weight = torch.randn(C, C4, device=device) * (2.0 / C4) ** 0.5

    # Input and grad_output at unit scale
    residual = torch.randn(B, C, H, W, device=device) * 0.1
    grad_output = torch.randn(B, C, H, W, device=device)

    # Drop mask
    drop_mask = (torch.rand(B, 1, 1, 1, device=device) > drop_path_prob).float()

    # --- Run forward pass to produce consistent intermediates ---
    with torch.no_grad():
        x_dwconv = F.conv2d(residual, dwconv_weight, padding=3, groups=C)
        x_nhwc = x_dwconv.permute(0, 2, 3, 1)

        mean = x_nhwc.mean(-1, keepdim=True)
        var = ((x_nhwc - mean) ** 2).mean(-1, keepdim=True)
        x_normalized = (x_nhwc - mean) / torch.sqrt(var + eps)
        x_ln = x_normalized * layernorm_weight

        x_expanded = x_ln @ pwconv1_weight.t()

        # GELU (tanh approximation)
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (x_expanded + 0.044715 * x_expanded.pow(3))
        x_gelu = 0.5 * x_expanded * (1.0 + torch.tanh(inner))

        # GRN
        global_features = torch.norm(x_gelu, p=2, dim=(1, 2), keepdim=True)
        gf_mean = global_features.mean(dim=-1, keepdim=True)
        norm_features = global_features / (gf_mean + eps)
        x_grn_scaled = x_gelu * norm_features
        x_grn = grn_weight * x_grn_scaled + x_gelu

    return {
        "grad_output": grad_output,
        "residual": residual,
        "x_dwconv": x_dwconv,
        "x_nhwc": x_nhwc,
        "mean": mean,
        "var": var,
        "x_normalized": x_normalized,
        "x_ln": x_ln,
        "x_expanded": x_expanded,
        "x_gelu": x_gelu,
        "global_features": global_features,
        "gf_mean": gf_mean,
        "norm_features": norm_features,
        "x_grn_scaled": x_grn_scaled,
        "x_grn": x_grn,
        "dwconv_weight": dwconv_weight,
        "layernorm_weight": layernorm_weight,
        "pwconv1_weight": pwconv1_weight,
        "grn_weight": grn_weight,
        "pwconv2_weight": pwconv2_weight,
        "drop_mask": drop_mask,
        "drop_path_prob": drop_path_prob,
        "eps": eps,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    x_dwconv: torch.Tensor,
    x_nhwc: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    x_normalized: torch.Tensor,
    x_ln: torch.Tensor,
    x_expanded: torch.Tensor,
    x_gelu: torch.Tensor,
    global_features: torch.Tensor,
    gf_mean: torch.Tensor,
    norm_features: torch.Tensor,
    x_grn_scaled: torch.Tensor,
    x_grn: torch.Tensor,
    dwconv_weight: torch.Tensor,
    layernorm_weight: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    grn_weight: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    drop_mask: torch.Tensor,
    drop_path_prob: float,
    eps: float,
):
    """
    Backward pass for ConvNextV2 layer with NHWC persistence.
    Computes gradients through the entire block in reverse order.
    """
    B = grad_output.shape[0]
    C = grad_output.shape[1]
    
    # Gradient through residual addition
    grad_residual = grad_output.clone()
    grad_x_nchw = grad_output.clone()
    
    # Gradient through drop path
    if drop_path_prob > 0.0:
        keep_prob = 1 - drop_path_prob
        grad_x_nchw = grad_x_nchw * drop_mask / keep_prob
    
    # Gradient through NHWC -> NCHW permutation
    # grad_x_nchw: (B, C, H, W) -> grad_x_projected: (B, H, W, C)
    grad_x_projected = grad_x_nchw.permute(0, 2, 3, 1)
    
    # Gradient through linear projection (pwconv2)
    # x_projected = pwconv2_weight @ x_grn + pwconv2_bias
    grad_x_grn = F.linear(grad_x_projected, pwconv2_weight.t())
    
    # grad_pwconv2_weight: (C, 4*C)
    grad_x_projected_flat = grad_x_projected.reshape(-1, grad_x_projected.shape[-1])
    x_grn_flat = x_grn.reshape(-1, x_grn.shape[-1])
    grad_pwconv2_weight = grad_x_projected_flat.t() @ x_grn_flat
    
    # grad_pwconv2_bias
    grad_pwconv2_bias = grad_x_projected.sum(dim=(0, 1, 2))
    
    # Gradient through GRN
    # x_grn = grn_weight * x_grn_scaled + grn_bias + x_gelu
    grad_x_gelu_from_grn = grad_x_grn.clone()  # From residual connection
    grad_x_grn_scaled = grad_x_grn * grn_weight
    
    # grad_grn_weight
    grad_grn_weight = (grad_x_grn * x_grn_scaled).sum(dim=(0, 1, 2), keepdim=True)
    
    # grad_grn_bias
    grad_grn_bias = grad_x_grn.sum(dim=(0, 1, 2), keepdim=True)
    
    # Gradient through x_grn_scaled = x_gelu * norm_features
    grad_x_gelu_from_scaled = grad_x_grn_scaled * norm_features
    grad_norm_features = (grad_x_grn_scaled * x_gelu).sum(dim=(1, 2), keepdim=True)
    
    # Combine gradients for x_gelu
    grad_x_gelu = grad_x_gelu_from_grn + grad_x_gelu_from_scaled
    
    # Gradient through norm_features = global_features / (gf_mean + eps)
    grad_global_features = grad_norm_features / (gf_mean + eps)
    grad_gf_mean = -grad_norm_features * global_features / ((gf_mean + eps) ** 2)
    
    # Gradient through gf_mean = global_features.mean(dim=-1, keepdim=True)
    C_expanded = global_features.shape[-1]
    grad_global_features = grad_global_features + grad_gf_mean / C_expanded
    
    # Gradient through global_features = ||x_gelu||_2 over spatial dims
    # d/dx ||x||_2 = x / ||x||_2
    grad_x_gelu = grad_x_gelu + x_gelu * grad_global_features / (global_features + eps)
    
    # Gradient through GELU
    # GELU(x) = x * Phi(x) where Phi is standard normal CDF
    # d/dx GELU(x) = Phi(x) + x * phi(x)
    sqrt_2_over_pi = 0.7978845608028654
    cdf_coeff = 0.044715
    inner = sqrt_2_over_pi * (x_expanded + cdf_coeff * x_expanded.pow(3))
    tanh_inner = torch.tanh(inner)
    cdf_approx = 0.5 * (1 + tanh_inner)
    pdf_approx = 0.5 * (1 - tanh_inner.pow(2)) * sqrt_2_over_pi * (1 + 3 * cdf_coeff * x_expanded.pow(2))
    gelu_grad = cdf_approx + x_expanded * pdf_approx
    grad_x_expanded = grad_x_gelu * gelu_grad
    
    # Gradient through linear expansion (pwconv1)
    grad_x_ln = F.linear(grad_x_expanded, pwconv1_weight.t())
    
    # grad_pwconv1_weight
    grad_x_expanded_flat = grad_x_expanded.reshape(-1, grad_x_expanded.shape[-1])
    x_ln_flat = x_ln.reshape(-1, x_ln.shape[-1])
    grad_pwconv1_weight = grad_x_expanded_flat.t() @ x_ln_flat
    
    # grad_pwconv1_bias
    grad_pwconv1_bias = grad_x_expanded.sum(dim=(0, 1, 2))
    
    # Gradient through LayerNorm
    # x_ln = x_normalized * layernorm_weight + layernorm_bias
    grad_x_normalized = grad_x_ln * layernorm_weight
    
    # grad_layernorm_weight
    grad_layernorm_weight = (grad_x_ln * x_normalized).sum(dim=(0, 1, 2))
    
    # grad_layernorm_bias
    grad_layernorm_bias = grad_x_ln.sum(dim=(0, 1, 2))
    
    # Gradient through normalization: x_normalized = (x_nhwc - mean) / sqrt(var + eps)
    std = torch.sqrt(var + eps)
    N = x_nhwc.shape[-1]  # Number of features
    
    grad_x_nhwc = grad_x_normalized / std
    grad_var = -(grad_x_normalized * (x_nhwc - mean)).sum(dim=-1, keepdim=True) / (2 * (var + eps) * std)
    grad_mean = -(grad_x_normalized / std).sum(dim=-1, keepdim=True)
    grad_mean = grad_mean + grad_var * (-2 * (x_nhwc - mean).sum(dim=-1, keepdim=True) / N)
    
    grad_x_nhwc = grad_x_nhwc + grad_var * (2 * (x_nhwc - mean) / N)
    grad_x_nhwc = grad_x_nhwc + grad_mean / N
    
    # Gradient through NCHW -> NHWC permutation
    # grad_x_nhwc: (B, H, W, C) -> grad_x_dwconv: (B, C, H, W)
    grad_x_dwconv = grad_x_nhwc.permute(0, 3, 1, 2)
    
    # Gradient through depthwise convolution
    # Input gradient using conv_transpose2d with groups=C for depthwise
    grad_x = F.conv_transpose2d(
        grad_x_dwconv,
        dwconv_weight,
        padding=3,
        groups=C
    )
    # Add gradient from residual connection
    grad_x = grad_x + grad_residual
    
    # Weight gradient for depthwise conv
    # For depthwise conv, each output channel g depends only on input channel g
    # grad_dwconv_weight[g, 0, :, :] = sum over batch of conv(residual[:, g], grad_x_dwconv[:, g])
    grad_dwconv_weight = torch.zeros_like(dwconv_weight)
    for g in range(C):
        # residual[:, g:g+1] has shape (B, 1, H, W)
        # grad_x_dwconv[:, g:g+1] has shape (B, 1, H, W)
        # We want to compute the correlation which gives us the weight gradient
        # Use conv2d with input as residual and weight as grad_x_dwconv (transposed)
        inp_channel = residual[:, g:g+1, :, :]  # (B, 1, H, W)
        grad_channel = grad_x_dwconv[:, g:g+1, :, :]  # (B, 1, H, W)
        # Reshape for conv: treat batch as output channels
        # grad_channel transposed: (1, B, H, W) as weight
        grad_channel_weight = grad_channel.permute(1, 0, 2, 3)  # (1, B, H, W)
        # inp_channel: (B, 1, H, W) -> (1, B, H, W) as input
        inp_channel_input = inp_channel.permute(1, 0, 2, 3)  # (1, B, H, W)
        # conv2d: input (1, B, H, W), weight (1, B, 7, 7) -> but we need different approach
        # Actually compute cross-correlation directly
        # grad_w[i,j] = sum_{b,h,w} residual[b,g,h+i,w+j] * grad[b,g,h,w]
        # This is F.conv2d(residual, grad) with appropriate padding
        # Simpler: use unfold and matmul
        H_in, W_in = residual.shape[2], residual.shape[3]
        # Pad residual for valid positions
        residual_padded = F.pad(residual[:, g:g+1, :, :], (3, 3, 3, 3))  # (B, 1, H+6, W+6)
        # Unfold to get patches
        patches = residual_padded.unfold(2, 7, 1).unfold(3, 7, 1)  # (B, 1, H, W, 7, 7)
        patches = patches.reshape(B, H_in, W_in, 49)  # (B, H, W, 49)
        grad_flat = grad_x_dwconv[:, g, :, :].reshape(B, H_in, W_in, 1)  # (B, H, W, 1)
        # Weight gradient: sum over B, H, W of patches * grad
        weight_grad = (patches * grad_flat).sum(dim=(0, 1, 2))  # (49,)
        grad_dwconv_weight[g, 0, :, :] = weight_grad.reshape(7, 7)
    
    # Bias gradient
    grad_dwconv_bias = grad_x_dwconv.sum(dim=(0, 2, 3))
    
    return (
        grad_x,
        grad_dwconv_weight,
        grad_dwconv_bias,
        grad_layernorm_weight,
        grad_layernorm_bias,
        grad_pwconv1_weight,
        grad_pwconv1_bias,
        grad_grn_weight,
        grad_grn_bias,
        grad_pwconv2_weight,
        grad_pwconv2_bias,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.005394 ms |
| - | Scoring Baseline | 0.500000 | 0.557452 ms |
| - | Reference Implementation | 0.065475 | 8.683990 ms |
