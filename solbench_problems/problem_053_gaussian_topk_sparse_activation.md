## Description

Gaussian-based top-k sparse activation mechanism. Computes adaptive sparsity by calculating a threshold based on input's mean and standard deviation, scaled by the inverse CDF of a normal distribution at the target sparsity level. Applies ReLU with this dynamic cutoff to create learned sparse activations.
| Name | Shape | Dtype |
| --- | --- | --- |
| inputs | [batch_size, seq_len, intermediate_size] | bfloat16 |
| target_sparsity | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, intermediate_size] | bfloat16 |

| # | batch_size | seq_len | intermediate_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 512 | 12288 | 0.0589 | 0.0037 |
| 2 | 32 | 256 | 16384 | 0.0974 | 0.0004 |
| 3 | 4 | 613 | 12288 | 0.0289 | 0.0004 |
| 4 | 4 | 2048 | 12288 | 0.1092 | 0.0529 |
| 5 | 2 | 373 | 16384 | 0.0142 | 0.0004 |
| 6 | 32 | 128 | 12288 | 0.0780 | 0.0267 |
| 7 | 1 | 128 | 4096 | 0.0121 | 0.0004 |
| 8 | 2 | 211 | 8192 | 0.0574 | 0.0022 |
| 9 | 1 | 8192 | 4096 | 0.0656 | 0.0179 |
| 10 | 1 | 1024 | 16384 | 0.0578 | 0.0092 |
| 11 | 16 | 1163 | 8192 | 0.1478 | 0.0799 |
| 12 | 4 | 541 | 8192 | 0.0583 | 0.0096 |
| 13 | 4 | 449 | 4096 | 0.0578 | 0.0042 |
| 14 | 64 | 1024 | 8192 | 0.3526 | 0.2805 |
| 15 | 2 | 131 | 4096 | 0.0581 | 0.0010 |
| 16 | 2 | 293 | 12288 | 0.0581 | 0.0042 |

```python
import torch
import torch.nn.functional as F
import math

def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    This is a rational approximation that works well for p in (0, 1).
    """
    # Constants for the approximation
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    
    p_low = 0.02425
    p_high = 1.0 - p_low
    
    result = torch.zeros_like(p)
    
    # Lower region
    mask_low = p < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
        result[mask_low] = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                           ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    
    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid.any():
        q = p[mask_mid] - 0.5
        r = q * q
        result[mask_mid] = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
                           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    
    # Upper region
    mask_high = p > p_high
    if mask_high.any():
        q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
        result[mask_high] = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    
    return result

@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    
    Computes adaptive sparsity threshold based on input statistics:
    1. Compute mean and std of input across feature dimension
    2. Calculate threshold = mean + std * norm.icdf(target_sparsity)
    3. Apply ReLU(input - threshold) to create sparse activations
    
    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level.
                        0.0 means no sparsity (all activations pass through).
    
    Returns:
        Sparsified tensor of same shape as input.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs
    
    # Convert to float32 for numerical stability in statistics computation
    inputs_f32 = inputs.to(torch.float32)
    
    # Compute statistics along the feature dimension (last dim)
    # Shape: [batch_size, seq_len, 1]
    inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
    
    # Compute the standard deviation multiplier using inverse CDF
    # This maps the target sparsity to a z-score in the normal distribution
    target_sparsity_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
    std_multiplier = _ndtri(target_sparsity_tensor)
    
    # Compute the adaptive cutoff threshold
    # Values below this threshold will be zeroed out
    # Shape: [batch_size, seq_len, 1]
    cutoff_threshold = inputs_mean + inputs_std * std_multiplier
    
    # Apply ReLU with the adaptive threshold
    # This is equivalent to: max(0, inputs - cutoff_threshold)
    sparse_output = F.relu(inputs_f32 - cutoff_threshold)
    
    return sparse_output.to(torch.bfloat16)
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.005059 ms |
| - | Scoring Baseline | 0.500000 | 0.060403 ms |
| - | Reference Implementation | 0.253256 | 0.233042 ms |
