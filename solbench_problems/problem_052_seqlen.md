## Description

Hyena gating mechanism and output projection that combines multiple orders of filtered outputs through cascaded gating with FFT convolution, followed by transpose and linear projection. For order=2: v = dropout(v * x[1]), v = fftconv(v, k[0], bias[0]), y = v * x[0], y = transpose(y), y = out_proj(y).
| Name | Shape | Dtype |
| --- | --- | --- |
| v | [batch_size, d_model, seq_len] | float32 |
| x0 | [batch_size, d_model, seq_len] | float32 |
| x1 | [batch_size, d_model, seq_len] | float32 |
| k | [order_minus_1, d_model, seq_len] | float32 |
| bias | [order_minus_1, d_model] | float32 |
| out_proj_weight | [d_model, d_model] | float32 |
| out_proj_bias | [d_model] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, seq_len, d_model] | float32 |

| # | d_model | hyena_order | order_minus_1 | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 2 | 1 | 1 | 1163 | 0.1486 | 0.0007 |
| 2 | 256 | 2 | 1 | 1 | 1721 | 0.1487 | 0.0009 |
| 3 | 256 | 2 | 1 | 16 | 2053 | 0.6454 | 0.0071 |
| 4 | 256 | 2 | 1 | 2 | 691 | 0.1384 | 0.0007 |
| 5 | 256 | 2 | 1 | 2 | 211 | 0.1365 | 0.0005 |
| 6 | 256 | 2 | 1 | 8 | 853 | 0.1764 | 0.0018 |
| 7 | 256 | 2 | 1 | 2 | 1801 | 0.1479 | 0.0013 |
| 8 | 256 | 2 | 1 | 16 | 449 | 0.1781 | 0.0019 |
| 9 | 256 | 2 | 1 | 4 | 1879 | 0.1810 | 0.0020 |
| 10 | 256 | 2 | 1 | 16 | 2048 | 0.4692 | 0.0071 |
| 11 | 256 | 2 | 1 | 1 | 613 | 0.1331 | 0.0006 |
| 12 | 256 | 2 | 1 | 16 | 1489 | 0.4096 | 0.0053 |
| 13 | 256 | 2 | 1 | 4 | 773 | 0.1609 | 0.0011 |
| 14 | 256 | 2 | 1 | 16 | 919 | 0.2641 | 0.0034 |
| 15 | 256 | 2 | 1 | 64 | 1087 | 1.1863 | 0.0144 |
| 16 | 256 | 2 | 1 | 4 | 293 | 0.1329 | 0.0007 |

```python
import torch

@torch.no_grad()
def run(
    v: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Hyena gating and output projection.
    
    Args:
        v: (batch, d_model, L) - filtered output
        x0: (batch, d_model, L) - first gating input
        x1: (batch, d_model, L) - second gating input (for order=2)
        k: (order-1, d_model, L) - filter coefficients
        bias: (order-1, d_model) - bias terms for FFT conv
        out_proj_weight: (d_model, d_model) - output projection weight
        out_proj_bias: (d_model,) - output projection bias
    
    Returns:
        (batch, L, d_model) - gated and projected output
    """
    seqlen = v.shape[-1]
    fft_size = 2 * seqlen
    
    # For order=2, we have one gating step with x1
    # Step 1: Element-wise gating multiplication (dropout=0.0, so no effect)
    v = v * x1
    
    # Step 2: FFT convolution with filter k[0] and bias[0]
    # fftconv_kernel implementation
    k_0 = k[0]  # (d_model, L)
    bias_0 = bias[0]  # (d_model,)
    
    # FFT of filter
    k_f = torch.fft.rfft(k_0.to(torch.float32), n=fft_size) / fft_size
    # FFT of input
    u_f = torch.fft.rfft(v.to(torch.float32), n=fft_size)
    
    # Element-wise multiplication in frequency domain and inverse FFT
    # k_f shape: (d_model, fft_size//2+1)
    # u_f shape: (batch, d_model, fft_size//2+1)
    # Need to broadcast k_f
    y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=fft_size, norm='forward')[..., :seqlen]
    
    # Add skip connection with bias
    v = y + v * bias_0.unsqueeze(-1)
    v = v.to(torch.float32)
    
    # Step 3: Final gating with x0
    y = v * x0
    
    # Step 4: Transpose from (batch, d_model, L) to (batch, L, d_model)
    y = y.transpose(1, 2)
    
    # Step 5: Output projection: y @ weight.T + bias
    y = torch.matmul(y, out_proj_weight.t()) + out_proj_bias
    
    return y
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001797 ms |
| - | Scoring Baseline | 0.500000 | 0.223532 ms |
| - | Reference Implementation | 0.414595 | 0.318415 ms |
