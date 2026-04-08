## Description

Fused 1D causal convolution with gated projection for Mamba layers. Performs gated linear projection (hidden_size -> 2*intermediate_size), splits into hidden states and gate, applies causal 1D depthwise convolution, SiLU activation, and optional attention masking.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| attention_mask | [batch_size, seq_len] | bfloat16 |
| in_proj_weight | [proj_out_size, hidden_size] | bfloat16 |
| in_proj_bias | [proj_out_size] | bfloat16 |
| conv1d_weight | [intermediate_size, 1, conv_kernel_size] | bfloat16 |
| conv1d_bias | [intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output_hidden_states | [batch_size, intermediate_size, seq_len] | bfloat16 |
| gate | [batch_size, intermediate_size, seq_len] | bfloat16 |

| # | hidden_size | intermediate_size | conv_kernel_size | proj_out_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8192 | 16384 | 4 | 32768 | 16 | 512 | 3.5450 | 2.4292 |
| 2 | 8192 | 16384 | 4 | 32768 | 8 | 128 | 0.5176 | 0.3040 |
| 3 | 8192 | 16384 | 4 | 32768 | 1 | 128 | 0.1873 | 0.0383 |
| 4 | 8192 | 16384 | 4 | 32768 | 1 | 2131 | 1.0862 | 0.6322 |
| 5 | 8192 | 16384 | 4 | 32768 | 4 | 373 | 0.7762 | 0.4427 |
| 6 | 8192 | 16384 | 4 | 32768 | 1 | 1024 | 0.5231 | 0.3040 |
| 7 | 8192 | 16384 | 4 | 32768 | 4 | 541 | 1.0935 | 0.6420 |
| 8 | 8192 | 16384 | 4 | 32768 | 32 | 256 | 3.5411 | 2.4292 |
| 9 | 8192 | 16384 | 4 | 32768 | 2 | 293 | 0.3963 | 0.1741 |
| 10 | 8192 | 16384 | 4 | 32768 | 1 | 8192 | 3.6555 | 2.4292 |
| 11 | 8192 | 16384 | 4 | 32768 | 2 | 1024 | 0.9347 | 0.6076 |
| 12 | 8192 | 16384 | 4 | 32768 | 2 | 128 | 0.2153 | 0.0763 |
| 13 | 8192 | 16384 | 4 | 32768 | 64 | 128 | 3.5519 | 2.4292 |
| 14 | 8192 | 16384 | 4 | 32768 | 8 | 1024 | 3.5301 | 2.4292 |
| 15 | 8192 | 16384 | 4 | 32768 | 2 | 256 | 0.3249 | 0.1522 |
| 16 | 8192 | 16384 | 4 | 32768 | 8 | 256 | 0.9369 | 0.6076 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
):
    """
    Fused Mamba conv1d with gating.
    
    Args:
        hidden_states: (batch_size, seq_len, 8192)
        attention_mask: (batch_size, seq_len)
        in_proj_weight: (32768, 8192)
        in_proj_bias: (32768,)
        conv1d_weight: (16384, 1, 4) - depthwise conv
        conv1d_bias: (16384,)
    
    Returns:
        output_hidden_states: (batch_size, 16384, seq_len)
        gate: (batch_size, 16384, seq_len)
    """
    batch_size, seq_len, _ = hidden_states.shape
    intermediate_size = 16384
    conv_kernel_size = 4
    
    # 1. Gated linear projection: (B, L, 8192) -> (B, L, 32768)
    projected_states = F.linear(hidden_states, in_proj_weight, in_proj_bias)
    
    # 2. Transpose for conv1d: (B, L, 32768) -> (B, 32768, L)
    projected_states = projected_states.transpose(1, 2)
    
    # 3. Split into hidden states and gate: (B, 32768, L) -> 2x (B, 16384, L)
    hidden_states_conv, gate = projected_states.chunk(2, dim=1)
    
    # 4. Apply attention mask before convolution
    hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
    
    # 5. Causal 1D convolution with grouped convolution (depthwise)
    # Pad on the left for causal convolution
    hidden_states_padded = F.pad(hidden_states_conv, (conv_kernel_size - 1, 0))
    
    # Depthwise conv1d: groups=intermediate_size
    hidden_states_conv = F.conv1d(
        hidden_states_padded,
        conv1d_weight,
        conv1d_bias,
        groups=intermediate_size
    )
    
    # 6. Apply SiLU activation: silu(x) = x * sigmoid(x)
    hidden_states_conv = hidden_states_conv * torch.sigmoid(hidden_states_conv)
    
    # 7. Apply attention mask after convolution
    hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
    
    return hidden_states_conv, gate
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.531170 ms |
| - | Scoring Baseline | 0.500000 | 0.973709 ms |
| - | Reference Implementation | 0.294007 | 1.535969 ms |
