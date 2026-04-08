## Description

Backward pass for fused embedding lookup and RMSNorm. Computes gradients for embedding table (scatter-add) and RMSNorm weight (reduction). The forward pass performs: embed_lookup -> RMSNorm(x * rsqrt(mean(x^2) + eps) * weight).
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| input_ids | [batch_size, seq_len] | int64 |
| hidden_states_fp32 | [batch_size, seq_len, hidden_size] | float32 |
| rstd | [batch_size, seq_len, 1] | float32 |
| norm_weight | [hidden_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_embed_weight | [vocab_size, hidden_size] | bfloat16 |
| grad_norm_weight | [hidden_size] | bfloat16 |

| # | vocab_size | hidden_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 65536 | 4096 | 64 | 256 | 0.7319 | 0.0529 |
| 2 | 65536 | 4096 | 2 | 512 | 0.1525 | 0.0037 |
| 3 | 65536 | 4096 | 32 | 1024 | 1.3751 | 0.1054 |
| 4 | 65536 | 4096 | 32 | 373 | 0.5623 | 0.0387 |
| 5 | 65536 | 4096 | 2 | 2048 | 0.2554 | 0.0135 |
| 6 | 65536 | 4096 | 1 | 1024 | 0.1142 | 0.0037 |
| 7 | 65536 | 4096 | 16 | 293 | 0.2667 | 0.0154 |
| 8 | 65536 | 4096 | 4 | 541 | 0.1697 | 0.0073 |
| 9 | 65536 | 4096 | 8 | 211 | 0.1491 | 0.0058 |
| 10 | 65536 | 4096 | 64 | 128 | 0.4065 | 0.0267 |
| 11 | 65536 | 4096 | 1 | 256 | 0.1024 | 0.0012 |
| 12 | 65536 | 4096 | 32 | 512 | 0.7312 | 0.0529 |
| 13 | 65536 | 4096 | 4 | 256 | 0.1218 | 0.0037 |
| 14 | 65536 | 4096 | 8 | 4096 | 1.3751 | 0.1054 |
| 15 | 65536 | 4096 | 16 | 512 | 0.4053 | 0.0267 |
| 16 | 65536 | 4096 | 1 | 512 | 0.1036 | 0.0020 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    vocab_size = axes_and_scalars["vocab_size"]
    hidden_size = axes_and_scalars["hidden_size"]
    
    # Gradient from next layer
    grad_output = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Token indices (must be valid indices into embedding table)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device)
    
    # Hidden states saved from forward pass (in float32)
    hidden_states_fp32 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Reciprocal standard deviation from forward pass
    # rstd = 1/sqrt(variance + eps), so it should be positive
    variance = torch.rand(batch_size, seq_len, 1, dtype=torch.float32, device=device) * 2.0 + 0.1
    rstd = torch.rsqrt(variance + 1e-6)
    
    # RMSNorm weight
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    
    return {
        "grad_output": grad_output,
        "input_ids": input_ids,
        "hidden_states_fp32": hidden_states_fp32,
        "rstd": rstd,
        "norm_weight": norm_weight,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    input_ids: torch.Tensor,
    hidden_states_fp32: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
):
    """
    Backward pass for fused embedding + RMSNorm.
    
    Computes:
    1. grad_norm_weight: gradient w.r.t. RMSNorm scale parameter
    2. grad_embed_weight: gradient w.r.t. embedding table (scatter-add)
    
    Args:
        grad_output: (batch_size, seq_len, hidden_size) gradient from next layer
        input_ids: (batch_size, seq_len) token indices
        hidden_states_fp32: (batch_size, seq_len, hidden_size) saved hidden states
        rstd: (batch_size, seq_len, 1) reciprocal standard deviation
        norm_weight: (hidden_size,) RMSNorm scale parameter
    
    Returns:
        grad_embed_weight: (vocab_size, hidden_size) gradient for embedding table
        grad_norm_weight: (hidden_size,) gradient for RMSNorm weight
    """
    vocab_size = 65536
    hidden_size = 4096
    
    # Convert grad_output to float32 for numerical stability
    grad_output_fp32 = grad_output.to(torch.float32)
    
    # Compute normalized hidden states
    hidden_states_normalized = hidden_states_fp32 * rstd
    
    # Step 1: Gradient w.r.t. norm_weight
    # output = norm_weight * hidden_states_normalized
    # grad_norm_weight = sum over (batch, seq) of (grad_output * hidden_states_normalized)
    grad_norm_weight = (grad_output_fp32 * hidden_states_normalized).sum(dim=(0, 1))
    
    # Step 2: Gradient w.r.t. normalized hidden states
    grad_hidden_states_normalized = grad_output_fp32 * norm_weight.to(torch.float32)
    
    # Step 3: Gradient through RMSNorm normalization
    # Using the chain rule for RMSNorm:
    # grad_h = rstd * (grad_h_norm - mean(grad_h_norm * h_norm) * h_norm)
    mean_grad_normalized = (grad_hidden_states_normalized * hidden_states_normalized).mean(dim=-1, keepdim=True)
    grad_hidden_states_fp32 = rstd * (grad_hidden_states_normalized - mean_grad_normalized * hidden_states_normalized)
    
    # Step 4: Gradient through embedding lookup (scatter-add)
    grad_embed_weight = torch.zeros(
        vocab_size, hidden_size,
        dtype=torch.float32,
        device=grad_output.device
    )
    
    # Flatten for scatter operation
    input_ids_flat = input_ids.view(-1)
    grad_hidden_states_flat = grad_hidden_states_fp32.view(-1, hidden_size)
    
    # Scatter add gradients to embedding table
    grad_embed_weight.index_add_(0, input_ids_flat, grad_hidden_states_flat)
    
    # Convert back to bfloat16
    grad_embed_weight = grad_embed_weight.to(torch.bfloat16)
    grad_norm_weight = grad_norm_weight.to(torch.bfloat16)
    
    return grad_embed_weight, grad_norm_weight
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.013295 ms |
| - | Scoring Baseline | 0.500000 | 0.296856 ms |
| - | Reference Implementation | 0.192240 | 1.207684 ms |
