## Description

Backward pass for fused FFN block with GELU activation, residual connection, and layer normalization. Computes gradients for hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, ln_weight, and ln_bias.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | float32 |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| fc1_weight | [intermediate_size, hidden_size] | float32 |
| fc1_output | [batch_size, seq_len, intermediate_size] | float32 |
| gelu_output | [batch_size, seq_len, intermediate_size] | float32 |
| fc2_weight | [hidden_size, intermediate_size] | float32 |
| residual_output | [batch_size, seq_len, hidden_size] | float32 |
| normalized | [batch_size, seq_len, hidden_size] | float32 |
| var | [batch_size, seq_len, 1] | float32 |
| ln_weight | [hidden_size] | float32 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| grad_fc1_weight | [intermediate_size, hidden_size] | float32 |
| grad_fc1_bias | [intermediate_size] | float32 |
| grad_fc2_weight | [hidden_size, intermediate_size] | float32 |
| grad_fc2_bias | [hidden_size] | float32 |
| grad_ln_weight | [hidden_size] | float32 |
| grad_ln_bias | [hidden_size] | float32 |

| # | hidden_size | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 2048 | 1 | 131 | 0.2094 | 0.0014 |
| 2 | 512 | 2048 | 8 | 373 | 0.2643 | 0.0142 |
| 3 | 512 | 2048 | 16 | 919 | 0.5300 | 0.0685 |
| 4 | 512 | 2048 | 4 | 1879 | 0.3597 | 0.0352 |
| 5 | 512 | 2048 | 64 | 1087 | 1.7812 | 0.3226 |
| 6 | 512 | 2048 | 2 | 1801 | 0.2752 | 0.0171 |
| 7 | 512 | 2048 | 16 | 449 | 0.3531 | 0.0337 |
| 8 | 512 | 2048 | 1 | 1163 | 0.2201 | 0.0058 |
| 9 | 512 | 2048 | 8 | 853 | 0.3381 | 0.0320 |
| 10 | 512 | 2048 | 8 | 1423 | 0.4456 | 0.0531 |
| 11 | 512 | 2048 | 1 | 128 | 0.1965 | 0.0014 |
| 12 | 512 | 2048 | 2 | 211 | 0.2348 | 0.0024 |
| 13 | 512 | 2048 | 32 | 541 | 0.5869 | 0.0806 |
| 14 | 512 | 2048 | 4 | 773 | 0.2681 | 0.0147 |
| 15 | 512 | 2048 | 1 | 613 | 0.2246 | 0.0032 |
| 16 | 512 | 2048 | 32 | 4096 | 3.1635 | 0.6074 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate realistic backward pass inputs by running the forward pass."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    eps = 1e-5

    with torch.no_grad():
        # Realistic input hidden states (small scale like embeddings)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.1

        # Weights with Kaiming-style initialization
        fc1_weight = torch.randn(intermediate_size, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        fc2_weight = torch.randn(hidden_size, intermediate_size, device=device) * (2.0 / intermediate_size) ** 0.5
        ln_weight = torch.ones(hidden_size, device=device)

        # Forward pass: FC1
        fc1_output = hidden_states.matmul(fc1_weight.t())

        # Forward pass: GELU (tanh approximation)
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (fc1_output + coeff * fc1_output.pow(3))
        gelu_output = 0.5 * fc1_output * (1.0 + torch.tanh(inner))

        # Forward pass: FC2
        fc2_output = gelu_output.matmul(fc2_weight.t())

        # Forward pass: Residual addition
        residual_output = fc2_output + hidden_states

        # Forward pass: Layer norm
        mean = residual_output.mean(dim=-1, keepdim=True)
        var = residual_output.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (residual_output - mean) / torch.sqrt(var + eps)

        # Unit-scale grad_output
        grad_output = torch.randn(batch_size, seq_len, hidden_size, device=device)

    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "fc1_weight": fc1_weight,
        "fc1_output": fc1_output,
        "gelu_output": gelu_output,
        "fc2_weight": fc2_weight,
        "residual_output": residual_output,
        "normalized": normalized,
        "var": var,
        "ln_weight": ln_weight,
        "eps": eps,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_output: torch.Tensor,
    gelu_output: torch.Tensor,
    fc2_weight: torch.Tensor,
    residual_output: torch.Tensor,
    normalized: torch.Tensor,
    var: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
):
    """
    Backward pass for fused FFN with GELU, residual, and layer norm.
    
    Computes gradients through:
    1. Layer norm backward
    2. Residual addition backward
    3. FC2 backward (contraction)
    4. GELU backward
    5. FC1 backward (expansion)
    """
    B, S, H = grad_output.shape  # H = 512
    I = fc1_weight.shape[0]  # I = 2048
    
    # ========================================================================
    # BACKWARD THROUGH LAYER NORM
    # ========================================================================
    # output = normalized * ln_weight + ln_bias
    
    # Gradient w.r.t. ln_weight: sum over batch and sequence
    grad_ln_weight = (grad_output * normalized).sum(dim=(0, 1))  # [512]
    
    # Gradient w.r.t. ln_bias: sum over batch and sequence
    grad_ln_bias = grad_output.sum(dim=(0, 1))  # [512]
    
    # Gradient w.r.t. normalized
    grad_normalized = grad_output * ln_weight  # [B, S, 512]
    
    # Backward through normalization: normalized = (x - mean) / sqrt(var + eps)
    std = torch.sqrt(var + eps)  # [B, S, 1]
    
    # Layer norm gradient formula
    grad_normalized_mean = grad_normalized.mean(dim=-1, keepdim=True)  # [B, S, 1]
    grad_normalized_normalized_mean = (grad_normalized * normalized).mean(dim=-1, keepdim=True)  # [B, S, 1]
    
    grad_residual_output = (1.0 / std) * (
        grad_normalized - grad_normalized_mean - normalized * grad_normalized_normalized_mean
    )  # [B, S, 512]
    
    # ========================================================================
    # BACKWARD THROUGH RESIDUAL ADDITION
    # ========================================================================
    # residual_output = fc2_output + residual
    grad_fc2_output = grad_residual_output  # [B, S, 512]
    grad_residual = grad_residual_output  # [B, S, 512]
    
    # ========================================================================
    # BACKWARD THROUGH FC2 (CONTRACTION LAYER)
    # ========================================================================
    # fc2_output = gelu_output @ fc2_weight.T + fc2_bias
    
    # Gradient w.r.t. fc2_bias
    grad_fc2_bias = grad_fc2_output.sum(dim=(0, 1))  # [512]
    
    # Gradient w.r.t. fc2_weight
    grad_fc2_output_reshaped = grad_fc2_output.view(-1, H)  # [B*S, 512]
    gelu_output_reshaped = gelu_output.view(-1, I)  # [B*S, 2048]
    grad_fc2_weight = grad_fc2_output_reshaped.t() @ gelu_output_reshaped  # [512, 2048]
    
    # Gradient w.r.t. gelu_output
    grad_gelu_output = grad_fc2_output @ fc2_weight  # [B, S, 2048]
    
    # ========================================================================
    # BACKWARD THROUGH GELU ACTIVATION
    # ========================================================================
    # GELU gradient using tanh approximation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    
    x = fc1_output
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_out = torch.tanh(tanh_arg)
    
    dtanh_arg_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
    sech_sq = 1.0 - tanh_out * tanh_out
    
    gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * dtanh_arg_dx
    
    grad_fc1_output = grad_gelu_output * gelu_grad  # [B, S, 2048]
    
    # ========================================================================
    # BACKWARD THROUGH FC1 (EXPANSION LAYER)
    # ========================================================================
    # fc1_output = hidden_states @ fc1_weight.T + fc1_bias
    
    # Gradient w.r.t. fc1_bias
    grad_fc1_bias = grad_fc1_output.sum(dim=(0, 1))  # [2048]
    
    # Gradient w.r.t. fc1_weight
    grad_fc1_output_reshaped = grad_fc1_output.view(-1, I)  # [B*S, 2048]
    hidden_states_reshaped = hidden_states.view(-1, H)  # [B*S, 512]
    grad_fc1_weight = grad_fc1_output_reshaped.t() @ hidden_states_reshaped  # [2048, 512]
    
    # Gradient w.r.t. hidden_states (from FC1)
    grad_hidden_states_fc1 = grad_fc1_output @ fc1_weight  # [B, S, 512]
    
    # ========================================================================
    # COMBINE GRADIENTS FOR HIDDEN_STATES
    # ========================================================================
    grad_hidden_states = grad_hidden_states_fc1 + grad_residual  # [B, S, 512]
    
    return (
        grad_hidden_states,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.020045 ms |
| - | Scoring Baseline | 0.500000 | 0.392129 ms |
| - | Reference Implementation | 0.302880 | 0.872783 ms |
