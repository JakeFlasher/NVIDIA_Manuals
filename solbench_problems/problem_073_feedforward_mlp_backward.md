## Description

Backward pass for feedforward MLP: computes gradients w.r.t. input hidden states, weight1, bias1, weight2, bias2 given upstream gradient. Forward: hidden_states -> Linear1 -> GELU -> Linear2 -> output.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | float32 |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| weight1 | [intermediate_size, hidden_size] | float32 |
| bias1 | [intermediate_size] | float32 |
| weight2 | [hidden_size, intermediate_size] | float32 |
| bias2 | [hidden_size] | float32 |
| intermediate | [batch_size, seq_len, intermediate_size] | float32 |
| intermediate_activated | [batch_size, seq_len, intermediate_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| grad_weight1 | [intermediate_size, hidden_size] | float32 |
| grad_bias1 | [intermediate_size] | float32 |
| grad_weight2 | [hidden_size, intermediate_size] | float32 |
| grad_bias2 | [hidden_size] | float32 |

| # | hidden_size | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 8192 | 2 | 256 | 0.2464 | 0.0383 |
| 2 | 2048 | 8192 | 32 | 256 | 1.7607 | 0.6074 |
| 3 | 2048 | 8192 | 2 | 4096 | 1.7643 | 0.6074 |
| 4 | 2048 | 8192 | 8 | 256 | 0.5429 | 0.1522 |
| 5 | 2048 | 8192 | 16 | 1024 | 3.5190 | 1.2145 |
| 6 | 2048 | 8192 | 8 | 773 | 1.3755 | 0.4586 |
| 7 | 2048 | 8192 | 2 | 512 | 0.3322 | 0.0763 |
| 8 | 2048 | 8192 | 64 | 128 | 1.7552 | 0.6074 |
| 9 | 2048 | 8192 | 8 | 128 | 0.3290 | 0.0763 |
| 10 | 2048 | 8192 | 4 | 512 | 0.5436 | 0.1522 |
| 11 | 2048 | 8192 | 1 | 1024 | 0.3326 | 0.0763 |
| 12 | 2048 | 8192 | 4 | 541 | 0.5617 | 0.1608 |
| 13 | 2048 | 8192 | 2 | 128 | 0.2045 | 0.0194 |
| 14 | 2048 | 8192 | 32 | 128 | 0.9445 | 0.3039 |
| 15 | 2048 | 8192 | 4 | 256 | 0.3311 | 0.0763 |
| 16 | 2048 | 8192 | 16 | 512 | 1.7506 | 0.6074 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    intermediate: torch.Tensor,
    intermediate_activated: torch.Tensor,
):
    """
    Backward pass for feedforward MLP.
    
    Forward was:
        intermediate = hidden_states @ weight1.T + bias1
        intermediate_activated = gelu(intermediate)
        output = intermediate_activated @ weight2.T + bias2
    
    Args:
        grad_output: [B, S, H] gradient w.r.t. output
        hidden_states: [B, S, H] original input
        weight1: [I, H] first linear weight
        bias1: [I] first linear bias
        weight2: [H, I] second linear weight
        bias2: [H] second linear bias
        intermediate: [B, S, I] pre-activation values
        intermediate_activated: [B, S, I] post-activation values
    
    Returns:
        grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2
    """
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = intermediate.shape[-1]
    
    # Backward through second linear layer
    # output = intermediate_activated @ weight2.T + bias2
    # grad_intermediate_activated = grad_output @ weight2
    grad_intermediate_activated = torch.matmul(grad_output, weight2)  # [B, S, I]
    
    # grad_weight2 = grad_output.T @ intermediate_activated
    # Reshape for matrix multiply: [B*S, H].T @ [B*S, I] -> [H, I]
    grad_output_reshaped = grad_output.reshape(-1, hidden_size)  # [B*S, H]
    intermediate_activated_reshaped = intermediate_activated.reshape(-1, intermediate_size)  # [B*S, I]
    grad_weight2 = torch.matmul(grad_output_reshaped.t(), intermediate_activated_reshaped)  # [H, I]
    
    # grad_bias2 = sum(grad_output) over batch and seq dims
    grad_bias2 = grad_output.sum(dim=[0, 1])  # [H]
    
    # Backward through GELU activation
    # GELU(x) = x * Phi(x) where Phi is standard normal CDF
    # d/dx GELU(x) = Phi(x) + x * phi(x) where phi is standard normal PDF
    # Using PyTorch's approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    
    x = intermediate
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_inner = torch.tanh(inner)
    
    # GELU = 0.5 * x * (1 + tanh(inner))
    # d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh(inner)^2) * d(inner)/dx
    # d(inner)/dx = sqrt(2/pi) * (1 + 3 * coeff * x^2)
    d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
    sech_squared = 1.0 - tanh_inner * tanh_inner
    
    gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * d_inner
    
    grad_intermediate = grad_intermediate_activated * gelu_grad  # [B, S, I]
    
    # Backward through first linear layer
    # intermediate = hidden_states @ weight1.T + bias1
    # grad_hidden_states = grad_intermediate @ weight1
    grad_hidden_states = torch.matmul(grad_intermediate, weight1)  # [B, S, H]
    
    # grad_weight1 = grad_intermediate.T @ hidden_states
    # Reshape: [B*S, I].T @ [B*S, H] -> [I, H]
    grad_intermediate_reshaped = grad_intermediate.reshape(-1, intermediate_size)  # [B*S, I]
    hidden_states_reshaped = hidden_states.reshape(-1, hidden_size)  # [B*S, H]
    grad_weight1 = torch.matmul(grad_intermediate_reshaped.t(), hidden_states_reshaped)  # [I, H]
    
    # grad_bias1 = sum(grad_intermediate) over batch and seq dims
    grad_bias1 = grad_intermediate.sum(dim=[0, 1])  # [I]
    
    return grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.186602 ms |
| - | Scoring Baseline | 0.500000 | 0.712256 ms |
| - | Reference Implementation | 0.308861 | 1.356564 ms |
