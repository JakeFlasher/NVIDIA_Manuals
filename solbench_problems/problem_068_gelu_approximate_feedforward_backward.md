## Description

Backward pass for GELU approximate feedforward network. Computes gradients through Linear2 -> GELU_approx -> Linear1 chain. The forward pass is: hidden_states -> Linear1 (expansion to 4x) -> GELU_approx -> Linear2 (contraction) -> output.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, dim] | float32 |
| hidden_states | [batch_size, seq_len, dim] | float32 |
| weight1 | [hidden_dim, dim] | float32 |
| weight2 | [dim, hidden_dim] | float32 |
| linear1_out | [batch_size, seq_len, hidden_dim] | float32 |
| tanh_inner | [batch_size, seq_len, hidden_dim] | float32 |
| gelu_out | [batch_size, seq_len, hidden_dim] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, dim] | float32 |
| grad_weight1 | [hidden_dim, dim] | float32 |
| grad_weight2 | [dim, hidden_dim] | float32 |

| # | dim | hidden_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1536 | 6144 | 4 | 512 | 0.3197 | 0.0858 |
| 2 | 1536 | 6144 | 4 | 2048 | 0.7684 | 0.3419 |
| 3 | 1536 | 6144 | 4 | 1024 | 0.4790 | 0.1711 |
| 4 | 1536 | 6144 | 8 | 256 | 0.3340 | 0.0858 |
| 5 | 1536 | 6144 | 8 | 512 | 0.4813 | 0.1711 |
| 6 | 1536 | 6144 | 1 | 256 | 0.1981 | 0.0111 |
| 7 | 1536 | 6144 | 16 | 256 | 0.4823 | 0.1711 |
| 8 | 1536 | 6144 | 2 | 1571 | 0.4100 | 0.1314 |
| 9 | 1536 | 6144 | 32 | 128 | 0.4839 | 0.1711 |
| 10 | 1536 | 6144 | 1 | 512 | 0.2126 | 0.0217 |
| 11 | 1536 | 6144 | 2 | 512 | 0.2595 | 0.0431 |
| 12 | 1536 | 6144 | 1 | 1024 | 0.2487 | 0.0431 |
| 13 | 1536 | 6144 | 2 | 4096 | 0.7758 | 0.3419 |
| 14 | 1536 | 6144 | 2 | 128 | 0.2051 | 0.0111 |
| 15 | 1536 | 6144 | 4 | 256 | 0.2601 | 0.0431 |
| 16 | 1536 | 6144 | 8 | 613 | 0.5437 | 0.2048 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs with properly constrained tanh_inner values."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    dim = axes_and_scalars["dim"]
    hidden_dim = axes_and_scalars["hidden_dim"]

    grad_output = torch.randn(batch_size, seq_len, dim, device=device)
    hidden_states = torch.randn(batch_size, seq_len, dim, device=device)
    weight1 = torch.randn(hidden_dim, dim, device=device) * 0.02
    weight2 = torch.randn(dim, hidden_dim, device=device) * 0.02

    # Compute linear1_out from forward pass
    linear1_out = hidden_states.matmul(weight1.t())

    # Compute tanh_inner from forward pass (must be in [-1, 1])
    inner = 0.7978845608028654 * (linear1_out + 0.044715 * linear1_out.pow(3))
    tanh_inner = torch.tanh(inner)

    # Compute gelu_out from forward pass
    gelu_out = 0.5 * linear1_out * (1.0 + tanh_inner)

    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "weight1": weight1,
        "weight2": weight2,
        "linear1_out": linear1_out,
        "tanh_inner": tanh_inner,
        "gelu_out": gelu_out,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    linear1_out: torch.Tensor,
    tanh_inner: torch.Tensor,
    gelu_out: torch.Tensor,
):
    """
    Backward pass for GELU approximate feedforward.

    Gradient flow:
    grad_output -> grad_linear2 -> grad_gelu -> grad_linear1 -> grad_input

    GELU(x) = 0.5 * x * (1 + tanh(inner))
    where inner = sqrt(2/pi) * (x + 0.044715 * x^3)

    d(GELU)/dx = 0.5 * (1 + tanh(inner)) +
                 0.5 * x * sech^2(inner) * d(inner)/dx

    d(inner)/dx = sqrt(2/pi) * (1 + 0.134145 * x^2)
    """
    # ============================================================
    # Backward through Linear2: output = gelu_out @ weight2.T
    # ============================================================

    # Gradient w.r.t. gelu_out
    grad_gelu_out = grad_output.matmul(weight2)

    # Gradient w.r.t. weight2
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    gelu_out_2d = gelu_out.reshape(-1, gelu_out.shape[-1])
    grad_weight2 = grad_output_2d.t().matmul(gelu_out_2d)

    # ============================================================
    # Backward through GELU approximate activation
    # ============================================================

    # Part 1: 0.5 * (1 + tanh(inner))
    gelu_grad_part1 = 0.5 * (1.0 + tanh_inner)

    # Part 2: 0.5 * x * sech^2(inner) * d(inner)/dx
    sech_squared = 1.0 - tanh_inner * tanh_inner
    d_inner_dx = 0.7978845608028654 * (1.0 + 0.134145 * linear1_out * linear1_out)
    gelu_grad_part2 = 0.5 * linear1_out * sech_squared * d_inner_dx

    # Total GELU gradient
    gelu_grad = gelu_grad_part1 + gelu_grad_part2

    # Apply chain rule
    grad_linear1_out = grad_gelu_out * gelu_grad

    # ============================================================
    # Backward through Linear1: linear1_out = hidden_states @ weight1.T
    # ============================================================

    # Gradient w.r.t. hidden_states
    grad_hidden_states = grad_linear1_out.matmul(weight1)

    # Gradient w.r.t. weight1
    grad_linear1_out_2d = grad_linear1_out.reshape(-1, grad_linear1_out.shape[-1])
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.082085 ms |
| - | Scoring Baseline | 0.500000 | 0.367507 ms |
| - | Reference Implementation | 0.333420 | 0.655418 ms |
