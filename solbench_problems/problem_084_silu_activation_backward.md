## Description

Backward pass for SiLU (Swish) activation function. Computes gradient with respect to input: grad_input = grad_output * sigmoid(x) * [1 + x * (1 - sigmoid(x))]
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [num_elements] | float32 |
| x | [num_elements] | float32 |
| sigmoid_x | [num_elements] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_input | [num_elements] | float32 |

| # | num_elements | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- |
| 1 | 40960 | 0.0097 | 0.0004 |
| 2 | 16384 | 0.0173 | 0.0004 |
| 3 | 16777216 | 0.0481 | 0.0092 |
| 4 | 131 | 0.0083 | 0.0004 |
| 5 | 655360 | 0.0095 | 0.0007 |
| 6 | 786432 | 0.0096 | 0.0008 |
| 7 | 163840 | 0.0087 | 0.0005 |
| 8 | 8388608 | 0.0298 | 0.0048 |
| 9 | 4194304 | 0.0210 | 0.0026 |
| 10 | 4096 | 0.0086 | 0.0004 |
| 11 | 3089 | 0.0126 | 0.0004 |
| 12 | 2097152 | 0.0133 | 0.0015 |
| 13 | 262144 | 0.0087 | 0.0005 |
| 14 | 5242880 | 0.0215 | 0.0031 |
| 15 | 2053 | 0.0087 | 0.0004 |
| 16 | 10485760 | 0.0338 | 0.0059 |

```python
import torch

@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    """
    Backward pass for SiLU activation.
    
    Gradient formula:
        grad_input = grad_output * sigmoid(x) * [1 + x * (1 - sigmoid(x))]
    
    Args:
        grad_output: Upstream gradient, shape [num_elements]
        x: Original input from forward pass, shape [num_elements]
        sigmoid_x: Cached sigmoid(x) from forward pass, shape [num_elements]
    
    Returns:
        grad_input: Gradient with respect to input x, shape [num_elements]
    """
    # Compute: 1 - sigmoid(x)
    one_minus_sigmoid = 1.0 - sigmoid_x
    
    # Compute: x * (1 - sigmoid(x))
    x_times_one_minus_sigmoid = x * one_minus_sigmoid
    
    # Compute: 1 + x * (1 - sigmoid(x))
    bracket_term = 1.0 + x_times_one_minus_sigmoid
    
    # Compute: sigmoid(x) * [1 + x * (1 - sigmoid(x))]
    local_grad = sigmoid_x * bracket_term
    
    # Apply chain rule: grad_output * local_gradient
    grad_input = grad_output * local_grad
    
    return grad_input
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001038 ms |
| - | Scoring Baseline | 0.500000 | 0.014174 ms |
| - | Reference Implementation | 0.271063 | 0.036231 ms |
