## Description

Backward pass for SwiGLU MLP: computes gradients for input, gate_weight, up_weight, and down_weight through the chain output = down_proj(silu(gate_proj(x)) * up_proj(x))
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | bfloat16 |
| x | [batch_size, seq_len, hidden_size] | bfloat16 |
| gate_output | [batch_size, seq_len, intermediate_size] | bfloat16 |
| up_output | [batch_size, seq_len, intermediate_size] | bfloat16 |
| activated_gate | [batch_size, seq_len, intermediate_size] | bfloat16 |
| gate_weight | [intermediate_size, hidden_size] | bfloat16 |
| up_weight | [intermediate_size, hidden_size] | bfloat16 |
| down_weight | [hidden_size, intermediate_size] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_x | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_gate_weight | [intermediate_size, hidden_size] | bfloat16 |
| grad_up_weight | [intermediate_size, hidden_size] | bfloat16 |
| grad_down_weight | [hidden_size, intermediate_size] | bfloat16 |

| # | hidden_size | intermediate_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16384 | 53248 | 32 | 128 | 32.5771 | 23.6752 |
| 2 | 16384 | 53248 | 4 | 256 | 7.8241 | 5.9191 |
| 3 | 16384 | 53248 | 8 | 128 | 8.0456 | 5.9191 |
| 4 | 16384 | 53248 | 16 | 256 | 32.3679 | 23.6752 |
| 5 | 16384 | 53248 | 1 | 4096 | 33.2339 | 23.6752 |
| 6 | 16384 | 53248 | 2 | 1024 | 15.9422 | 11.8378 |
| 7 | 16384 | 53248 | 1 | 256 | 2.6236 | 1.4801 |
| 8 | 16384 | 53248 | 4 | 1024 | 33.2523 | 23.6752 |
| 9 | 16384 | 53248 | 8 | 512 | 33.1329 | 23.6752 |
| 10 | 16384 | 53248 | 16 | 211 | 28.7763 | 19.5136 |
| 11 | 16384 | 53248 | 16 | 128 | 16.2176 | 11.8378 |
| 12 | 16384 | 53248 | 2 | 256 | 4.1173 | 2.9597 |
| 13 | 16384 | 53248 | 64 | 128 | 70.0833 | 47.3499 |
| 14 | 16384 | 53248 | 4 | 1571 | 53.3161 | 36.3217 |
| 15 | 16384 | 53248 | 1 | 128 | 2.2404 | 0.9200 |
| 16 | 16384 | 53248 | 8 | 373 | 24.9173 | 17.2478 |
| 17 | 16384 | 53248 | 4 | 512 | 16.0692 | 11.8378 |
| 18 | 16384 | 53248 | 2 | 2048 | 33.5678 | 23.6752 |

```python
import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    activated_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    """
    Backward pass for SwiGLU MLP.
    
    Gradient flow:
    1. grad_output -> through down_proj
    2. grad_gated_output -> through element-wise multiply
    3. grad_activated_gate -> through SiLU
    4. grad_gate_output -> through gate_proj
    5. grad_up_output -> through up_proj
    6. Accumulate grad_x from both gate and up paths
    """
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = gate_output.shape[-1]
    
    # Step 1: Gradient through down_proj
    # output = gated_output @ down_weight.T
    # grad_gated_output = grad_output @ down_weight
    grad_gated_output = grad_output.matmul(down_weight)  # [B, S, I]
    
    # grad_down_weight = grad_output.T @ gated_output
    grad_output_2d = grad_output.reshape(-1, hidden_size)  # [B*S, H]
    gated_output = activated_gate * up_output  # Recompute
    gated_output_2d = gated_output.reshape(-1, intermediate_size)  # [B*S, I]
    grad_down_weight = grad_output_2d.t().matmul(gated_output_2d)  # [H, I]
    
    # Step 2: Gradient through element-wise multiply
    # gated_output = activated_gate * up_output
    grad_activated_gate = grad_gated_output * up_output  # [B, S, I]
    grad_up_output = grad_gated_output * activated_gate  # [B, S, I]
    
    # Step 3: Gradient through SiLU activation
    # activated_gate = silu(gate_output) = gate_output * sigmoid(gate_output)
    # d/dx[silu(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sigmoid_gate = torch.sigmoid(gate_output.to(torch.float32))  # [B, S, I]
    gate_output_f32 = gate_output.to(torch.float32)
    silu_grad = sigmoid_gate * (1.0 + gate_output_f32 * (1.0 - sigmoid_gate))
    grad_gate_output = (grad_activated_gate.to(torch.float32) * silu_grad).to(grad_output.dtype)  # [B, S, I]
    
    # Step 4: Gradient through gate_proj
    # gate_output = x @ gate_weight.T
    # grad_x_gate = grad_gate_output @ gate_weight
    grad_x_gate = grad_gate_output.matmul(gate_weight)  # [B, S, H]
    
    # grad_gate_weight = grad_gate_output.T @ x
    grad_gate_output_2d = grad_gate_output.reshape(-1, intermediate_size)  # [B*S, I]
    x_2d = x.reshape(-1, hidden_size)  # [B*S, H]
    grad_gate_weight = grad_gate_output_2d.t().matmul(x_2d)  # [I, H]
    
    # Step 5: Gradient through up_proj
    # up_output = x @ up_weight.T
    # grad_x_up = grad_up_output @ up_weight
    grad_x_up = grad_up_output.matmul(up_weight)  # [B, S, H]
    
    # grad_up_weight = grad_up_output.T @ x
    grad_up_output_2d = grad_up_output.reshape(-1, intermediate_size)  # [B*S, I]
    grad_up_weight = grad_up_output_2d.t().matmul(x_2d)  # [I, H]
    
    # Step 6: Accumulate input gradients from both paths
    grad_x = grad_x_gate + grad_x_up  # [B, S, H]
    
    return grad_x, grad_gate_weight, grad_up_weight, grad_down_weight
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 11.927621 ms |
| - | Scoring Baseline | 0.500000 | 17.426926 ms |
| - | Reference Implementation | 0.394639 | 20.181747 ms |
