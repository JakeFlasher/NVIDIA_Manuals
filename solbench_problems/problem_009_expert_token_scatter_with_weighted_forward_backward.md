## Description

Backward pass for expert token scatter with weighted forward. Computes gradients for hidden_states, routing_weights, and expert MLP weights (w1, w2, w3) through SwiGLU activation.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_seq_len, hidden_dim] | bfloat16 |
| token_indices | [num_tokens] | int64 |
| selected_tokens | [num_tokens, hidden_dim] | bfloat16 |
| w1_output | [num_tokens, ffn_dim] | bfloat16 |
| gate_output | [num_tokens, ffn_dim] | bfloat16 |
| up_output | [num_tokens, ffn_dim] | bfloat16 |
| gated_output | [num_tokens, ffn_dim] | bfloat16 |
| expert_output | [num_tokens, hidden_dim] | bfloat16 |
| selected_weights | [num_tokens] | bfloat16 |
| w1_weight | [ffn_dim, hidden_dim] | bfloat16 |
| w2_weight | [hidden_dim, ffn_dim] | bfloat16 |
| w3_weight | [ffn_dim, hidden_dim] | bfloat16 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_seq_len, hidden_dim] | bfloat16 |
| grad_routing_weights | [batch_seq_len] | bfloat16 |
| grad_w1_weight | [ffn_dim, hidden_dim] | bfloat16 |
| grad_w2_weight | [hidden_dim, ffn_dim] | bfloat16 |
| grad_w3_weight | [ffn_dim, hidden_dim] | bfloat16 |

| # | hidden_dim | ffn_dim | batch_seq_len | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4096 | 14336 | 512 | 64 | 1.0954 | 0.0951 |
| 2 | 4096 | 14336 | 997 | 128 | 1.6829 | 0.0978 |
| 3 | 4096 | 14336 | 512 | 256 | 2.5942 | 0.1000 |
| 4 | 4096 | 14336 | 1571 | 196 | 2.2631 | 0.1009 |
| 5 | 4096 | 14336 | 1024 | 128 | 1.6845 | 0.0979 |
| 6 | 4096 | 14336 | 2689 | 336 | 3.2295 | 0.1311 |
| 7 | 4096 | 14336 | 4096 | 256 | 2.6247 | 0.1101 |
| 8 | 4096 | 14336 | 4096 | 2048 | 15.6486 | 0.7971 |
| 9 | 4096 | 14336 | 541 | 64 | 1.1032 | 0.0952 |
| 10 | 4096 | 14336 | 2053 | 256 | 2.6063 | 0.1035 |
| 11 | 4096 | 14336 | 1024 | 512 | 4.4478 | 0.1996 |
| 12 | 4096 | 14336 | 1721 | 215 | 2.3749 | 0.1017 |
| 13 | 4096 | 14336 | 2048 | 256 | 2.6066 | 0.1035 |
| 14 | 4096 | 14336 | 3967 | 496 | 4.3532 | 0.1934 |
| 15 | 4096 | 14336 | 256 | 64 | 1.0973 | 0.0943 |
| 16 | 4096 | 14336 | 2371 | 296 | 2.9587 | 0.1156 |

```python
import torch

def get_inputs(
    axes_and_scalars: dict[str, int], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_dim = axes_and_scalars["hidden_dim"]
    ffn_dim = axes_and_scalars["ffn_dim"]
    
    # Ensure num_tokens <= batch_seq_len
    num_tokens = min(num_tokens, batch_seq_len)
    
    # Generate random token indices (unique indices into batch_seq_len)
    perm = torch.randperm(batch_seq_len, device=device)
    token_indices = perm[:num_tokens].sort().values.to(torch.int64)
    
    # Gradient from next layer
    grad_output = torch.randn(batch_seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Selected tokens (gathered from hidden_states)
    selected_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Forward pass intermediate values
    w1_output = torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    gate_output = torch.nn.functional.silu(w1_output.float()).to(torch.bfloat16)
    up_output = torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    gated_output = gate_output * up_output
    expert_output = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Routing weights for selected tokens
    selected_weights = torch.rand(num_tokens, device=device, dtype=torch.bfloat16)
    
    # Expert MLP weights
    w1_weight = torch.randn(ffn_dim, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02
    w2_weight = torch.randn(hidden_dim, ffn_dim, device=device, dtype=torch.bfloat16) * 0.02
    w3_weight = torch.randn(ffn_dim, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02
    
    return {
        "grad_output": grad_output,
        "token_indices": token_indices,
        "selected_tokens": selected_tokens,
        "w1_output": w1_output,
        "gate_output": gate_output,
        "up_output": up_output,
        "gated_output": gated_output,
        "expert_output": expert_output,
        "selected_weights": selected_weights,
        "w1_weight": w1_weight,
        "w2_weight": w2_weight,
        "w3_weight": w3_weight,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    token_indices: torch.Tensor,
    selected_tokens: torch.Tensor,
    w1_output: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    gated_output: torch.Tensor,
    expert_output: torch.Tensor,
    selected_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
):
    """
    Backward pass for expert token scatter with weighted forward.
    
    Gradient flow:
    1. Gather gradients from scattered positions
    2. Backprop through routing weight multiplication
    3. Backprop through w2 linear layer
    4. Backprop through element-wise gating
    5. Backprop through w3 linear layer (up branch)
    6. Backprop through SiLU activation and w1 linear layer (gate branch)
    7. Accumulate gradients and scatter back
    """
    batch_seq_len = grad_output.shape[0]
    hidden_dim = grad_output.shape[1]
    num_tokens = token_indices.shape[0]
    ffn_dim = w1_weight.shape[0]
    
    device = grad_output.device
    
    # Convert to float32 for numerical stability
    grad_output_f = grad_output.to(torch.float32)
    selected_tokens_f = selected_tokens.to(torch.float32)
    w1_output_f = w1_output.to(torch.float32)
    gate_output_f = gate_output.to(torch.float32)
    up_output_f = up_output.to(torch.float32)
    gated_output_f = gated_output.to(torch.float32)
    expert_output_f = expert_output.to(torch.float32)
    selected_weights_f = selected_weights.to(torch.float32)
    w1_weight_f = w1_weight.to(torch.float32)
    w2_weight_f = w2_weight.to(torch.float32)
    w3_weight_f = w3_weight.to(torch.float32)
    
    # Step 1: Gather gradients from scattered positions
    grad_weighted_output = grad_output_f[token_indices]  # (num_tokens, hidden_dim)
    
    # Step 2: Backprop through routing weight multiplication
    # weighted_output = expert_output * selected_weights.unsqueeze(-1)
    # grad_selected_weights = (grad_weighted_output * expert_output).sum(dim=-1)
    grad_selected_weights = (grad_weighted_output * expert_output_f).sum(dim=-1)  # (num_tokens,)
    
    # Scatter to full routing_weights tensor
    grad_routing_weights = torch.zeros(batch_seq_len, dtype=torch.float32, device=device)
    grad_routing_weights[token_indices] = grad_selected_weights
    
    # Gradient w.r.t. expert_output
    grad_expert_output = grad_weighted_output * selected_weights_f.unsqueeze(-1)  # (num_tokens, hidden_dim)
    
    # Step 3: Backprop through w2 linear layer
    # expert_output = gated_output @ w2_weight.T
    # grad_w2_weight = grad_expert_output.T @ gated_output
    grad_w2_weight = grad_expert_output.t() @ gated_output_f  # (hidden_dim, ffn_dim)
    
    # Gradient w.r.t. gated_output
    grad_gated_output = grad_expert_output @ w2_weight_f  # (num_tokens, ffn_dim)
    
    # Step 4: Backprop through element-wise gating
    # gated_output = gate_output * up_output
    grad_gate_output = grad_gated_output * up_output_f  # (num_tokens, ffn_dim)
    grad_up_output = grad_gated_output * gate_output_f  # (num_tokens, ffn_dim)
    
    # Step 5: Backprop through w3 linear layer (up branch)
    # up_output = selected_tokens @ w3_weight.T
    grad_w3_weight = grad_up_output.t() @ selected_tokens_f  # (ffn_dim, hidden_dim)
    grad_selected_tokens_w3 = grad_up_output @ w3_weight_f  # (num_tokens, hidden_dim)
    
    # Step 6: Backprop through SiLU activation and w1 linear layer
    # gate_output = silu(w1_output) = w1_output * sigmoid(w1_output)
    # d(silu)/d(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sigmoid_w1 = torch.sigmoid(w1_output_f)
    grad_w1_output = grad_gate_output * (sigmoid_w1 * (1 + w1_output_f * (1 - sigmoid_w1)))  # (num_tokens, ffn_dim)
    
    # w1_output = selected_tokens @ w1_weight.T
    grad_w1_weight = grad_w1_output.t() @ selected_tokens_f  # (ffn_dim, hidden_dim)
    grad_selected_tokens_w1 = grad_w1_output @ w1_weight_f  # (num_tokens, hidden_dim)
    
    # Step 7: Accumulate gradients for selected_tokens
    grad_selected_tokens = grad_selected_tokens_w1 + grad_selected_tokens_w3  # (num_tokens, hidden_dim)
    
    # Step 8: Scatter gradients back to full hidden_states tensor
    grad_hidden_states = torch.zeros(batch_seq_len, hidden_dim, dtype=torch.float32, device=device)
    grad_hidden_states[token_indices] = grad_selected_tokens
    
    # Convert back to bfloat16
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_routing_weights.to(torch.bfloat16),
        grad_w1_weight.to(torch.bfloat16),
        grad_w2_weight.to(torch.bfloat16),
        grad_w3_weight.to(torch.bfloat16),
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.127058 ms |
| - | Scoring Baseline | 0.500000 | 2.523576 ms |
| - | Reference Implementation | 0.303834 | 5.638984 ms |
