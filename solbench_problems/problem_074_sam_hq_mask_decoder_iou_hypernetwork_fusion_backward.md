## Description

Backward pass for SAM-HQ mask decoder IoU prediction and hypernetwork weight generation. Computes gradients for 5 MLPs: 1 IoU prediction MLP (256->1024->1024->4) and 4 parallel hypernetwork MLPs (256->256->256->32).
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_iou_scores | [batch_size, point_batch_size, num_mask_tokens] | float32 |
| grad_hyper_weights | [batch_size, point_batch_size, num_mask_tokens, output_channels] | float32 |
| iou_token_out | [batch_size, point_batch_size, hidden_size] | float32 |
| mask_tokens_out | [batch_size, point_batch_size, num_mask_tokens, hidden_size] | float32 |
| iou_proj_in_weight | [iou_head_hidden_dim, hidden_size] | float32 |
| iou_proj_in_bias | [iou_head_hidden_dim] | float32 |
| iou_hidden_weight | [iou_head_hidden_dim, iou_head_hidden_dim] | float32 |
| iou_hidden_bias | [iou_head_hidden_dim] | float32 |
| iou_proj_out_weight | [num_mask_tokens, iou_head_hidden_dim] | float32 |
| iou_proj_out_bias | [num_mask_tokens] | float32 |
| iou_hidden1 | [batch_size, point_batch_size, iou_head_hidden_dim] | float32 |
| iou_hidden1_relu | [batch_size, point_batch_size, iou_head_hidden_dim] | float32 |
| iou_hidden2 | [batch_size, point_batch_size, iou_head_hidden_dim] | float32 |
| iou_hidden2_relu | [batch_size, point_batch_size, iou_head_hidden_dim] | float32 |
| hyper_proj_in_weights | [num_mask_tokens, hidden_size, hidden_size] | float32 |
| hyper_proj_in_biases | [num_mask_tokens, hidden_size] | float32 |
| hyper_hidden_weights | [num_mask_tokens, hidden_size, hidden_size] | float32 |
| hyper_hidden_biases | [num_mask_tokens, hidden_size] | float32 |
| hyper_proj_out_weights | [num_mask_tokens, output_channels, hidden_size] | float32 |
| hyper_proj_out_biases | [num_mask_tokens, output_channels] | float32 |
| hyper_hidden1_0 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_1 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_2 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_3 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_relu_0 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_relu_1 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_relu_2 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden1_relu_3 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_0 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_1 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_2 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_3 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_relu_0 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_relu_1 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_relu_2 | [batch_size, point_batch_size, hidden_size] | float32 |
| hyper_hidden2_relu_3 | [batch_size, point_batch_size, hidden_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_iou_token_out | [batch_size, point_batch_size, hidden_size] | float32 |
| grad_mask_tokens_out | [batch_size, point_batch_size, num_mask_tokens, hidden_size] | float32 |
| grad_iou_proj_in_weight | [iou_head_hidden_dim, hidden_size] | float32 |
| grad_iou_proj_in_bias | [iou_head_hidden_dim] | float32 |
| grad_iou_hidden_weight | [iou_head_hidden_dim, iou_head_hidden_dim] | float32 |
| grad_iou_hidden_bias | [iou_head_hidden_dim] | float32 |
| grad_iou_proj_out_weight | [num_mask_tokens, iou_head_hidden_dim] | float32 |
| grad_iou_proj_out_bias | [num_mask_tokens] | float32 |
| grad_hyper_proj_in_weights | [num_mask_tokens, hidden_size, hidden_size] | float32 |
| grad_hyper_proj_in_biases | [num_mask_tokens, hidden_size] | float32 |
| grad_hyper_hidden_weights | [num_mask_tokens, hidden_size, hidden_size] | float32 |
| grad_hyper_hidden_biases | [num_mask_tokens, hidden_size] | float32 |
| grad_hyper_proj_out_weights | [num_mask_tokens, output_channels, hidden_size] | float32 |
| grad_hyper_proj_out_biases | [num_mask_tokens, output_channels] | float32 |

| # | hidden_size | iou_head_hidden_dim | num_mask_tokens | output_channels | batch_size | point_batch_size | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 1024 | 4 | 32 | 4 | 16 | 0.1783 | 0.0015 |
| 2 | 256 | 1024 | 4 | 32 | 64 | 1 | 0.1843 | 0.0015 |
| 3 | 256 | 1024 | 4 | 32 | 32 | 16 | 0.2656 | 0.0031 |
| 4 | 256 | 1024 | 4 | 32 | 8 | 16 | 0.2612 | 0.0018 |
| 5 | 256 | 1024 | 4 | 32 | 1 | 449 | 0.2305 | 0.0028 |
| 6 | 256 | 1024 | 4 | 32 | 4 | 128 | 0.2648 | 0.0031 |
| 7 | 256 | 1024 | 4 | 32 | 1 | 512 | 0.2293 | 0.0031 |
| 8 | 256 | 1024 | 4 | 32 | 64 | 8 | 0.2635 | 0.0031 |
| 9 | 256 | 1024 | 4 | 32 | 4 | 32 | 0.2370 | 0.0018 |
| 10 | 256 | 1024 | 4 | 32 | 16 | 32 | 0.2667 | 0.0031 |
| 11 | 256 | 1024 | 4 | 32 | 1 | 131 | 0.2091 | 0.0018 |
| 12 | 256 | 1024 | 4 | 32 | 8 | 32 | 0.2469 | 0.0022 |
| 13 | 256 | 1024 | 4 | 32 | 8 | 64 | 0.2624 | 0.0031 |
| 14 | 256 | 1024 | 4 | 32 | 2 | 256 | 0.2597 | 0.0031 |
| 15 | 256 | 1024 | 4 | 32 | 2 | 128 | 0.2433 | 0.0022 |
| 16 | 256 | 1024 | 4 | 32 | 16 | 4 | 0.2282 | 0.0015 |

```python
import torch

@torch.no_grad()
def run(
    grad_iou_scores: torch.Tensor,
    grad_hyper_weights: torch.Tensor,
    iou_token_out: torch.Tensor,
    mask_tokens_out: torch.Tensor,
    iou_proj_in_weight: torch.Tensor,
    iou_proj_in_bias: torch.Tensor,
    iou_hidden_weight: torch.Tensor,
    iou_hidden_bias: torch.Tensor,
    iou_proj_out_weight: torch.Tensor,
    iou_proj_out_bias: torch.Tensor,
    iou_hidden1: torch.Tensor,
    iou_hidden1_relu: torch.Tensor,
    iou_hidden2: torch.Tensor,
    iou_hidden2_relu: torch.Tensor,
    hyper_proj_in_weights: torch.Tensor,
    hyper_proj_in_biases: torch.Tensor,
    hyper_hidden_weights: torch.Tensor,
    hyper_hidden_biases: torch.Tensor,
    hyper_proj_out_weights: torch.Tensor,
    hyper_proj_out_biases: torch.Tensor,
    hyper_hidden1_0: torch.Tensor,
    hyper_hidden1_1: torch.Tensor,
    hyper_hidden1_2: torch.Tensor,
    hyper_hidden1_3: torch.Tensor,
    hyper_hidden1_relu_0: torch.Tensor,
    hyper_hidden1_relu_1: torch.Tensor,
    hyper_hidden1_relu_2: torch.Tensor,
    hyper_hidden1_relu_3: torch.Tensor,
    hyper_hidden2_0: torch.Tensor,
    hyper_hidden2_1: torch.Tensor,
    hyper_hidden2_2: torch.Tensor,
    hyper_hidden2_3: torch.Tensor,
    hyper_hidden2_relu_0: torch.Tensor,
    hyper_hidden2_relu_1: torch.Tensor,
    hyper_hidden2_relu_2: torch.Tensor,
    hyper_hidden2_relu_3: torch.Tensor,
):
    batch_size, point_batch_size, _ = iou_token_out.shape
    
    hyper_hidden1_list = [hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2, hyper_hidden1_3]
    hyper_hidden1_relu_list = [hyper_hidden1_relu_0, hyper_hidden1_relu_1, hyper_hidden1_relu_2, hyper_hidden1_relu_3]
    hyper_hidden2_list = [hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3]
    hyper_hidden2_relu_list = [hyper_hidden2_relu_0, hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3]
    
    # ==================== IoU Path Backward ====================
    # Gradient through IoU output layer: Linear(1024 -> 4)
    grad_iou_scores_flat = grad_iou_scores.reshape(-1, 4)  # (B*P, 4)
    iou_hidden2_relu_flat = iou_hidden2_relu.reshape(-1, 1024)  # (B*P, 1024)
    
    # Gradient w.r.t. iou_proj_out_weight: (4, 1024)
    grad_iou_proj_out_weight = grad_iou_scores_flat.t() @ iou_hidden2_relu_flat
    
    # Gradient w.r.t. iou_proj_out_bias: (4,)
    grad_iou_proj_out_bias = grad_iou_scores_flat.sum(dim=0)
    
    # Gradient w.r.t. iou_hidden2_relu: (B, P, 1024)
    grad_iou_hidden2_relu = grad_iou_scores_flat @ iou_proj_out_weight  # (B*P, 1024)
    grad_iou_hidden2_relu = grad_iou_hidden2_relu.reshape(batch_size, point_batch_size, 1024)
    
    # Gradient through ReLU
    grad_iou_hidden2 = grad_iou_hidden2_relu * (iou_hidden2 > 0).float()
    
    # Gradient through IoU hidden layer: Linear(1024 -> 1024)
    grad_iou_hidden2_flat = grad_iou_hidden2.reshape(-1, 1024)  # (B*P, 1024)
    iou_hidden1_relu_flat = iou_hidden1_relu.reshape(-1, 1024)  # (B*P, 1024)
    
    # Gradient w.r.t. iou_hidden_weight: (1024, 1024)
    grad_iou_hidden_weight = grad_iou_hidden2_flat.t() @ iou_hidden1_relu_flat
    
    # Gradient w.r.t. iou_hidden_bias: (1024,)
    grad_iou_hidden_bias = grad_iou_hidden2_flat.sum(dim=0)
    
    # Gradient w.r.t. iou_hidden1_relu: (B, P, 1024)
    grad_iou_hidden1_relu = grad_iou_hidden2_flat @ iou_hidden_weight  # (B*P, 1024)
    grad_iou_hidden1_relu = grad_iou_hidden1_relu.reshape(batch_size, point_batch_size, 1024)
    
    # Gradient through ReLU
    grad_iou_hidden1 = grad_iou_hidden1_relu * (iou_hidden1 > 0).float()
    
    # Gradient through IoU input layer: Linear(256 -> 1024)
    grad_iou_hidden1_flat = grad_iou_hidden1.reshape(-1, 1024)  # (B*P, 1024)
    iou_token_out_flat = iou_token_out.reshape(-1, 256)  # (B*P, 256)
    
    # Gradient w.r.t. iou_proj_in_weight: (1024, 256)
    grad_iou_proj_in_weight = grad_iou_hidden1_flat.t() @ iou_token_out_flat
    
    # Gradient w.r.t. iou_proj_in_bias: (1024,)
    grad_iou_proj_in_bias = grad_iou_hidden1_flat.sum(dim=0)
    
    # Gradient w.r.t. iou_token_out: (B, P, 256)
    grad_iou_token_out = grad_iou_hidden1_flat @ iou_proj_in_weight  # (B*P, 256)
    grad_iou_token_out = grad_iou_token_out.reshape(batch_size, point_batch_size, 256)
    
    # ==================== Hypernetwork Path Backward ====================
    grad_mask_tokens_out = torch.zeros_like(mask_tokens_out)
    
    grad_hyper_proj_in_weights = torch.zeros_like(hyper_proj_in_weights)
    grad_hyper_proj_in_biases = torch.zeros_like(hyper_proj_in_biases)
    grad_hyper_hidden_weights = torch.zeros_like(hyper_hidden_weights)
    grad_hyper_hidden_biases = torch.zeros_like(hyper_hidden_biases)
    grad_hyper_proj_out_weights = torch.zeros_like(hyper_proj_out_weights)
    grad_hyper_proj_out_biases = torch.zeros_like(hyper_proj_out_biases)
    
    for mask_idx in range(4):
        # Extract gradient for this mask: (B, P, 32)
        grad_weights = grad_hyper_weights[:, :, mask_idx, :]
        
        # Get saved activations for this mask
        hidden1 = hyper_hidden1_list[mask_idx]
        hidden1_relu = hyper_hidden1_relu_list[mask_idx]
        hidden2 = hyper_hidden2_list[mask_idx]
        hidden2_relu = hyper_hidden2_relu_list[mask_idx]
        token = mask_tokens_out[:, :, mask_idx, :]
        
        # Flatten for batch processing
        grad_weights_flat = grad_weights.reshape(-1, 32)  # (B*P, 32)
        hidden2_relu_flat = hidden2_relu.reshape(-1, 256)  # (B*P, 256)
        hidden1_relu_flat = hidden1_relu.reshape(-1, 256)  # (B*P, 256)
        token_flat = token.reshape(-1, 256)  # (B*P, 256)
        
        # Gradient through output layer: Linear(256 -> 32)
        grad_hyper_proj_out_weights[mask_idx] = grad_weights_flat.t() @ hidden2_relu_flat
        grad_hyper_proj_out_biases[mask_idx] = grad_weights_flat.sum(dim=0)
        
        # Gradient w.r.t. hidden2_relu: (B*P, 256)
        grad_hidden2_relu = grad_weights_flat @ hyper_proj_out_weights[mask_idx]
        grad_hidden2_relu = grad_hidden2_relu.reshape(batch_size, point_batch_size, 256)
        
        # Gradient through ReLU
        grad_hidden2 = grad_hidden2_relu * (hidden2 > 0).float()
        grad_hidden2_flat = grad_hidden2.reshape(-1, 256)
        
        # Gradient through hidden layer: Linear(256 -> 256)
        grad_hyper_hidden_weights[mask_idx] = grad_hidden2_flat.t() @ hidden1_relu_flat
        grad_hyper_hidden_biases[mask_idx] = grad_hidden2_flat.sum(dim=0)
        
        # Gradient w.r.t. hidden1_relu: (B*P, 256)
        grad_hidden1_relu = grad_hidden2_flat @ hyper_hidden_weights[mask_idx]
        grad_hidden1_relu = grad_hidden1_relu.reshape(batch_size, point_batch_size, 256)
        
        # Gradient through ReLU
        grad_hidden1 = grad_hidden1_relu * (hidden1 > 0).float()
        grad_hidden1_flat = grad_hidden1.reshape(-1, 256)
        
        # Gradient through input layer: Linear(256 -> 256)
        grad_hyper_proj_in_weights[mask_idx] = grad_hidden1_flat.t() @ token_flat
        grad_hyper_proj_in_biases[mask_idx] = grad_hidden1_flat.sum(dim=0)
        
        # Gradient w.r.t. token: (B, P, 256)
        grad_token = grad_hidden1_flat @ hyper_proj_in_weights[mask_idx]
        grad_token = grad_token.reshape(batch_size, point_batch_size, 256)
        
        grad_mask_tokens_out[:, :, mask_idx, :] = grad_token
    
    return (
        grad_iou_token_out,
        grad_mask_tokens_out,
        grad_iou_proj_in_weight,
        grad_iou_proj_in_bias,
        grad_iou_hidden_weight,
        grad_iou_hidden_bias,
        grad_iou_proj_out_weight,
        grad_iou_proj_out_bias,
        grad_hyper_proj_in_weights,
        grad_hyper_proj_in_biases,
        grad_hyper_hidden_weights,
        grad_hyper_hidden_biases,
        grad_hyper_proj_out_weights,
        grad_hyper_proj_out_biases,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002326 ms |
| - | Scoring Baseline | 0.500000 | 0.237705 ms |
| - | Reference Implementation | 0.133052 | 1.589237 ms |
