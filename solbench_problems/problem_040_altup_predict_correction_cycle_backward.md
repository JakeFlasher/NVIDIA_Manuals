## Description

Backward pass for AltUp predict-correct cycle. Computes gradients for prediction coefficients, correction coefficients, router weights, and norm weights through the combined predict and correct operations.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_corrected | [altup_num_inputs, batch_size, seq_len, hidden_size] | bfloat16 |
| hidden_states | [altup_num_inputs, batch_size, seq_len, hidden_size] | bfloat16 |
| activated | [batch_size, seq_len, hidden_size] | bfloat16 |
| prediction_coef_weight | [altup_num_inputs_sq, altup_num_inputs] | bfloat16 |
| correction_coef_weight | [altup_num_inputs, altup_num_inputs] | bfloat16 |
| router_weight | [altup_num_inputs, hidden_size] | bfloat16 |
| norm_weight | [hidden_size] | bfloat16 |
| altup_active_idx | scalar | int32 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [altup_num_inputs, batch_size, seq_len, hidden_size] | bfloat16 |
| grad_activated | [batch_size, seq_len, hidden_size] | bfloat16 |
| grad_prediction_coef_weight | [altup_num_inputs_sq, altup_num_inputs] | float32 |
| grad_correction_coef_weight | [altup_num_inputs, altup_num_inputs] | float32 |
| grad_router_weight | [altup_num_inputs, hidden_size] | float32 |
| grad_norm_weight | [hidden_size] | float32 |

| # | hidden_size | altup_num_inputs | altup_num_inputs_sq | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2304 | 3 | 9 | 64 | 256 | 2.1496 | 0.1777 |
| 2 | 2304 | 3 | 9 | 1 | 1024 | 0.3518 | 0.0115 |
| 3 | 2304 | 3 | 9 | 4 | 293 | 0.3955 | 0.0131 |
| 4 | 2304 | 3 | 9 | 1 | 256 | 0.2297 | 0.0032 |
| 5 | 2304 | 3 | 9 | 8 | 373 | 0.7280 | 0.0327 |
| 6 | 2304 | 3 | 9 | 1 | 512 | 0.2916 | 0.0060 |
| 7 | 2304 | 3 | 9 | 4 | 256 | 0.4554 | 0.0115 |
| 8 | 2304 | 3 | 9 | 8 | 256 | 0.5345 | 0.0226 |
| 9 | 2304 | 3 | 9 | 16 | 512 | 1.6037 | 0.0890 |
| 10 | 2304 | 3 | 9 | 16 | 256 | 1.0733 | 0.0447 |
| 11 | 2304 | 3 | 9 | 4 | 1024 | 1.0633 | 0.0447 |
| 12 | 2304 | 3 | 9 | 8 | 512 | 1.0605 | 0.0447 |
| 13 | 2304 | 3 | 9 | 64 | 613 | 5.2872 | 0.4248 |
| 14 | 2304 | 3 | 9 | 16 | 449 | 1.5354 | 0.0781 |
| 15 | 2304 | 3 | 9 | 2 | 512 | 0.3533 | 0.0115 |
| 16 | 2304 | 3 | 9 | 64 | 128 | 1.6036 | 0.0890 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_corrected: torch.Tensor,
    hidden_states: torch.Tensor,
    activated: torch.Tensor,
    prediction_coef_weight: torch.Tensor,
    correction_coef_weight: torch.Tensor,
    router_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    altup_active_idx: int,
    rms_norm_eps: float,
):
    """
    Backward pass for AltUp predict-correct cycle.
    
    This computes gradients through:
    1. Correct step backward
    2. Predict step backward
    
    Returns gradients for all learnable parameters and inputs.
    """
    altup_num_inputs = 3
    hidden_size = 2304
    router_scale = hidden_size ** -1.0
    
    batch_size = hidden_states.shape[1]
    seq_len = hidden_states.shape[2]
    
    # ========== FORWARD RECOMPUTATION FOR PREDICT STEP ==========
    # We need intermediate values from forward pass
    
    # Predict step forward recomputation
    active_input_predict = hidden_states[altup_active_idx]
    x_float_predict = active_input_predict.float()
    variance_predict = x_float_predict.pow(2).mean(-1, keepdim=True)
    rstd_predict = torch.rsqrt(variance_predict + rms_norm_eps)
    normalized_predict = x_float_predict * rstd_predict
    normed_predict = normalized_predict * norm_weight.float()
    scaled_predict = normed_predict * router_scale
    routed_predict = F.linear(scaled_predict, router_weight.float())
    modalities_predict = torch.tanh(routed_predict)
    
    all_coefs_flat = F.linear(modalities_predict, prediction_coef_weight.float())
    all_coefs = all_coefs_flat.reshape(batch_size, seq_len, altup_num_inputs, altup_num_inputs)
    all_coefs = all_coefs.permute(0, 1, 3, 2)
    
    h_permuted = hidden_states.float().permute(1, 2, 3, 0)
    predictions_before_residual = torch.matmul(h_permuted, all_coefs)
    predictions_permuted = predictions_before_residual.permute(3, 0, 1, 2)
    predictions = predictions_permuted + hidden_states.float()
    
    # ========== FORWARD RECOMPUTATION FOR CORRECT STEP ==========
    x_float_correct = activated.float()
    variance_correct = x_float_correct.pow(2).mean(-1, keepdim=True)
    rstd_correct = torch.rsqrt(variance_correct + rms_norm_eps)
    normalized_correct = x_float_correct * rstd_correct
    normed_correct = normalized_correct * norm_weight.float()
    scaled_correct = normed_correct * router_scale
    routed_correct = F.linear(scaled_correct, router_weight.float())
    modalities_correct = torch.tanh(routed_correct)
    
    innovation = activated.float() - predictions[altup_active_idx]
    innovation_repeated = innovation.unsqueeze(0).expand(altup_num_inputs, -1, -1, -1)
    
    all_coefs_correct = F.linear(modalities_correct, correction_coef_weight.float()) + 1.0
    all_coefs_expanded = all_coefs_correct.permute(2, 0, 1).unsqueeze(-1)
    
    # ========== BACKWARD PASS FOR CORRECT STEP ==========
    grad_corrected_float = grad_corrected.float()
    
    # Gradient through residual connection
    grad_predictions = grad_corrected_float.clone()
    
    # Gradient through element-wise multiplication
    # corrected = innovation_repeated * all_coefs_expanded + predictions
    grad_innovation_repeated = grad_corrected_float * all_coefs_expanded
    grad_all_coefs_expanded = (grad_corrected_float * innovation_repeated).sum(dim=-1, keepdim=True)
    
    # Gradient through unsqueeze and permute of all_coefs
    grad_all_coefs_correct = grad_all_coefs_expanded.squeeze(-1).permute(1, 2, 0)
    
    # Gradient through +1.0 (identity)
    # Gradient through linear projection for correction coefficients
    grad_correction_coef_weight = torch.matmul(
        grad_all_coefs_correct.reshape(-1, altup_num_inputs).t(),
        modalities_correct.reshape(-1, altup_num_inputs)
    )
    
    grad_modalities_correct = torch.matmul(
        grad_all_coefs_correct,
        correction_coef_weight.float()
    )
    
    # Gradient through repeat of innovation (sum over repeated dimension)
    grad_innovation = grad_innovation_repeated.sum(dim=0)
    
    # Gradient through innovation computation
    # innovation = activated - predictions[altup_active_idx]
    grad_activated = grad_innovation.clone()
    grad_predictions[altup_active_idx] = grad_predictions[altup_active_idx] - grad_innovation
    
    # Gradient through router modalities for correct step
    tanh_modalities_correct = modalities_correct
    grad_routed_correct = grad_modalities_correct * (1 - tanh_modalities_correct.pow(2))
    
    grad_router_weight_correct = torch.matmul(
        grad_routed_correct.reshape(-1, altup_num_inputs).t(),
        scaled_correct.reshape(-1, hidden_size)
    )
    
    grad_scaled_correct = torch.matmul(grad_routed_correct, router_weight.float())
    grad_normed_correct = grad_scaled_correct * router_scale
    
    grad_norm_weight_correct = (grad_normed_correct * normalized_correct).sum(dim=[0, 1])
    
    grad_normalized_correct = grad_normed_correct * norm_weight.float()
    mean_grad_normalized_x_correct = (grad_normalized_correct * x_float_correct).mean(-1, keepdim=True)
    grad_activated_from_router = grad_normalized_correct * rstd_correct - x_float_correct * rstd_correct.pow(3) * mean_grad_normalized_x_correct
    grad_activated = grad_activated + grad_activated_from_router
    
    # ========== BACKWARD PASS FOR PREDICT STEP ==========
    # Gradient through residual connection
    grad_hidden_states = grad_predictions.clone()
    
    # Gradient through permute back
    grad_predictions_permuted = grad_predictions.permute(1, 2, 3, 0)
    
    # Gradient through matmul
    # predictions = h_permuted @ all_coefs
    grad_h_permuted = torch.matmul(
        grad_predictions_permuted,
        all_coefs.transpose(-2, -1)
    )
    
    grad_all_coefs_matmul = torch.matmul(
        h_permuted.transpose(-2, -1),
        grad_predictions_permuted
    )
    
    # Gradient through permute of hidden_states
    grad_hidden_states = grad_hidden_states + grad_h_permuted.permute(3, 0, 1, 2)
    
    # Gradient through transpose of all_coefs
    grad_all_coefs_before_transpose = grad_all_coefs_matmul.permute(0, 1, 3, 2)
    
    # Gradient through reshape
    grad_all_coefs_flat = grad_all_coefs_before_transpose.reshape(
        batch_size, seq_len, altup_num_inputs * altup_num_inputs
    )
    
    # Gradient through linear projection for prediction coefficients
    grad_prediction_coef_weight = torch.matmul(
        grad_all_coefs_flat.reshape(-1, altup_num_inputs * altup_num_inputs).t(),
        modalities_predict.reshape(-1, altup_num_inputs)
    )
    
    grad_modalities_predict = torch.matmul(
        grad_all_coefs_flat,
        prediction_coef_weight.float()
    )
    
    # Gradient through router modalities for predict step
    tanh_modalities_predict = modalities_predict
    grad_routed_predict = grad_modalities_predict * (1 - tanh_modalities_predict.pow(2))
    
    grad_router_weight_predict = torch.matmul(
        grad_routed_predict.reshape(-1, altup_num_inputs).t(),
        scaled_predict.reshape(-1, hidden_size)
    )
    
    grad_scaled_predict = torch.matmul(grad_routed_predict, router_weight.float())
    grad_normed_predict = grad_scaled_predict * router_scale
    
    grad_norm_weight_predict = (grad_normed_predict * normalized_predict).sum(dim=[0, 1])
    
    grad_normalized_predict = grad_normed_predict * norm_weight.float()
    mean_grad_normalized_x_predict = (grad_normalized_predict * x_float_predict).mean(-1, keepdim=True)
    grad_active_input = grad_normalized_predict * rstd_predict - x_float_predict * rstd_predict.pow(3) * mean_grad_normalized_x_predict
    
    grad_hidden_states[altup_active_idx] = grad_hidden_states[altup_active_idx] + grad_active_input
    
    # Combine gradients from predict and correct steps
    grad_router_weight = grad_router_weight_predict + grad_router_weight_correct
    grad_norm_weight = grad_norm_weight_predict + grad_norm_weight_correct
    
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_activated.to(torch.bfloat16),
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.031950 ms |
| - | Scoring Baseline | 0.500000 | 0.805953 ms |
| - | Reference Implementation | 0.189382 | 3.360290 ms |
