## Description

Backward pass for region-aware self-attention with edit bias. Computes gradients through QKV projection, scaled dot-product attention with position-based and edit-region biases, softmax, dropout, and output projection.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, seq_len, hidden_size] | float32 |
| hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| edit_region_mask | [batch_size, seq_len] | float32 |
| qkv_weight | [qkv_size, hidden_size] | float32 |
| qkv_bias | [qkv_size] | float32 |
| out_weight | [hidden_size, hidden_size] | float32 |
| out_bias | [hidden_size] | float32 |
| edit_region_bias | [num_attention_heads, max_position_embeddings, max_position_embeddings] | float32 |
| within_edit_bias | [num_attention_heads, 1, 1] | float32 |
| cross_edit_bias | [num_attention_heads, 1, 1] | float32 |
| attention_mask | [batch_size, seq_len] | float32 |
| query | [batch_size, num_attention_heads, seq_len, head_dim] | float32 |
| key | [batch_size, num_attention_heads, seq_len, head_dim] | float32 |
| value | [batch_size, num_attention_heads, seq_len, head_dim] | float32 |
| attention_scores | [batch_size, num_attention_heads, seq_len, seq_len] | float32 |
| attention_probs | [batch_size, num_attention_heads, seq_len, seq_len] | float32 |
| attention_probs_dropped | [batch_size, num_attention_heads, seq_len, seq_len] | float32 |
| attention_output | [batch_size, seq_len, hidden_size] | float32 |
| dropout_mask | [batch_size, num_attention_heads, seq_len, seq_len] | float32 |
| scale | scalar | float32 |
| dropout_p | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_hidden_states | [batch_size, seq_len, hidden_size] | float32 |
| grad_qkv_weight | [qkv_size, hidden_size] | float32 |
| grad_qkv_bias | [qkv_size] | float32 |
| grad_out_weight | [hidden_size, hidden_size] | float32 |
| grad_out_bias | [hidden_size] | float32 |
| grad_edit_region_bias | [num_attention_heads, max_position_embeddings, max_position_embeddings] | float32 |
| grad_within_edit_bias | [num_attention_heads, 1, 1] | float32 |
| grad_cross_edit_bias | [num_attention_heads, 1, 1] | float32 |

| # | hidden_size | num_attention_heads | head_dim | max_position_embeddings | qkv_size | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 16 | 64 | 1024 | 3072 | 64 | 128 | 0.9254 | 0.0810 |
| 2 | 1024 | 16 | 64 | 1024 | 3072 | 1 | 373 | 0.3951 | 0.0124 |
| 3 | 1024 | 16 | 64 | 1024 | 3072 | 32 | 293 | 2.1579 | 0.0997 |
| 4 | 1024 | 16 | 64 | 1024 | 3072 | 4 | 640 | 1.0868 | 0.0315 |
| 5 | 1024 | 16 | 64 | 1024 | 3072 | 4 | 288 | 0.5085 | 0.0144 |
| 6 | 1024 | 16 | 64 | 1024 | 3072 | 8 | 448 | 1.1006 | 0.0409 |
| 7 | 1024 | 16 | 64 | 1024 | 3072 | 64 | 256 | 2.5091 | 0.1711 |
| 8 | 1024 | 16 | 64 | 1024 | 3072 | 16 | 176 | 0.6932 | 0.0287 |
| 9 | 1024 | 16 | 64 | 1024 | 3072 | 1 | 1024 | 0.8038 | 0.0212 |
| 10 | 1024 | 16 | 64 | 1024 | 3072 | 2 | 768 | 0.9344 | 0.0208 |
| 11 | 1024 | 16 | 64 | 1024 | 3072 | 16 | 211 | 0.9183 | 0.0349 |
| 12 | 1024 | 16 | 64 | 1024 | 3072 | 1 | 853 | 0.7710 | 0.0182 |
| 13 | 1024 | 16 | 64 | 1024 | 3072 | 2 | 449 | 0.5890 | 0.0147 |
| 14 | 1024 | 16 | 64 | 1024 | 3072 | 2 | 352 | 0.4640 | 0.0134 |
| 15 | 1024 | 16 | 64 | 1024 | 3072 | 32 | 773 | 9.6977 | 0.3160 |
| 16 | 1024 | 16 | 64 | 1024 | 3072 | 16 | 691 | 4.1769 | 0.1374 |

```python
import torch
import torch.nn.functional as F

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = 1024
    num_attention_heads = 16
    head_dim = 64
    max_position_embeddings = 1024
    qkv_size = 3 * hidden_size
    
    # Gradients from upstream
    grad_output = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    
    # Forward pass inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    
    # Edit region mask - binary mask
    edit_region_mask = (torch.rand(batch_size, seq_len, device=device) > 0.5).float()
    
    # Projection weights
    qkv_weight = torch.randn(qkv_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    qkv_bias = torch.zeros(qkv_size, device=device, dtype=torch.float32)
    out_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    out_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    # Bias parameters
    edit_region_bias = torch.randn(num_attention_heads, max_position_embeddings, max_position_embeddings, device=device, dtype=torch.float32) * 0.02
    within_edit_bias = torch.randn(num_attention_heads, 1, 1, device=device, dtype=torch.float32) * 0.02
    cross_edit_bias = torch.randn(num_attention_heads, 1, 1, device=device, dtype=torch.float32) * 0.02
    
    # Attention mask - 1 for valid positions
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
    
    # Compute forward pass to get saved tensors
    scale = 1.0 / (head_dim ** 0.5)
    dropout_p = 0.1
    
    # QKV projection
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_attention_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    
    # Attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Add biases
    if seq_len <= max_position_embeddings:
        position_bias = edit_region_bias[:, :seq_len, :seq_len]
        attention_scores = attention_scores + position_bias.unsqueeze(0)
    
    edit_mask_q = edit_region_mask.unsqueeze(2)
    edit_mask_k = edit_region_mask.unsqueeze(1)
    within_edit = (edit_mask_q * edit_mask_k).unsqueeze(1)
    attention_scores = attention_scores + within_edit_bias * within_edit
    cross_edit = (edit_mask_q * (1 - edit_mask_k) + (1 - edit_mask_q) * edit_mask_k).unsqueeze(1)
    attention_scores = attention_scores + cross_edit_bias * cross_edit
    
    # Apply attention mask
    attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_scores_masked = attention_scores.masked_fill(attention_mask_expanded == 0, float('-inf'))
    
    # Softmax
    attention_probs = F.softmax(attention_scores_masked, dim=-1, dtype=torch.float32)
    
    # Dropout
    dropout_mask = (torch.rand_like(attention_probs) > dropout_p).float()
    attention_probs_dropped = attention_probs * dropout_mask / (1 - dropout_p)
    
    # Attention output
    attn_out = torch.matmul(attention_probs_dropped, value)
    attn_out = attn_out.transpose(1, 2).contiguous()
    attention_output = attn_out.reshape(batch_size, seq_len, hidden_size)
    
    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "edit_region_mask": edit_region_mask,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "out_weight": out_weight,
        "out_bias": out_bias,
        "edit_region_bias": edit_region_bias,
        "within_edit_bias": within_edit_bias,
        "cross_edit_bias": cross_edit_bias,
        "attention_mask": attention_mask,
        "query": query.contiguous(),
        "key": key.contiguous(),
        "value": value.contiguous(),
        "attention_scores": attention_scores_masked.contiguous(),
        "attention_probs": attention_probs.contiguous(),
        "attention_probs_dropped": attention_probs_dropped.contiguous(),
        "attention_output": attention_output.contiguous(),
        "dropout_mask": dropout_mask.contiguous(),
        "scale": scale,
        "dropout_p": dropout_p,
    }

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    edit_region_mask: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    edit_region_bias: torch.Tensor,
    within_edit_bias: torch.Tensor,
    cross_edit_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_scores: torch.Tensor,
    attention_probs: torch.Tensor,
    attention_probs_dropped: torch.Tensor,
    attention_output: torch.Tensor,
    dropout_mask: torch.Tensor,
    scale: float,
    dropout_p: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 16
    head_dim = 64
    max_position_embeddings = 1024
    
    # Gradient through output projection
    grad_attention_output = torch.matmul(grad_output, out_weight)
    grad_out_weight = torch.matmul(
        grad_output.reshape(-1, hidden_size).t(),
        attention_output.reshape(-1, hidden_size)
    )
    grad_out_bias = grad_output.sum(dim=[0, 1])
    
    # Reshape gradient back to multi-head format
    grad_attention_output = grad_attention_output.reshape(
        batch_size, seq_len, num_attention_heads, head_dim
    )
    grad_attention_output = grad_attention_output.transpose(1, 2)
    
    # Gradient through attention-value multiplication
    grad_attention_probs_dropped = torch.matmul(
        grad_attention_output, value.transpose(-2, -1)
    )
    grad_value = torch.matmul(
        attention_probs_dropped.transpose(-2, -1), grad_attention_output
    )
    
    # Gradient through dropout
    if dropout_p > 0:
        grad_attention_probs = grad_attention_probs_dropped * dropout_mask / (1 - dropout_p)
    else:
        grad_attention_probs = grad_attention_probs_dropped
    
    # Gradient through softmax
    sum_grad = (grad_attention_probs * attention_probs).sum(dim=-1, keepdim=True)
    grad_attention_scores = attention_probs * (grad_attention_probs - sum_grad)
    
    # Handle masked positions
    attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
    grad_attention_scores = grad_attention_scores.masked_fill(
        attention_mask_expanded == 0, 0.0
    )
    
    # Gradient for cross_edit_bias
    edit_mask_q = edit_region_mask.unsqueeze(2)
    edit_mask_k = edit_region_mask.unsqueeze(1)
    cross_edit = (edit_mask_q * (1 - edit_mask_k) + (1 - edit_mask_q) * edit_mask_k).unsqueeze(1)
    grad_cross_edit_bias = (grad_attention_scores * cross_edit).sum(dim=[0, 2, 3], keepdim=True)
    grad_cross_edit_bias = grad_cross_edit_bias.squeeze(0).unsqueeze(-1).unsqueeze(-1)
    grad_cross_edit_bias = grad_cross_edit_bias.reshape(num_attention_heads, 1, 1)
    
    # Gradient for within_edit_bias
    within_edit = (edit_mask_q * edit_mask_k).unsqueeze(1)
    grad_within_edit_bias = (grad_attention_scores * within_edit).sum(dim=[0, 2, 3], keepdim=True)
    grad_within_edit_bias = grad_within_edit_bias.squeeze(0).unsqueeze(-1).unsqueeze(-1)
    grad_within_edit_bias = grad_within_edit_bias.reshape(num_attention_heads, 1, 1)
    
    # Gradient for edit_region_bias
    grad_edit_region_bias = torch.zeros_like(edit_region_bias)
    if seq_len <= max_position_embeddings:
        grad_edit_region_bias[:, :seq_len, :seq_len] = grad_attention_scores.sum(dim=0)
    
    # Gradient through scaled Q @ K^T
    grad_attention_scores_scaled = grad_attention_scores * scale
    grad_query = torch.matmul(grad_attention_scores_scaled, key)
    grad_key = torch.matmul(grad_attention_scores_scaled.transpose(-2, -1), query)
    
    # Combine gradients for Q, K, V
    grad_qkv = torch.stack([grad_query, grad_key, grad_value], dim=0)
    grad_qkv = grad_qkv.permute(1, 3, 0, 2, 4)
    grad_qkv = grad_qkv.reshape(batch_size, seq_len, 3 * hidden_size)
    
    # Gradient through QKV projection
    grad_hidden_states = torch.matmul(grad_qkv, qkv_weight)
    grad_qkv_weight = torch.matmul(
        grad_qkv.reshape(-1, 3 * hidden_size).t(),
        hidden_states.reshape(-1, hidden_size)
    )
    grad_qkv_bias = grad_qkv.sum(dim=[0, 1])
    
    return (
        grad_hidden_states,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_edit_region_bias,
        grad_within_edit_bias,
        grad_cross_edit_bias,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.038346 ms |
| - | Scoring Baseline | 0.500000 | 1.104626 ms |
| - | Reference Implementation | 0.443237 | 1.378685 ms |
