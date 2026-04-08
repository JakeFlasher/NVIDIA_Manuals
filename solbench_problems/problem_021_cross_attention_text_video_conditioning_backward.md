## Description

Backward pass for cross-attention text-video conditioning. Computes gradients through output projection, attention mechanism (softmax, Q@K^T), and Q/K/V linear projections. Queries from video latents, keys/values from text embeddings.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_output | [batch_size, num_video_tokens, hidden_size] | float32 |
| video_latents | [batch_size, num_video_tokens, hidden_size] | float32 |
| text_embeddings | [batch_size, num_text_tokens, hidden_size] | float32 |
| query_weight | [hidden_size, hidden_size] | float32 |
| query_bias | [hidden_size] | float32 |
| key_weight | [hidden_size, hidden_size] | float32 |
| key_bias | [hidden_size] | float32 |
| value_weight | [hidden_size, hidden_size] | float32 |
| value_bias | [hidden_size] | float32 |
| output_weight | [hidden_size, hidden_size] | float32 |
| output_bias | [hidden_size] | float32 |
| scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_video_latents | [batch_size, num_video_tokens, hidden_size] | float32 |
| grad_text_embeddings | [batch_size, num_text_tokens, hidden_size] | float32 |
| grad_query_weight | [hidden_size, hidden_size] | float32 |
| grad_query_bias | [hidden_size] | float32 |
| grad_key_weight | [hidden_size, hidden_size] | float32 |
| grad_key_bias | [hidden_size] | float32 |
| grad_value_weight | [hidden_size, hidden_size] | float32 |
| grad_value_bias | [hidden_size] | float32 |
| grad_output_weight | [hidden_size, hidden_size] | float32 |
| grad_output_bias | [hidden_size] | float32 |

| # | hidden_size | num_heads | head_dim | batch_size | num_video_tokens | num_text_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1024 | 16 | 64 | 64 | 128 | 77 | 0.5101 | 0.0560 |
| 2 | 1024 | 16 | 64 | 8 | 512 | 77 | 0.3053 | 0.0196 |
| 3 | 1024 | 16 | 64 | 2 | 512 | 77 | 0.2233 | 0.0052 |
| 4 | 1024 | 16 | 64 | 2 | 691 | 77 | 0.2553 | 0.0066 |
| 5 | 1024 | 16 | 64 | 2 | 2048 | 77 | 0.3865 | 0.0175 |
| 6 | 1024 | 16 | 64 | 8 | 256 | 373 | 0.3370 | 0.0265 |
| 7 | 1024 | 16 | 64 | 1 | 4096 | 77 | 0.4768 | 0.0171 |
| 8 | 1024 | 16 | 64 | 4 | 512 | 77 | 0.2568 | 0.0100 |
| 9 | 1024 | 16 | 64 | 1 | 512 | 77 | 0.2210 | 0.0028 |
| 10 | 1024 | 16 | 64 | 2 | 1024 | 77 | 0.2792 | 0.0093 |
| 11 | 1024 | 16 | 64 | 2 | 4096 | 77 | 0.5772 | 0.0339 |
| 12 | 1024 | 16 | 64 | 4 | 256 | 293 | 0.2624 | 0.0114 |
| 13 | 1024 | 16 | 64 | 2 | 256 | 77 | 0.2168 | 0.0032 |
| 14 | 1024 | 16 | 64 | 2 | 256 | 211 | 0.2206 | 0.0049 |
| 15 | 1024 | 16 | 64 | 4 | 853 | 77 | 0.3179 | 0.0155 |
| 16 | 1024 | 16 | 64 | 32 | 128 | 77 | 0.3563 | 0.0282 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    video_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    scale: float,
):
    """
    Backward pass for cross-attention text-video conditioning.
    
    Computes gradients through:
    1. Output projection
    2. Attention-value multiplication
    3. Softmax
    4. Attention score computation (Q @ K^T)
    5. Q, K, V linear projections
    """
    batch_size, num_video_tokens, hidden_size = video_latents.shape
    num_text_tokens = text_embeddings.shape[1]
    num_heads = 16
    head_dim = 64
    
    # Recompute forward pass intermediates
    # Project queries from video latents
    queries = F.linear(video_latents, query_weight, query_bias)  # [B, N_v, D]
    queries = queries.view(batch_size, num_video_tokens, num_heads, head_dim)
    queries = queries.transpose(1, 2)  # [B, H, N_v, d]
    
    # Project keys and values from text embeddings
    keys = F.linear(text_embeddings, key_weight, key_bias)  # [B, N_t, D]
    keys = keys.view(batch_size, num_text_tokens, num_heads, head_dim)
    keys = keys.transpose(1, 2)  # [B, H, N_t, d]
    
    values = F.linear(text_embeddings, value_weight, value_bias)  # [B, N_t, D]
    values = values.view(batch_size, num_text_tokens, num_heads, head_dim)
    values = values.transpose(1, 2)  # [B, H, N_t, d]
    
    # Compute attention scores
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # [B, H, N_v, N_t]
    
    # Compute attention probabilities
    attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)  # [B, H, N_v, N_t]
    
    # Apply attention to values
    context = torch.matmul(attention_probs, values)  # [B, H, N_v, d]
    
    # Reshape back to [B, N_v, D]
    context = context.transpose(1, 2).contiguous()  # [B, N_v, H, d]
    context = context.view(batch_size, num_video_tokens, hidden_size)  # [B, N_v, D]
    
    # ========================================
    # Backward through output projection
    # ========================================
    grad_context = torch.matmul(grad_output, output_weight)  # [B, N_v, D]
    
    # grad_output_weight = grad_output^T @ context (summed over batch)
    grad_output_weight = torch.einsum('bnd,bnk->dk', grad_output, context)  # [D, D]
    grad_output_bias = grad_output.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Backward through reshape from attention
    # ========================================
    grad_context_heads = grad_context.view(batch_size, num_video_tokens, num_heads, head_dim)
    grad_context_heads = grad_context_heads.transpose(1, 2)  # [B, H, N_v, d]
    
    # ========================================
    # Backward through attention @ values
    # ========================================
    grad_attention_probs = torch.matmul(grad_context_heads, values.transpose(-2, -1))  # [B, H, N_v, N_t]
    grad_values = torch.matmul(attention_probs.transpose(-2, -1), grad_context_heads)  # [B, H, N_t, d]
    
    # ========================================
    # Backward through softmax
    # ========================================
    sum_grad_probs = (grad_attention_probs * attention_probs).sum(dim=-1, keepdim=True)  # [B, H, N_v, 1]
    grad_attention_scores = attention_probs * (grad_attention_probs - sum_grad_probs)  # [B, H, N_v, N_t]
    
    # ========================================
    # Backward through scaling
    # ========================================
    grad_attention_scores = grad_attention_scores * scale
    
    # ========================================
    # Backward through Q @ K^T
    # ========================================
    grad_queries = torch.matmul(grad_attention_scores, keys)  # [B, H, N_v, d]
    grad_keys = torch.matmul(grad_attention_scores.transpose(-2, -1), queries)  # [B, H, N_t, d]
    
    # ========================================
    # Backward through reshape for Q, K, V
    # ========================================
    grad_queries = grad_queries.transpose(1, 2).contiguous()  # [B, N_v, H, d]
    grad_queries = grad_queries.view(batch_size, num_video_tokens, hidden_size)  # [B, N_v, D]
    
    grad_keys = grad_keys.transpose(1, 2).contiguous()  # [B, N_t, H, d]
    grad_keys = grad_keys.view(batch_size, num_text_tokens, hidden_size)  # [B, N_t, D]
    
    grad_values = grad_values.transpose(1, 2).contiguous()  # [B, N_t, H, d]
    grad_values = grad_values.view(batch_size, num_text_tokens, hidden_size)  # [B, N_t, D]
    
    # ========================================
    # Backward through Q projection
    # ========================================
    grad_video_latents = torch.matmul(grad_queries, query_weight)  # [B, N_v, D]
    grad_query_weight = torch.einsum('bnd,bnk->dk', grad_queries, video_latents)  # [D, D]
    grad_query_bias = grad_queries.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Backward through K projection
    # ========================================
    grad_text_from_keys = torch.matmul(grad_keys, key_weight)  # [B, N_t, D]
    grad_key_weight = torch.einsum('bnd,bnk->dk', grad_keys, text_embeddings)  # [D, D]
    grad_key_bias = grad_keys.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Backward through V projection
    # ========================================
    grad_text_from_values = torch.matmul(grad_values, value_weight)  # [B, N_t, D]
    grad_value_weight = torch.einsum('bnd,bnk->dk', grad_values, text_embeddings)  # [D, D]
    grad_value_bias = grad_values.sum(dim=(0, 1))  # [D]
    
    # ========================================
    # Accumulate gradients for text_embeddings
    # ========================================
    grad_text_embeddings = grad_text_from_keys + grad_text_from_values
    
    return (
        grad_video_latents,
        grad_text_embeddings,
        grad_query_weight,
        grad_query_bias,
        grad_key_weight,
        grad_key_bias,
        grad_value_weight,
        grad_value_bias,
        grad_output_weight,
        grad_output_bias,
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.012082 ms |
| - | Scoring Baseline | 0.500000 | 0.309697 ms |
| - | Reference Implementation | 0.243751 | 0.945868 ms |
