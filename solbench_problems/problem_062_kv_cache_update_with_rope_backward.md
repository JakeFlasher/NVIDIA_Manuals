## Description

Backward pass for KV cache update with RoPE. Computes gradients for key_states, value_states, cos, sin, and input caches given gradients from downstream attention operations.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_key_cache | [batch_size, num_kv_heads, max_seq_len, head_dim] | bfloat16 |
| grad_value_cache | [batch_size, num_kv_heads, max_seq_len, head_dim] | bfloat16 |
| key_states | [batch_size, num_kv_heads, new_seq_len, head_dim] | bfloat16 |
| cos | [batch_size, new_seq_len, head_dim] | bfloat16 |
| sin | [batch_size, new_seq_len, head_dim] | bfloat16 |
| cache_position | [new_seq_len] | int64 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_key_states | [batch_size, num_kv_heads, new_seq_len, head_dim] | bfloat16 |
| grad_value_states | [batch_size, num_kv_heads, new_seq_len, head_dim] | bfloat16 |
| grad_cos | [batch_size, new_seq_len, head_dim] | bfloat16 |
| grad_sin | [batch_size, new_seq_len, head_dim] | bfloat16 |
| grad_key_cache_input | [batch_size, num_kv_heads, max_seq_len, head_dim] | bfloat16 |
| grad_value_cache_input | [batch_size, num_kv_heads, max_seq_len, head_dim] | bfloat16 |

| # | num_kv_heads | head_dim | half_dim | batch_size | new_seq_len | max_seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 128 | 64 | 1 | 691 | 2048 | 0.2058 | 0.0036 |
| 2 | 8 | 128 | 64 | 16 | 919 | 8192 | 0.9156 | 0.1620 |
| 3 | 8 | 128 | 64 | 4 | 773 | 4096 | 0.3012 | 0.0224 |
| 4 | 8 | 128 | 64 | 2 | 211 | 1024 | 0.2084 | 0.0032 |
| 5 | 8 | 128 | 64 | 4 | 256 | 2048 | 0.2128 | 0.0107 |
| 6 | 8 | 128 | 64 | 8 | 256 | 2048 | 0.2577 | 0.0209 |
| 7 | 8 | 128 | 64 | 2 | 541 | 2048 | 0.2106 | 0.0064 |
| 8 | 8 | 128 | 64 | 1 | 1 | 256 | 0.2046 | 0.0007 |
| 9 | 8 | 128 | 64 | 2 | 2048 | 4096 | 0.3345 | 0.0152 |
| 10 | 8 | 128 | 64 | 32 | 1 | 2048 | 0.2887 | 0.0705 |
| 11 | 8 | 128 | 64 | 8 | 449 | 4096 | 0.3434 | 0.0407 |
| 12 | 8 | 128 | 64 | 4 | 1 | 512 | 0.2073 | 0.0026 |
| 13 | 8 | 128 | 64 | 32 | 512 | 4096 | 0.9826 | 0.1645 |
| 14 | 8 | 128 | 64 | 16 | 613 | 4096 | 0.6302 | 0.0848 |
| 15 | 8 | 128 | 64 | 32 | 2048 | 8192 | 3.1355 | 0.3768 |
| 16 | 8 | 128 | 64 | 1 | 293 | 2048 | 0.2082 | 0.0030 |

```python
import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars['batch_size']
    num_kv_heads = axes_and_scalars['num_kv_heads']
    new_seq_len = axes_and_scalars['new_seq_len']
    head_dim = axes_and_scalars['head_dim']
    max_seq_len = axes_and_scalars['max_seq_len']
    
    grad_key_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    grad_value_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key_states = torch.randn(batch_size, num_kv_heads, new_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    cos = torch.randn(batch_size, new_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(batch_size, new_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    cache_position = torch.arange(new_seq_len, dtype=torch.int64, device=device)
    
    return {
        'grad_key_cache': grad_key_cache,
        'grad_value_cache': grad_value_cache,
        'key_states': key_states,
        'cos': cos,
        'sin': sin,
        'cache_position': cache_position,
    }

@torch.no_grad()
def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    half_dim = key_states.shape[-1] // 2
    
    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    k_rotated_half = torch.cat((-k2, k1), dim=-1)
    
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    grad_key_states_rotated = grad_key_cache[:, :, cache_position]
    grad_value_states = grad_value_cache[:, :, cache_position]
    
    grad_from_cos_term = grad_key_states_rotated * cos_expanded
    grad_k_rotated_half = grad_key_states_rotated * sin_expanded
    
    grad_k_rotated_half_1 = grad_k_rotated_half[..., :half_dim]
    grad_k_rotated_half_2 = grad_k_rotated_half[..., half_dim:]
    
    grad_k2_from_rotate = -grad_k_rotated_half_1
    grad_k1_from_rotate = grad_k_rotated_half_2
    
    grad_k1_total = grad_from_cos_term[..., :half_dim] + grad_k1_from_rotate
    grad_k2_total = grad_from_cos_term[..., half_dim:] + grad_k2_from_rotate
    
    grad_key_states = torch.cat([grad_k1_total, grad_k2_total], dim=-1)
    
    grad_cos_expanded = grad_key_states_rotated * key_states
    grad_cos = grad_cos_expanded.sum(dim=1)
    
    grad_sin_expanded = grad_key_states_rotated * k_rotated_half
    grad_sin = grad_sin_expanded.sum(dim=1)
    
    grad_key_cache_input = grad_key_cache.clone()
    grad_key_cache_input[:, :, cache_position] = 0
    
    grad_value_cache_input = grad_value_cache.clone()
    grad_value_cache_input[:, :, cache_position] = 0
    
    return (
        grad_key_states.to(torch.bfloat16),
        grad_value_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_key_cache_input.to(torch.bfloat16),
        grad_value_cache_input.to(torch.bfloat16),
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.017464 ms |
| - | Scoring Baseline | 0.500000 | 0.359432 ms |
| - | Reference Implementation | 0.480597 | 0.416817 ms |
