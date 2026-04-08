## Description

Backward pass for cross-attention QKV projection with GQA repeat. Computes gradients for decoder_hidden_states, encoder_hidden_states, q_weight, k_weight, and v_weight given upstream gradients from query_states, key_states, and value_states.
| Name | Shape | Dtype |
| --- | --- | --- |
| grad_query_states | [batch_size, num_heads, seq_len_dec, head_dim] | float32 |
| grad_key_states | [batch_size, num_heads, seq_len_enc, head_dim] | float32 |
| grad_value_states | [batch_size, num_heads, seq_len_enc, head_dim] | float32 |
| decoder_hidden_states | [batch_size, seq_len_dec, hidden_size] | float32 |
| encoder_hidden_states | [batch_size, seq_len_enc, cross_hidden_size] | float32 |
| q_weight | [q_out_dim, hidden_size] | float32 |
| k_weight | [kv_out_dim, cross_hidden_size] | float32 |
| v_weight | [kv_out_dim, cross_hidden_size] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| grad_decoder_hidden_states | [batch_size, seq_len_dec, hidden_size] | float32 |
| grad_encoder_hidden_states | [batch_size, seq_len_enc, cross_hidden_size] | float32 |
| grad_q_weight | [q_out_dim, hidden_size] | float32 |
| grad_k_weight | [kv_out_dim, cross_hidden_size] | float32 |
| grad_v_weight | [kv_out_dim, cross_hidden_size] | float32 |

| # | hidden_size | cross_hidden_size | num_heads | num_key_value_heads | num_key_value_groups | head_dim | q_out_dim | kv_out_dim | batch_size | seq_len_dec | seq_len_enc | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 4 | 2048 | 4096 | 0.7729 | 0.1901 |
| 2 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 1 | 613 | 691 | 0.1623 | 0.0121 |
| 3 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 8 | 128 | 128 | 0.1861 | 0.0194 |
| 4 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 2 | 128 | 128 | 0.1316 | 0.0051 |
| 5 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 16 | 128 | 256 | 0.3161 | 0.0478 |
| 6 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 1 | 1024 | 2048 | 0.2197 | 0.0241 |
| 7 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 64 | 256 | 256 | 1.1175 | 0.3039 |
| 8 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 8 | 919 | 997 | 0.6130 | 0.1395 |
| 9 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 1 | 4096 | 8192 | 0.4585 | 0.0953 |
| 10 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 32 | 449 | 541 | 1.0455 | 0.2802 |
| 11 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 32 | 128 | 128 | 0.3983 | 0.0763 |
| 12 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 16 | 211 | 373 | 0.4133 | 0.0749 |
| 13 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 1 | 256 | 512 | 0.1375 | 0.0063 |
| 14 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 2 | 256 | 256 | 0.1460 | 0.0099 |
| 15 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 16 | 256 | 512 | 0.4726 | 0.0953 |
| 16 | 1536 | 1024 | 16 | 4 | 4 | 256 | 4096 | 1024 | 2 | 512 | 512 | 0.1863 | 0.0194 |

```python
import torch
import math

def gen_inputs(axes_and_scalars, device):
    batch_size = axes_and_scalars['batch_size']
    seq_len_dec = axes_and_scalars['seq_len_dec']
    seq_len_enc = axes_and_scalars['seq_len_enc']
    hidden_size = 1536
    cross_hidden_size = 1024
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 256
    q_out_dim = num_heads * head_dim
    kv_out_dim = num_key_value_heads * head_dim

    # Xavier-scaled weights
    q_weight = torch.randn(q_out_dim, hidden_size, dtype=torch.float32, device=device) / math.sqrt(hidden_size)
    k_weight = torch.randn(kv_out_dim, cross_hidden_size, dtype=torch.float32, device=device) / math.sqrt(cross_hidden_size)
    v_weight = torch.randn(kv_out_dim, cross_hidden_size, dtype=torch.float32, device=device) / math.sqrt(cross_hidden_size)

    # Activation-scale hidden states
    decoder_hidden_states = torch.randn(batch_size, seq_len_dec, hidden_size, dtype=torch.float32, device=device) / math.sqrt(hidden_size)
    encoder_hidden_states = torch.randn(batch_size, seq_len_enc, cross_hidden_size, dtype=torch.float32, device=device) / math.sqrt(cross_hidden_size)

    # Small-magnitude upstream gradients
    grad_query_states = torch.randn(batch_size, num_heads, seq_len_dec, head_dim, dtype=torch.float32, device=device) / math.sqrt(q_out_dim)
    grad_key_states = torch.randn(batch_size, num_heads, seq_len_enc, head_dim, dtype=torch.float32, device=device) / math.sqrt(q_out_dim)
    grad_value_states = torch.randn(batch_size, num_heads, seq_len_enc, head_dim, dtype=torch.float32, device=device) / math.sqrt(q_out_dim)

    return {
        'grad_query_states': grad_query_states,
        'grad_key_states': grad_key_states,
        'grad_value_states': grad_value_states,
        'decoder_hidden_states': decoder_hidden_states,
        'encoder_hidden_states': encoder_hidden_states,
        'q_weight': q_weight,
        'k_weight': k_weight,
        'v_weight': v_weight,
    }

@torch.no_grad()
def run(
    grad_query_states: torch.Tensor,
    grad_key_states: torch.Tensor,
    grad_value_states: torch.Tensor,
    decoder_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
):
    # Constants
    num_heads = 16
    num_key_value_heads = 4
    num_key_value_groups = 4
    head_dim = 256
    
    batch_size = decoder_hidden_states.shape[0]
    seq_len_dec = decoder_hidden_states.shape[1]
    seq_len_enc = encoder_hidden_states.shape[1]
    
    # ========== Query Gradient Computation ==========
    # Reverse transpose: (batch, 16, seq_dec, 256) -> (batch, seq_dec, 16, 256)
    grad_query_proj = grad_query_states.transpose(1, 2)
    
    # Reverse reshape: (batch, seq_dec, 16, 256) -> (batch, seq_dec, 4096)
    grad_query_proj = grad_query_proj.contiguous().view(
        batch_size, seq_len_dec, num_heads * head_dim
    )
    
    # Gradient w.r.t. decoder_hidden_states
    # grad_query_proj: (batch, seq_dec, 4096)
    # q_weight: (4096, 1536)
    # Result: (batch, seq_dec, 1536)
    grad_decoder_hidden_states = torch.matmul(grad_query_proj, q_weight)
    
    # Gradient w.r.t. q_weight
    # decoder_hidden_states: (batch, seq_dec, 1536)
    # grad_query_proj: (batch, seq_dec, 4096)
    # Result: (4096, 1536)
    decoder_flat = decoder_hidden_states.view(-1, decoder_hidden_states.shape[-1])
    grad_query_flat = grad_query_proj.view(-1, grad_query_proj.shape[-1])
    grad_q_weight = torch.matmul(grad_query_flat.t(), decoder_flat)
    
    # ========== Key Gradient Computation ==========
    # Reverse GQA repetition: (batch, 16, seq_enc, 256) -> (batch, 4, seq_enc, 256)
    # Reshape: (batch, 16, seq_enc, 256) -> (batch, 4, 4, seq_enc, 256)
    grad_key_unrepeated = grad_key_states.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len_enc, head_dim
    )
    # Sum over the repetition dimension: (batch, 4, 4, seq_enc, 256) -> (batch, 4, seq_enc, 256)
    grad_key_unrepeated = grad_key_unrepeated.sum(dim=2)
    
    # Reverse transpose: (batch, 4, seq_enc, 256) -> (batch, seq_enc, 4, 256)
    grad_key_proj = grad_key_unrepeated.transpose(1, 2)
    
    # Reverse reshape: (batch, seq_enc, 4, 256) -> (batch, seq_enc, 1024)
    grad_key_proj = grad_key_proj.contiguous().view(
        batch_size, seq_len_enc, num_key_value_heads * head_dim
    )
    
    # Gradient w.r.t. encoder_hidden_states from K
    grad_encoder_hidden_states = torch.matmul(grad_key_proj, k_weight)
    
    # Gradient w.r.t. k_weight
    encoder_flat = encoder_hidden_states.view(-1, encoder_hidden_states.shape[-1])
    grad_key_flat = grad_key_proj.view(-1, grad_key_proj.shape[-1])
    grad_k_weight = torch.matmul(grad_key_flat.t(), encoder_flat)
    
    # ========== Value Gradient Computation ==========
    # Reverse GQA repetition: (batch, 16, seq_enc, 256) -> (batch, 4, seq_enc, 256)
    grad_value_unrepeated = grad_value_states.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len_enc, head_dim
    )
    grad_value_unrepeated = grad_value_unrepeated.sum(dim=2)
    
    # Reverse transpose: (batch, 4, seq_enc, 256) -> (batch, seq_enc, 4, 256)
    grad_value_proj = grad_value_unrepeated.transpose(1, 2)
    
    # Reverse reshape: (batch, seq_enc, 4, 256) -> (batch, seq_enc, 1024)
    grad_value_proj = grad_value_proj.contiguous().view(
        batch_size, seq_len_enc, num_key_value_heads * head_dim
    )
    
    # Gradient w.r.t. encoder_hidden_states from V (accumulate with K gradient)
    grad_from_value = torch.matmul(grad_value_proj, v_weight)
    grad_encoder_hidden_states = grad_encoder_hidden_states + grad_from_value
    
    # Gradient w.r.t. v_weight
    grad_value_flat = grad_value_proj.view(-1, grad_value_proj.shape[-1])
    grad_v_weight = torch.matmul(grad_value_flat.t(), encoder_flat)
    
    return (
        grad_decoder_hidden_states,
        grad_encoder_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight
    )
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.043975 ms |
| - | Scoring Baseline | 0.500000 | 0.330622 ms |
| - | Reference Implementation | 0.452224 | 0.388303 ms |
