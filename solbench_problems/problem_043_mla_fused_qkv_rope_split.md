## Description

Fused Multi-head Latent Attention (MLA) QKV projection with RoPE split. Combines query low-rank projection (q_a_proj -> RMSNorm -> q_b_proj) with tensor splitting and key-value joint projection with split into a single operation. Eliminates intermediate memory writes for the latent representations.
| Name | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 |
| q_a_proj_weight | [q_lora_rank, hidden_size] | bfloat16 |
| q_a_layernorm_weight | [q_lora_rank] | bfloat16 |
| q_b_proj_weight | [q_proj_out_dim, q_lora_rank] | bfloat16 |
| kv_a_proj_weight | [kv_proj_out_dim, hidden_size] | bfloat16 |
| rms_norm_eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| q_nope | [batch_size, seq_len, num_heads, qk_nope_head_dim] | bfloat16 |
| q_pe | [batch_size, seq_len, num_heads, qk_rope_head_dim] | bfloat16 |
| compressed_kv | [batch_size, seq_len, kv_lora_rank] | bfloat16 |
| k_pe | [batch_size, seq_len, 1, qk_rope_head_dim] | bfloat16 |

| # | hidden_size | num_heads | q_lora_rank | qk_nope_head_dim | qk_rope_head_dim | q_head_dim | kv_lora_rank | kv_proj_out_dim | q_proj_out_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 4 | 373 | 0.2737 | 0.0875 |
| 2 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 1 | 1024 | 0.2103 | 0.0602 |
| 3 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 2 | 4096 | 0.8222 | 0.4788 |
| 4 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 4 | 1024 | 0.4845 | 0.2396 |
| 5 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 32 | 128 | 0.4686 | 0.2396 |
| 6 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 1 | 256 | 0.1613 | 0.0154 |
| 7 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 8 | 128 | 0.2188 | 0.0602 |
| 8 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 1 | 4096 | 0.4688 | 0.2396 |
| 9 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 4 | 512 | 0.3318 | 0.1200 |
| 10 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 8 | 1024 | 0.8132 | 0.4788 |
| 11 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 1 | 131 | 0.1535 | 0.0081 |
| 12 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 1 | 293 | 0.1531 | 0.0175 |
| 13 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 1 | 997 | 0.2177 | 0.0586 |
| 14 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 16 | 128 | 0.3014 | 0.1200 |
| 15 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 2 | 512 | 0.2188 | 0.0602 |
| 16 | 7168 | 128 | 1536 | 128 | 64 | 192 | 512 | 576 | 24576 | 16 | 256 | 0.4680 | 0.2396 |

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """
    Fused MLA QKV projection with RoPE split.
    
    Query path: hidden -> q_a_proj -> RMSNorm -> q_b_proj -> split(q_nope, q_pe)
    KV path: hidden -> kv_a_proj -> split(compressed_kv, k_pe)
    """
    # Constants
    num_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    q_head_dim = 192  # qk_nope_head_dim + qk_rope_head_dim
    kv_lora_rank = 512
    
    bsz, seq_len, _ = hidden_states.shape
    
    # Query pathway: hidden -> q_a_proj -> layernorm -> q_b_proj -> split
    # q_a_proj: Linear(7168, 1536)
    q_latent = F.linear(hidden_states, q_a_proj_weight)  # (bsz, seq_len, 1536)
    
    # RMSNorm on q_latent
    input_dtype = q_latent.dtype
    q_latent_fp32 = q_latent.to(torch.float32)
    variance = q_latent_fp32.pow(2).mean(-1, keepdim=True)
    q_latent_normed = q_latent_fp32 * torch.rsqrt(variance + rms_norm_eps)
    q_latent = (q_a_layernorm_weight * q_latent_normed.to(input_dtype))
    
    # q_b_proj: Linear(1536, 24576)
    q = F.linear(q_latent, q_b_proj_weight)  # (bsz, seq_len, 24576)
    q = q.view(bsz, seq_len, num_heads, q_head_dim)  # (bsz, seq_len, 128, 192)
    
    # Split query into nope and pe components
    q_nope = q[..., :qk_nope_head_dim].contiguous()  # (bsz, seq_len, 128, 128)
    q_pe = q[..., qk_nope_head_dim:].contiguous()  # (bsz, seq_len, 128, 64)
    
    # KV pathway: hidden -> kv_a_proj -> split
    # kv_a_proj: Linear(7168, 576)
    kv_combined = F.linear(hidden_states, kv_a_proj_weight)  # (bsz, seq_len, 576)
    
    # Split into compressed_kv and k_pe
    compressed_kv = kv_combined[..., :kv_lora_rank].contiguous()  # (bsz, seq_len, 512)
    k_pe = kv_combined[..., kv_lora_rank:].contiguous()  # (bsz, seq_len, 64)
    k_pe = k_pe.view(bsz, seq_len, 1, qk_rope_head_dim)  # (bsz, seq_len, 1, 64)
    
    return q_nope, q_pe, compressed_kv, k_pe
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.092050 ms |
| - | Scoring Baseline | 0.500000 | 0.310780 ms |
| - | Reference Implementation | 0.379491 | 0.431321 ms |
