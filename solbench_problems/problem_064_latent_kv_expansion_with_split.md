## Description

DeepSeek-V3 latent KV expansion: RMSNorm on compressed KV (kv_lora_rank=512), linear projection to expanded space (num_heads * (qk_nope_head_dim + v_head_dim) = 128*256 = 32768), reshape, transpose, and split into K_nope and V tensors. Fuses normalization, projection, and memory layout transformations.
| Name | Shape | Dtype |
| --- | --- | --- |
| compressed_kv | [batch_size, seq_len, kv_lora_rank] | bfloat16 |
| kv_a_layernorm_weight | [kv_lora_rank] | bfloat16 |
| kv_b_proj_weight | [kv_expanded_dim, kv_lora_rank] | bfloat16 |
| eps | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| k_nope | [batch_size, num_heads, seq_len, qk_nope_head_dim] | bfloat16 |
| value_states | [batch_size, num_heads, seq_len, v_head_dim] | bfloat16 |

| # | kv_lora_rank | num_heads | qk_nope_head_dim | v_head_dim | kv_expanded_dim | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | 128 | 128 | 128 | 32768 | 2 | 293 | 0.1231 | 0.0113 |
| 2 | 512 | 128 | 128 | 128 | 32768 | 32 | 128 | 0.2749 | 0.0763 |
| 3 | 512 | 128 | 128 | 128 | 32768 | 8 | 997 | 0.5013 | 0.1482 |
| 4 | 512 | 128 | 128 | 128 | 32768 | 8 | 512 | 0.3046 | 0.0763 |
| 5 | 512 | 128 | 128 | 128 | 32768 | 1 | 1571 | 0.1675 | 0.0295 |
| 6 | 512 | 128 | 128 | 128 | 32768 | 8 | 128 | 0.1426 | 0.0194 |
| 7 | 512 | 128 | 128 | 128 | 32768 | 1 | 128 | 0.1116 | 0.0059 |
| 8 | 512 | 128 | 128 | 128 | 32768 | 1 | 4096 | 0.2782 | 0.0763 |
| 9 | 512 | 128 | 128 | 128 | 32768 | 1 | 8192 | 0.4571 | 0.1522 |
| 10 | 512 | 128 | 128 | 128 | 32768 | 4 | 256 | 0.1475 | 0.0194 |
| 11 | 512 | 128 | 128 | 128 | 32768 | 1 | 131 | 0.1122 | 0.0059 |
| 12 | 512 | 128 | 128 | 128 | 32768 | 2 | 2048 | 0.3026 | 0.0763 |
| 13 | 512 | 128 | 128 | 128 | 32768 | 1 | 2048 | 0.1888 | 0.0383 |
| 14 | 512 | 128 | 128 | 128 | 32768 | 32 | 256 | 0.5055 | 0.1522 |
| 15 | 512 | 128 | 128 | 128 | 32768 | 1 | 1024 | 0.1360 | 0.0194 |
| 16 | 512 | 128 | 128 | 128 | 32768 | 16 | 512 | 0.5061 | 0.1522 |

```python
import torch

@torch.no_grad()
def run(
    compressed_kv: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    eps: float,
):
    """
    DeepSeek-V3 latent KV expansion with split.
    
    1. RMSNorm on compressed KV
    2. Linear projection to expanded space
    3. Reshape and transpose
    4. Split into K_nope and V
    
    Args:
        compressed_kv: [batch_size, seq_len, kv_lora_rank=512]
        kv_a_layernorm_weight: [kv_lora_rank=512]
        kv_b_proj_weight: [kv_expanded_dim=32768, kv_lora_rank=512]
        eps: RMSNorm epsilon
    
    Returns:
        k_nope: [batch_size, num_heads=128, seq_len, qk_nope_head_dim=128]
        value_states: [batch_size, num_heads=128, seq_len, v_head_dim=128]
    """
    # Constants
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128
    
    bsz, seq_len, _ = compressed_kv.shape
    
    # Step 1: RMSNorm on compressed KV
    # Convert to float32 for numerical stability
    input_dtype = compressed_kv.dtype
    hidden_states = compressed_kv.to(torch.float32)
    
    # Compute variance and normalize
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    
    # Apply weight and convert back to original dtype
    normalized_kv = (kv_a_layernorm_weight.to(torch.float32) * hidden_states).to(input_dtype)
    
    # Step 2: Linear projection
    # [batch, seq_len, 512] @ [512, 32768] -> [batch, seq_len, 32768]
    expanded_kv = torch.matmul(normalized_kv, kv_b_proj_weight.t())
    
    # Step 3: Reshape and transpose
    # [batch, seq_len, 32768] -> [batch, seq_len, 128, 256]
    kv = expanded_kv.view(bsz, seq_len, num_heads, qk_nope_head_dim + v_head_dim)
    
    # [batch, seq_len, 128, 256] -> [batch, 128, seq_len, 256]
    kv = kv.transpose(1, 2)
    
    # Step 4: Split into K_nope and V along the last dimension
    # k_nope: [batch, 128, seq_len, 128]
    # value_states: [batch, 128, seq_len, 128]
    k_nope = kv[:, :, :, :qk_nope_head_dim].contiguous()
    value_states = kv[:, :, :, qk_nope_head_dim:].contiguous()
    
    return k_nope, value_states
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.040735 ms |
| - | Scoring Baseline | 0.500000 | 0.229106 ms |
| - | Reference Implementation | 0.336909 | 0.407339 ms |
