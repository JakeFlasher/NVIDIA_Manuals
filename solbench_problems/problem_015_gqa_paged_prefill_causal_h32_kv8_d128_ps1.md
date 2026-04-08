## Description

Batched Grouped Query Attention prefill with a paged KV cache. Causal mask is applied. Captured from Llama-3.1-8B during incremental prefill.
| Name | Shape | Dtype |
| --- | --- | --- |
| q | [total_q, num_qo_heads, head_dim] | bfloat16 |
| k_cache | [num_pages, page_size, num_kv_heads, head_dim] | bfloat16 |
| v_cache | [num_pages, page_size, num_kv_heads, head_dim] | bfloat16 |
| qo_indptr | [len_indptr] | int32 |
| kv_indptr | [len_indptr] | int32 |
| kv_indices | [num_kv_indices] | int32 |
| sm_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [total_q, num_qo_heads, head_dim] | bfloat16 |
| lse | [total_q, num_qo_heads] | float32 |

| # | num_qo_heads | num_kv_heads | head_dim | page_size | len_indptr | total_q | num_kv_indices | num_pages | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | 1 | 2 | 34 | 1 | 2 | 0.1719 | 0.0005 |
| 2 | 32 | 8 | 128 | 1 | 2 | 1 | 34 | 51 | 0.1857 | 0.0004 |
| 3 | 32 | 8 | 128 | 1 | 29 | 13515 | 28 | 2 | 0.4359 | 0.0294 |
| 4 | 32 | 8 | 128 | 1 | 33 | 12734 | 33 | 86 | 0.5934 | 0.0277 |
| 5 | 32 | 8 | 128 | 1 | 2 | 16 | 3 | 87 | 0.1791 | 0.0004 |
| 6 | 32 | 8 | 128 | 1 | 3 | 78 | 4 | 13719 | 0.3000 | 0.0006 |
| 7 | 32 | 8 | 128 | 1 | 53 | 10749 | 99 | 20585 | 1.2680 | 0.0235 |
| 8 | 32 | 8 | 128 | 1 | 12 | 8981 | 16 | 20585 | 0.4247 | 0.0197 |
| 9 | 32 | 8 | 128 | 1 | 31 | 4803 | 66 | 111721 | 0.8327 | 0.0107 |
| 10 | 32 | 8 | 128 | 1 | 41 | 8140 | 104 | 136181 | 1.2439 | 0.0179 |
| 11 | 32 | 8 | 128 | 1 | 4 | 202 | 7 | 174706 | 0.3199 | 0.0008 |
| 12 | 32 | 8 | 128 | 1 | 44 | 6558 | 101 | 175078 | 1.1834 | 0.0145 |
| 13 | 32 | 8 | 128 | 1 | 13 | 1286 | 35 | 178800 | 0.8305 | 0.0032 |
| 14 | 32 | 8 | 128 | 1 | 2 | 178 | 3 | 59402 | 0.1821 | 0.0008 |
| 15 | 32 | 8 | 128 | 1 | 5 | 70 | 13 | 222053 | 0.5207 | 0.0006 |
| 16 | 32 | 8 | 128 | 1 | 13 | 7009 | 30 | 70149 | 0.6448 | 0.0155 |
| 17 | 32 | 8 | 128 | 1 | 2 | 16384 | 3 | 25634 | 0.2204 | 0.0356 |
| 18 | 32 | 8 | 128 | 1 | 2 | 42 | 3 | 13719 | 0.1786 | 0.0005 |
| 19 | 32 | 8 | 128 | 1 | 2 | 398 | 3 | 14547 | 0.1795 | 0.0013 |
| 20 | 32 | 8 | 128 | 1 | 3 | 14 | 4 | 181871 | 0.2987 | 0.0004 |
| 21 | 32 | 8 | 128 | 1 | 17 | 6102 | 41 | 256226 | 0.8832 | 0.0135 |
| 22 | 32 | 8 | 128 | 1 | 8 | 7419 | 25 | 270692 | 0.4965 | 0.0163 |
| 23 | 32 | 8 | 128 | 1 | 2 | 2 | 3 | 13469 | 0.1811 | 0.0004 |
| 24 | 32 | 8 | 128 | 1 | 5 | 134 | 10 | 383616 | 0.5209 | 0.0007 |
| 25 | 32 | 8 | 128 | 1 | 39 | 7140 | 97 | 399096 | 1.1306 | 0.0158 |
| 26 | 32 | 8 | 128 | 1 | 2 | 10447 | 1 | 2 | 0.1932 | 0.0228 |
| 27 | 32 | 8 | 128 | 1 | 4 | 123 | 12 | 135698 | 0.3074 | 0.0007 |
| 28 | 32 | 8 | 128 | 1 | 27 | 7786 | 74 | 397053 | 0.9149 | 0.0171 |
| 29 | 32 | 8 | 128 | 1 | 2 | 2473 | 2 | 176551 | 0.1812 | 0.0057 |
| 30 | 32 | 8 | 128 | 1 | 21 | 2002 | 67 | 508530 | 1.1980 | 0.0047 |
| 31 | 32 | 8 | 128 | 1 | 2 | 30 | 1 | 2 | 0.1686 | 0.0005 |
| 32 | 32 | 8 | 128 | 1 | 8 | 10393 | 17 | 395233 | 0.3858 | 0.0227 |
| 33 | 32 | 8 | 128 | 1 | 11 | 4950 | 25 | 296392 | 0.6199 | 0.0110 |
| 34 | 32 | 8 | 128 | 1 | 2 | 1171 | 4 | 439455 | 0.1780 | 0.0029 |
| 35 | 32 | 8 | 128 | 1 | 2 | 5 | 2 | 682 | 0.1781 | 0.0004 |
| 36 | 32 | 8 | 128 | 1 | 42 | 8298 | 143 | 605498 | 1.3730 | 0.0183 |
| 37 | 32 | 8 | 128 | 1 | 2 | 17 | 2 | 552310 | 0.1771 | 0.0004 |
| 38 | 32 | 8 | 128 | 1 | 18 | 3478 | 2560 | 716370 | 1.1027 | 0.0092 |

```python
import torch
import math

@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]
    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    # Flatten page dimension since page_size=1
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            # No queries or KV for this batch element
            continue

        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        
        # Number of KV tokens is equal to number of pages for page_size=1
        num_kv_tokens = page_ids.shape[0]
        k_batch = k_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]
        
        # Get queries for this sequence
        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
        num_q_tokens = q_batch.shape[0]

        # Delta for causal masking
        delta = num_kv_tokens - num_q_tokens

        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx

            # Apply causal mask
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue

            q_pos = q_batch[q_idx]  # [num_qo_heads, head_dim]

            for h in range(num_qo_heads):
                # Find corresponding KV head for GQA
                kv_head = h // gqa_ratio

                q_head = q_pos[h]  # [head_dim]
                k_head = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, head_dim]
                v_head = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, head_dim]

                logits = torch.matmul(q_head, k_head.T)  # [max_kv_idx]
                logits_scaled = logits * sm_scale

                # Compute 2-base LSE
                lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                attn = torch.softmax(logits_scaled, dim=-1)  # [max_kv_idx]
                out_head = torch.matmul(attn, v_head)  # [head_dim]
                output[global_q_idx, h] = out_head.to(torch.bfloat16)

    return output, lse
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.003527 ms |
| - | Scoring Baseline | 0.500000 | 0.409814 ms |
| - | Reference Implementation | 0.015407 | 35.484549 ms |
