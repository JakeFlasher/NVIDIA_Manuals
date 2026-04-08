## Description

Batched Grouped Query Attention prefill with a paged KV cache. Causal mask is applied. Captured from Qwen3-30B-A3B during incremental prefill.
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
| 1 | 32 | 4 | 128 | 1 | 2 | 1 | 33 | 50 | 0.1469 | 0.0004 |
| 2 | 32 | 4 | 128 | 1 | 29 | 12383 | 1 | 85 | 0.1631 | 0.0270 |
| 3 | 32 | 4 | 128 | 1 | 2 | 16 | 2 | 86 | 0.1474 | 0.0004 |
| 4 | 32 | 4 | 128 | 1 | 4 | 80 | 4 | 15898 | 0.2222 | 0.0006 |
| 5 | 32 | 4 | 128 | 1 | 29 | 15783 | 25 | 62875 | 1.2171 | 0.0343 |
| 6 | 32 | 4 | 128 | 1 | 36 | 9145 | 39 | 114143 | 1.8681 | 0.0200 |
| 7 | 32 | 4 | 128 | 1 | 64 | 13108 | 94 | 139067 | 3.3253 | 0.0285 |
| 8 | 32 | 4 | 128 | 1 | 64 | 8342 | 88 | 181724 | 3.4236 | 0.0183 |
| 9 | 32 | 4 | 128 | 1 | 2 | 178 | 2 | 61699 | 0.1523 | 0.0008 |
| 10 | 32 | 4 | 128 | 1 | 43 | 15910 | 55 | 224323 | 2.2639 | 0.0345 |
| 11 | 32 | 4 | 128 | 1 | 2 | 16384 | 2 | 27915 | 0.1904 | 0.0356 |
| 12 | 32 | 4 | 128 | 1 | 19 | 7938 | 16403 | 291614 | 4.0988 | 0.0218 |
| 13 | 32 | 4 | 128 | 1 | 2 | 42 | 2 | 15898 | 0.1517 | 0.0005 |
| 14 | 32 | 4 | 128 | 1 | 52 | 16189 | 57 | 274149 | 2.1785 | 0.0351 |
| 15 | 32 | 4 | 128 | 1 | 2 | 398 | 2 | 16732 | 0.1501 | 0.0013 |
| 16 | 32 | 4 | 128 | 1 | 3 | 14 | 2 | 184865 | 0.1510 | 0.0004 |
| 17 | 32 | 4 | 128 | 1 | 23 | 5404 | 28 | 224501 | 1.1625 | 0.0120 |
| 18 | 32 | 4 | 128 | 1 | 2 | 2 | 2 | 15615 | 0.1478 | 0.0004 |
| 19 | 32 | 4 | 128 | 1 | 62 | 9823 | 133 | 403995 | 3.4723 | 0.0215 |
| 20 | 32 | 4 | 128 | 1 | 5 | 126 | 11 | 138584 | 0.3711 | 0.0007 |
| 21 | 32 | 4 | 128 | 1 | 27 | 4061 | 47 | 401913 | 1.8352 | 0.0091 |
| 22 | 32 | 4 | 128 | 1 | 64 | 9277 | 117 | 514213 | 3.1470 | 0.0203 |
| 23 | 32 | 4 | 128 | 1 | 9 | 10143 | 37 | 553157 | 0.7058 | 0.0222 |
| 24 | 32 | 4 | 128 | 1 | 28 | 15344 | 48 | 556622 | 1.6708 | 0.0333 |
| 25 | 32 | 4 | 128 | 1 | 2 | 1210 | 1 | 176707 | 0.1362 | 0.0030 |
| 26 | 32 | 4 | 128 | 1 | 2 | 1200 | 3 | 444449 | 0.1522 | 0.0030 |
| 27 | 32 | 4 | 128 | 1 | 64 | 14263 | 142 | 626709 | 3.9610 | 0.0310 |
| 28 | 32 | 4 | 128 | 1 | 2 | 5 | 1 | 683 | 0.1360 | 0.0004 |
| 29 | 32 | 4 | 128 | 1 | 63 | 11763 | 148 | 612942 | 3.9029 | 0.0257 |
| 30 | 32 | 4 | 128 | 1 | 2 | 13 | 2 | 62875 | 0.1519 | 0.0004 |

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
    assert num_kv_heads == 4
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
| - | SOL Bound | 1.000000 | 0.005133 ms |
| - | Scoring Baseline | 0.500000 | 0.612443 ms |
| - | Reference Implementation | 0.016794 | 43.342654 ms |
