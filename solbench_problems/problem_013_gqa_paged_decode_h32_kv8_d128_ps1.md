## Description

Batched Grouped Query Attention decode with a paged KV cache. Captured from Llama-3.1-8B.
| Name | Shape | Dtype |
| --- | --- | --- |
| q | [batch_size, num_qo_heads, head_dim] | bfloat16 |
| k_cache | [num_pages, page_size, num_kv_heads, head_dim] | bfloat16 |
| v_cache | [num_pages, page_size, num_kv_heads, head_dim] | bfloat16 |
| kv_indptr | [len_indptr] | int32 |
| kv_indices | [num_kv_indices] | int32 |
| sm_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, num_qo_heads, head_dim] | bfloat16 |
| lse | [batch_size, num_qo_heads] | float32 |

| # | num_qo_heads | num_kv_heads | head_dim | page_size | batch_size | num_pages | len_indptr | num_kv_indices | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 128 | 1 | 1 | 9316 | 2 | 73 | 0.0710 | 0.0004 |
| 2 | 32 | 8 | 128 | 1 | 1 | 63 | 2 | 46 | 0.0730 | 0.0004 |
| 3 | 32 | 8 | 128 | 1 | 1 | 82 | 2 | 65 | 0.0700 | 0.0004 |
| 4 | 32 | 8 | 128 | 1 | 1 | 18 | 2 | 2 | 0.0697 | 0.0004 |
| 5 | 32 | 8 | 128 | 1 | 1 | 67 | 2 | 50 | 0.0697 | 0.0004 |
| 6 | 32 | 8 | 128 | 1 | 1 | 406 | 2 | 356 | 0.0759 | 0.0006 |
| 7 | 32 | 8 | 128 | 1 | 1 | 597 | 2 | 547 | 0.0756 | 0.0007 |
| 8 | 32 | 8 | 128 | 1 | 1 | 71 | 2 | 54 | 0.0699 | 0.0004 |
| 9 | 32 | 8 | 128 | 1 | 1 | 223 | 2 | 173 | 0.0749 | 0.0005 |
| 10 | 32 | 8 | 128 | 1 | 1 | 74 | 2 | 57 | 0.0706 | 0.0004 |
| 11 | 32 | 8 | 128 | 1 | 1 | 78 | 2 | 61 | 0.0704 | 0.0004 |
| 12 | 32 | 8 | 128 | 1 | 1 | 11 | 2 | 10 | 0.0704 | 0.0004 |
| 13 | 32 | 8 | 128 | 1 | 1 | 15 | 2 | 14 | 0.0704 | 0.0004 |
| 14 | 32 | 8 | 128 | 1 | 1 | 55 | 2 | 38 | 0.0702 | 0.0004 |
| 15 | 32 | 8 | 128 | 1 | 1 | 9341 | 2 | 98 | 0.0707 | 0.0005 |
| 16 | 32 | 8 | 128 | 1 | 1 | 59 | 2 | 42 | 0.0703 | 0.0004 |
| 17 | 32 | 8 | 128 | 1 | 16 | 2708 | 17 | 2681 | 0.1163 | 0.0019 |
| 18 | 32 | 8 | 128 | 1 | 16 | 2212 | 17 | 2185 | 0.1137 | 0.0016 |
| 19 | 32 | 8 | 128 | 1 | 16 | 1220 | 17 | 1193 | 0.1133 | 0.0011 |
| 20 | 32 | 8 | 128 | 1 | 16 | 1732 | 17 | 1705 | 0.1378 | 0.0013 |
| 21 | 32 | 8 | 128 | 1 | 16 | 81390 | 17 | 12942 | 0.1546 | 0.0073 |
| 22 | 32 | 8 | 128 | 1 | 16 | 1069 | 17 | 1034 | 0.1135 | 0.0010 |
| 23 | 32 | 8 | 128 | 1 | 16 | 2868 | 17 | 2841 | 0.1154 | 0.0020 |
| 24 | 32 | 8 | 128 | 1 | 16 | 2372 | 17 | 2345 | 0.1148 | 0.0017 |
| 25 | 32 | 8 | 128 | 1 | 16 | 30163 | 17 | 20911 | 0.1792 | 0.0116 |
| 26 | 32 | 8 | 128 | 1 | 16 | 1892 | 17 | 1865 | 0.1136 | 0.0014 |
| 27 | 32 | 8 | 128 | 1 | 16 | 1396 | 17 | 1369 | 0.1134 | 0.0012 |
| 28 | 32 | 8 | 128 | 1 | 16 | 4333 | 17 | 4298 | 0.1176 | 0.0027 |
| 29 | 32 | 8 | 128 | 1 | 16 | 3044 | 17 | 3017 | 0.1151 | 0.0020 |
| 30 | 32 | 8 | 128 | 1 | 16 | 2548 | 17 | 2521 | 0.1154 | 0.0018 |
| 31 | 32 | 8 | 128 | 1 | 16 | 2052 | 17 | 2025 | 0.1132 | 0.0015 |
| 32 | 32 | 8 | 128 | 1 | 16 | 1556 | 17 | 1529 | 0.1142 | 0.0013 |
| 33 | 32 | 8 | 128 | 1 | 64 | 60071 | 65 | 50902 | 0.3549 | 0.0277 |
| 34 | 32 | 8 | 128 | 1 | 64 | 66637 | 65 | 57366 | 0.3728 | 0.0312 |
| 35 | 32 | 8 | 128 | 1 | 64 | 62605 | 65 | 53334 | 0.3594 | 0.0290 |
| 36 | 32 | 8 | 128 | 1 | 64 | 64653 | 65 | 55382 | 0.3637 | 0.0301 |
| 37 | 32 | 8 | 128 | 1 | 64 | 67021 | 65 | 57750 | 0.3721 | 0.0314 |
| 38 | 32 | 8 | 128 | 1 | 64 | 65037 | 65 | 55766 | 0.3666 | 0.0303 |
| 39 | 32 | 8 | 128 | 1 | 64 | 63053 | 65 | 53782 | 0.3571 | 0.0293 |
| 40 | 32 | 8 | 128 | 1 | 64 | 67405 | 65 | 58134 | 0.3736 | 0.0316 |
| 41 | 32 | 8 | 128 | 1 | 64 | 65421 | 65 | 56150 | 0.3718 | 0.0305 |
| 42 | 32 | 8 | 128 | 1 | 64 | 63437 | 65 | 54166 | 0.3574 | 0.0295 |
| 43 | 32 | 8 | 128 | 1 | 64 | 65805 | 65 | 56534 | 0.3710 | 0.0307 |
| 44 | 32 | 8 | 128 | 1 | 64 | 63821 | 65 | 54550 | 0.3614 | 0.0297 |
| 45 | 32 | 8 | 128 | 1 | 64 | 67853 | 65 | 58582 | 0.3776 | 0.0318 |
| 46 | 32 | 8 | 128 | 1 | 64 | 66253 | 65 | 56982 | 0.3690 | 0.0310 |
| 47 | 32 | 8 | 128 | 1 | 64 | 68237 | 65 | 58966 | 0.3779 | 0.0320 |
| 48 | 32 | 8 | 128 | 1 | 64 | 64205 | 65 | 54934 | 0.3634 | 0.0299 |

```python
import torch
import math

@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    k_cache_flat = k_cache.squeeze(1).to(
        torch.float32
    )  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(
        torch.float32
    )  # [num_pages, num_kv_heads, head_dim]

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_start >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        # Pages are the token indices for page_size=1
        token_indices = kv_indices[page_start:page_end].to(torch.long)
        # Number of tokens is the number of pages for page_size=1
        num_tokens = token_indices.shape[0]

        if num_tokens == 0:
            output[b].zero_()
            continue

        # Get Q, K, V for this batch
        k_batch = k_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

        for h in range(num_qo_heads):
            # Find corresponding KV head for GQA
            kv_head = h // gqa_ratio

            q_head = q_batch[h]  # [head_dim]
            k_head = k_batch[:, kv_head]  # [num_tokens, head_dim]
            v_head = v_batch[:, kv_head]  # [num_tokens, head_dim]

            logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
            logits_scaled = logits * sm_scale

            # Compute 2-base LSE
            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
            out_head = torch.matmul(attn, v_head)  # [head_dim]
            output[b, h] = out_head.to(torch.bfloat16)

    return output, lse
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002951 ms |
| 1st place | AKO4ALL_L2 | 0.677439 | 0.073034 ms |
| - | Scoring Baseline | 0.500000 | 0.147054 ms |
| - | Reference Implementation | 0.008627 | 31.493185 ms |
