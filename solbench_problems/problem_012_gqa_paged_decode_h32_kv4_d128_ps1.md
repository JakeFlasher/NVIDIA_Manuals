## Description

Batched Grouped Query Attention decode with a paged KV cache. Captured from Qwen3-30B-A3B.
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
| 1 | 32 | 4 | 128 | 1 | 1 | 8 | 2 | 7 | 0.1186 | 0.0004 |
| 2 | 32 | 4 | 128 | 1 | 1 | 412 | 2 | 362 | 0.1003 | 0.0005 |
| 3 | 32 | 4 | 128 | 1 | 1 | 10 | 2 | 9 | 0.0931 | 0.0004 |
| 4 | 32 | 4 | 128 | 1 | 1 | 191 | 2 | 141 | 0.0977 | 0.0004 |
| 5 | 32 | 4 | 128 | 1 | 1 | 12 | 2 | 11 | 0.0930 | 0.0004 |
| 6 | 32 | 4 | 128 | 1 | 1 | 486 | 2 | 436 | 0.0984 | 0.0005 |
| 7 | 32 | 4 | 128 | 1 | 1 | 15 | 2 | 14 | 0.0934 | 0.0004 |
| 8 | 32 | 4 | 128 | 1 | 1 | 17 | 2 | 2 | 0.0926 | 0.0004 |
| 9 | 32 | 4 | 128 | 1 | 1 | 81 | 2 | 64 | 0.0926 | 0.0004 |
| 10 | 32 | 4 | 128 | 1 | 1 | 9347 | 2 | 102 | 0.0933 | 0.0004 |
| 11 | 32 | 4 | 128 | 1 | 1 | 9317 | 2 | 72 | 0.0935 | 0.0004 |
| 12 | 32 | 4 | 128 | 1 | 1 | 57 | 2 | 40 | 0.0932 | 0.0004 |
| 13 | 32 | 4 | 128 | 1 | 1 | 67 | 2 | 50 | 0.0936 | 0.0004 |
| 14 | 32 | 4 | 128 | 1 | 1 | 9332 | 2 | 87 | 0.0933 | 0.0004 |
| 15 | 32 | 4 | 128 | 1 | 1 | 302 | 2 | 252 | 0.0973 | 0.0005 |
| 16 | 32 | 4 | 128 | 1 | 1 | 596 | 2 | 546 | 0.0985 | 0.0005 |
| 17 | 32 | 4 | 128 | 1 | 64 | 28831 | 65 | 28815 | 0.2595 | 0.0082 |
| 18 | 32 | 4 | 128 | 1 | 64 | 31007 | 65 | 30991 | 0.2663 | 0.0088 |
| 19 | 32 | 4 | 128 | 1 | 64 | 33183 | 65 | 33167 | 0.2724 | 0.0094 |
| 20 | 32 | 4 | 128 | 1 | 64 | 35359 | 65 | 35343 | 0.2779 | 0.0100 |
| 21 | 32 | 4 | 128 | 1 | 64 | 37535 | 65 | 37519 | 0.2830 | 0.0106 |
| 22 | 32 | 4 | 128 | 1 | 64 | 39711 | 65 | 39695 | 0.2869 | 0.0111 |
| 23 | 32 | 4 | 128 | 1 | 64 | 41887 | 65 | 41871 | 0.3036 | 0.0117 |
| 24 | 32 | 4 | 128 | 1 | 64 | 44063 | 65 | 44047 | 0.3029 | 0.0123 |
| 25 | 32 | 4 | 128 | 1 | 64 | 46303 | 65 | 46287 | 0.3025 | 0.0129 |
| 26 | 32 | 4 | 128 | 1 | 64 | 48479 | 65 | 48463 | 0.3096 | 0.0135 |
| 27 | 32 | 4 | 128 | 1 | 64 | 50655 | 65 | 50639 | 0.3151 | 0.0141 |
| 28 | 32 | 4 | 128 | 1 | 64 | 52831 | 65 | 52815 | 0.3227 | 0.0146 |
| 29 | 32 | 4 | 128 | 1 | 64 | 55007 | 65 | 54991 | 0.3267 | 0.0152 |
| 30 | 32 | 4 | 128 | 1 | 64 | 57183 | 65 | 57167 | 0.3322 | 0.0158 |
| 31 | 32 | 4 | 128 | 1 | 64 | 59359 | 65 | 59343 | 0.3350 | 0.0164 |
| 32 | 32 | 4 | 128 | 1 | 64 | 61535 | 65 | 61519 | 0.3387 | 0.0170 |
| 33 | 32 | 4 | 128 | 1 | 16 | 1070 | 17 | 1020 | 0.1350 | 0.0007 |
| 34 | 32 | 4 | 128 | 1 | 16 | 24732 | 17 | 15463 | 0.1485 | 0.0046 |
| 35 | 32 | 4 | 128 | 1 | 16 | 2158 | 17 | 2108 | 0.1357 | 0.0010 |
| 36 | 32 | 4 | 128 | 1 | 16 | 25820 | 17 | 16551 | 0.1512 | 0.0049 |
| 37 | 32 | 4 | 128 | 1 | 16 | 3246 | 17 | 3196 | 0.1346 | 0.0013 |
| 38 | 32 | 4 | 128 | 1 | 16 | 26908 | 17 | 17639 | 0.1559 | 0.0051 |
| 39 | 32 | 4 | 128 | 1 | 16 | 4334 | 17 | 4284 | 0.1341 | 0.0016 |
| 40 | 32 | 4 | 128 | 1 | 16 | 27996 | 17 | 18727 | 0.1560 | 0.0054 |
| 41 | 32 | 4 | 128 | 1 | 16 | 5422 | 17 | 5372 | 0.1354 | 0.0019 |
| 42 | 32 | 4 | 128 | 1 | 16 | 29084 | 17 | 19815 | 0.1589 | 0.0057 |
| 43 | 32 | 4 | 128 | 1 | 16 | 6510 | 17 | 6460 | 0.1419 | 0.0022 |
| 44 | 32 | 4 | 128 | 1 | 16 | 30172 | 17 | 20903 | 0.1622 | 0.0060 |
| 45 | 32 | 4 | 128 | 1 | 16 | 7598 | 17 | 7548 | 0.1413 | 0.0025 |
| 46 | 32 | 4 | 128 | 1 | 16 | 31260 | 17 | 21991 | 0.1644 | 0.0063 |
| 47 | 32 | 4 | 128 | 1 | 16 | 8686 | 17 | 8636 | 0.1359 | 0.0027 |
| 48 | 32 | 4 | 128 | 1 | 16 | 32348 | 17 | 23079 | 0.1647 | 0.0066 |

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
    assert num_kv_heads == 4
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
| - | SOL Bound | 1.000000 | 0.002490 ms |
| 1st place | AKO4ALL_L2 | 0.743791 | 0.058193 ms |
| 2nd place | Bold Tiger | 0.736976 | 0.059675 ms |
| - | Scoring Baseline | 0.500000 | 0.162024 ms |
| - | Reference Implementation | 0.011030 | 32.069030 ms |
