## Description

Batched Multi-head Latent Attention prefill with a paged KV cache. Causal mask is applied. Captured from DeepSeek-V3 during incremental prefill with tensor parallel size 8.
| Name | Shape | Dtype |
| --- | --- | --- |
| q_nope | [total_q, num_qo_heads, head_dim_ckv] | bfloat16 |
| q_pe | [total_q, num_qo_heads, head_dim_kpe] | bfloat16 |
| ckv_cache | [num_pages, page_size, head_dim_ckv] | bfloat16 |
| kpe_cache | [num_pages, page_size, head_dim_kpe] | bfloat16 |
| qo_indptr | [len_indptr] | int32 |
| kv_indptr | [len_indptr] | int32 |
| kv_indices | [num_kv_indices] | int32 |
| sm_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [total_q, num_qo_heads, head_dim_ckv] | bfloat16 |
| lse | [total_q, num_qo_heads] | float32 |

| # | num_qo_heads | head_dim_ckv | head_dim_kpe | page_size | total_q | num_pages | len_indptr | num_kv_indices | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 512 | 64 | 1 | 33 | 989669 | 2 | 34 | 0.3444 | 0.0006 |
| 2 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 34 | 0.3219 | 0.0004 |
| 3 | 16 | 512 | 64 | 1 | 17 | 989669 | 2 | 19 | 0.3565 | 0.0005 |
| 4 | 16 | 512 | 64 | 1 | 52 | 989669 | 5 | 67 | 1.2087 | 0.0006 |
| 5 | 16 | 512 | 64 | 1 | 376 | 989669 | 2 | 381 | 0.4518 | 0.0022 |
| 6 | 16 | 512 | 64 | 1 | 287 | 989669 | 2 | 288 | 0.3871 | 0.0017 |
| 7 | 16 | 512 | 64 | 1 | 5 | 989669 | 2 | 7 | 0.3383 | 0.0004 |
| 8 | 16 | 512 | 64 | 1 | 1187 | 989669 | 4 | 1205 | 2.2398 | 0.0060 |
| 9 | 16 | 512 | 64 | 1 | 10 | 989669 | 2 | 12 | 0.3392 | 0.0004 |
| 10 | 16 | 512 | 64 | 1 | 3 | 989669 | 2 | 5 | 0.3371 | 0.0004 |
| 11 | 16 | 512 | 64 | 1 | 13 | 989669 | 2 | 14 | 0.3389 | 0.0005 |
| 12 | 16 | 512 | 64 | 1 | 26 | 989669 | 3 | 32 | 0.6337 | 0.0005 |
| 13 | 16 | 512 | 64 | 1 | 8987 | 989669 | 57 | 14390 | 28.7350 | 0.0434 |
| 14 | 16 | 512 | 64 | 1 | 29 | 989669 | 2 | 34 | 0.3405 | 0.0005 |
| 15 | 16 | 512 | 64 | 1 | 2 | 989669 | 3 | 53 | 0.5886 | 0.0004 |
| 16 | 16 | 512 | 64 | 1 | 1028 | 989669 | 2 | 1038 | 1.5761 | 0.0052 |
| 17 | 16 | 512 | 64 | 1 | 22 | 989669 | 23 | 17759 | 6.0582 | 0.0032 |
| 18 | 16 | 512 | 64 | 1 | 15 | 989669 | 2 | 18 | 0.3360 | 0.0005 |
| 19 | 16 | 512 | 64 | 1 | 69 | 989669 | 4 | 90 | 0.9317 | 0.0007 |
| 20 | 16 | 512 | 64 | 1 | 3024 | 989669 | 4 | 3029 | 9.1338 | 0.0146 |
| 21 | 16 | 512 | 64 | 1 | 1954 | 989669 | 29 | 2044 | 8.3187 | 0.0096 |
| 22 | 16 | 512 | 64 | 1 | 199 | 989669 | 2 | 203 | 0.3459 | 0.0013 |
| 23 | 16 | 512 | 64 | 1 | 473 | 989669 | 6 | 491 | 1.5897 | 0.0026 |
| 24 | 16 | 512 | 64 | 1 | 96 | 989669 | 2 | 98 | 0.3376 | 0.0009 |
| 25 | 16 | 512 | 64 | 1 | 6053 | 989669 | 12 | 6091 | 11.1734 | 0.0288 |
| 26 | 16 | 512 | 64 | 1 | 16384 | 989669 | 2 | 16387 | 128.1799 | 0.0773 |
| 27 | 16 | 512 | 64 | 1 | 43 | 989669 | 2 | 46 | 0.3381 | 0.0006 |
| 28 | 16 | 512 | 64 | 1 | 6 | 989669 | 7 | 109 | 1.6325 | 0.0004 |
| 29 | 16 | 512 | 64 | 1 | 805 | 989669 | 5 | 814 | 1.7851 | 0.0042 |
| 30 | 16 | 512 | 64 | 1 | 58 | 989669 | 2 | 60 | 0.3364 | 0.0007 |
| 31 | 16 | 512 | 64 | 1 | 123 | 989669 | 3 | 185 | 0.6448 | 0.0010 |
| 32 | 16 | 512 | 64 | 1 | 4 | 989669 | 5 | 121 | 1.1112 | 0.0004 |
| 33 | 16 | 512 | 64 | 1 | 3842 | 989669 | 21 | 3916 | 8.6976 | 0.0185 |
| 34 | 16 | 512 | 64 | 1 | 15883 | 989669 | 19 | 15937 | 46.1163 | 0.0750 |
| 35 | 16 | 512 | 64 | 1 | 15092 | 989669 | 27 | 15187 | 37.3299 | 0.0713 |
| 36 | 16 | 512 | 64 | 1 | 138 | 989669 | 5 | 151 | 1.2373 | 0.0010 |
| 37 | 16 | 512 | 64 | 1 | 8 | 989669 | 2 | 12 | 0.3400 | 0.0004 |
| 38 | 16 | 512 | 64 | 1 | 10870 | 989669 | 3 | 10875 | 32.5885 | 0.0514 |

```python
import torch
import math

@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or page_beg >= page_end:
            # No queries or KV for this batch element
            continue

        kv_len = page_end - page_beg
        pages = kv_indices[page_beg:page_end]

        # Since page_size=1, pages are token indices
        tok_idx = pages[:kv_len].to(torch.long)
        Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

        q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        q_pe_batch = q_pe[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_kpe]

        q_len = q_end - q_start

        for i in range(q_len):
            qn = q_nope_batch[i]  # [num_heads, head_dim_ckv]
            qp = q_pe_batch[i]  # [num_heads, head_dim_kpe]

            logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_heads, kv_len]
            logits_scaled = logits * sm_scale

            # Apply causal mask
            prefix_len = kv_len - q_len  # Number of previously cached tokens
            query_abs_pos = prefix_len + i  # Absolute position of current query
            
            causal_mask = torch.arange(kv_len, device=logits_scaled.device) > query_abs_pos
            logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))

            # Compute 2-base LSE
            lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_heads, L_tokens]
            out = attn @ Kc  # [num_heads, head_dim_ckv]
            output[q_start + i] = out.to(torch.bfloat16)

    return output, lse
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.002051 ms |
| - | Scoring Baseline | 0.500000 | 1.475600 ms |
| - | Reference Implementation | 0.080111 | 28.968272 ms |
