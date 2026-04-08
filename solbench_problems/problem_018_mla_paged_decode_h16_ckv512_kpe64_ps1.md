## Description

Batched Multi-head Latent Attention decode with a paged KV cache. Captured from DeepSeek-V3 with tensor parallel size 8.
| Name | Shape | Dtype |
| --- | --- | --- |
| q_nope | [batch_size, num_qo_heads, head_dim_ckv] | bfloat16 |
| q_pe | [batch_size, num_qo_heads, head_dim_kpe] | bfloat16 |
| ckv_cache | [num_pages, page_size, head_dim_ckv] | bfloat16 |
| kpe_cache | [num_pages, page_size, head_dim_kpe] | bfloat16 |
| kv_indptr | [len_indptr] | int32 |
| kv_indices | [num_kv_indices] | int32 |
| sm_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [batch_size, num_qo_heads, head_dim_ckv] | bfloat16 |
| lse | [batch_size, num_qo_heads] | float32 |

| # | num_qo_heads | head_dim_ckv | head_dim_kpe | page_size | batch_size | num_pages | len_indptr | num_kv_indices | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 8 | 0.1545 | 0.0004 |
| 2 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 108 | 0.1510 | 0.0004 |
| 3 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 208 | 0.1908 | 0.0004 |
| 4 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 308 | 0.1613 | 0.0005 |
| 5 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 408 | 0.1682 | 0.0005 |
| 6 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 508 | 0.1779 | 0.0005 |
| 7 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 608 | 0.1842 | 0.0005 |
| 8 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 708 | 0.2004 | 0.0005 |
| 9 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 808 | 0.2064 | 0.0005 |
| 10 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 2708 | 0.4316 | 0.0008 |
| 11 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 1908 | 0.3167 | 0.0007 |
| 12 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 1008 | 0.2260 | 0.0006 |
| 13 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 2408 | 0.3660 | 0.0008 |
| 14 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 1108 | 0.2419 | 0.0006 |
| 15 | 16 | 512 | 64 | 1 | 1 | 989669 | 2 | 1208 | 0.2425 | 0.0006 |
| 16 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 2757 | 0.2951 | 0.0009 |
| 17 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 1857 | 0.2365 | 0.0008 |
| 18 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 3557 | 0.2977 | 0.0010 |
| 19 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 4357 | 0.2998 | 0.0011 |
| 20 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 7257 | 0.2382 | 0.0016 |
| 21 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 8057 | 0.2520 | 0.0017 |
| 22 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 5057 | 0.2506 | 0.0013 |
| 23 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 9657 | 0.3405 | 0.0020 |
| 24 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 5857 | 0.2538 | 0.0014 |
| 25 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 10857 | 0.3445 | 0.0021 |
| 26 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 6657 | 0.2545 | 0.0015 |
| 27 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 12857 | 0.3488 | 0.0025 |
| 28 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 457 | 0.2357 | 0.0005 |
| 29 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 14857 | 0.3644 | 0.0028 |
| 30 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 17257 | 0.3734 | 0.0031 |
| 31 | 16 | 512 | 64 | 1 | 16 | 989669 | 17 | 8857 | 0.3037 | 0.0018 |
| 32 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 27545 | 0.7386 | 0.0049 |
| 33 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 30745 | 0.7425 | 0.0054 |
| 34 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 33945 | 0.7962 | 0.0059 |
| 35 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 37145 | 0.8042 | 0.0064 |
| 36 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 40345 | 0.8105 | 0.0069 |
| 37 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 44845 | 1.0286 | 0.0076 |
| 38 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 48045 | 1.0359 | 0.0081 |
| 39 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 51245 | 1.0507 | 0.0086 |
| 40 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 54445 | 1.0488 | 0.0091 |
| 41 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 57645 | 1.0557 | 0.0096 |
| 42 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 75145 | 2.6250 | 0.0123 |
| 43 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 9945 | 0.6993 | 0.0022 |
| 44 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 62345 | 2.5505 | 0.0103 |
| 45 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 16345 | 0.7150 | 0.0032 |
| 46 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 22745 | 0.7251 | 0.0042 |
| 47 | 16 | 512 | 64 | 1 | 64 | 989669 | 65 | 68745 | 2.6149 | 0.0113 |

```python
import math
import torch

@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        pages = kv_indices[page_beg:page_end]
        # Derive kv_len from kv_indptr (for page_size=1, num_pages == num_tokens)
        L_tokens = page_end - page_beg

        if L_tokens <= 0 or pages.numel() == 0:
            output[b].zero_()
            continue

        # Pages are token indices for page_size=1
        tok_idx = pages[:L_tokens].to(torch.long)

        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.001783 ms |
| 1st place | AKO4ALL_L1 | 0.840697 | 0.073959 ms |
| 2nd place | Bold Tiger | 0.819086 | 0.090741 ms |
| 3rd place | AKO4ALL_L2 | 0.793406 | 0.099316 ms |
| - | Scoring Baseline | 0.500000 | 0.409349 ms |
| - | Reference Implementation | 0.078728 | 5.130090 ms |
