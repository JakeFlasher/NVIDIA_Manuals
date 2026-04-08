## Description

Batched Grouped Query Attention prefill with ragged (variable-length) inputs. Causal mask is applied. Captured from Qwen3-30B-A3B during total prefill.
| Name | Shape | Dtype |
| --- | --- | --- |
| q | [total_q, num_qo_heads, head_dim] | bfloat16 |
| k | [total_kv, num_kv_heads, head_dim] | bfloat16 |
| v | [total_kv, num_kv_heads, head_dim] | bfloat16 |
| qo_indptr | [len_indptr] | int32 |
| kv_indptr | [len_indptr] | int32 |
| sm_scale | scalar | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| output | [total_q, num_qo_heads, head_dim] | bfloat16 |
| lse | [total_q, num_qo_heads] | float32 |

| # | num_qo_heads | num_kv_heads | head_dim | len_indptr | total_q | total_kv | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 4 | 128 | 2 | 6 | 6 | 0.0602 | 0.0004 |
| 2 | 32 | 4 | 128 | 2 | 1 | 1 | 0.0580 | 0.0004 |
| 3 | 32 | 4 | 128 | 2 | 34 | 34 | 0.0571 | 0.0005 |
| 4 | 32 | 4 | 128 | 4 | 172 | 172 | 0.2173 | 0.0008 |
| 5 | 32 | 4 | 128 | 34 | 16294 | 16294 | 3.5145 | 0.0397 |
| 6 | 32 | 4 | 128 | 28 | 12251 | 12251 | 2.6597 | 0.0300 |
| 7 | 32 | 4 | 128 | 2 | 6 | 6 | 0.0572 | 0.0004 |
| 8 | 32 | 4 | 128 | 2 | 1 | 1 | 0.0558 | 0.0004 |
| 9 | 32 | 4 | 128 | 2 | 34 | 34 | 0.0574 | 0.0005 |
| 10 | 32 | 4 | 128 | 2 | 34 | 34 | 0.0569 | 0.0005 |
| 11 | 32 | 4 | 128 | 2 | 6 | 6 | 0.0569 | 0.0004 |
| 12 | 32 | 4 | 128 | 2 | 1 | 1 | 0.0559 | 0.0004 |
| 13 | 32 | 4 | 128 | 2 | 34 | 34 | 0.0569 | 0.0005 |
| 14 | 32 | 4 | 128 | 2 | 34 | 34 | 0.0570 | 0.0005 |
| 15 | 32 | 4 | 128 | 16 | 969 | 969 | 0.9983 | 0.0027 |

```python
import torch
import math

@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            # No queries or KV for this batch element
            continue

        # Get Q, K, V for this batch
        q_batch = q_f32[q_start:q_end]  # [num_q_tokens, num_qo_heads, head_dim]
        k_batch = k_f32[kv_start:kv_end]  # [num_kv_tokens, num_kv_heads, head_dim]
        v_batch = v_f32[kv_start:kv_end]  # [num_kv_tokens, num_kv_heads, head_dim]

        num_q_tokens = q_batch.shape[0]
        num_kv_tokens = k_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

        # Compute attention scores: Q @ K^T
        logits = torch.einsum('qhd,khd->qhk', q_batch, k_expanded) * sm_scale

        # For position q_idx, can attend to KV positions [0, min(q_idx + 1 + delta, num_kv_tokens))
        q_positions = torch.arange(num_q_tokens, device=device)  # [num_q_tokens]
        kv_positions = torch.arange(num_kv_tokens, device=device)  # [num_kv_tokens]
        
        # Apply causal mask
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], float('-inf'))

        # Compute 2-base LSE
        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)  # [num_q_tokens, num_qo_heads, num_kv_tokens]
        output_batch = torch.einsum('qhk,khd->qhd', attn_weights, v_expanded)
        output[q_start:q_end] = output_batch.to(torch.bfloat16)

    return output, lse
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000929 ms |
| - | Scoring Baseline | 0.500000 | 0.128610 ms |
| - | Reference Implementation | 0.129345 | 0.879785 ms |
