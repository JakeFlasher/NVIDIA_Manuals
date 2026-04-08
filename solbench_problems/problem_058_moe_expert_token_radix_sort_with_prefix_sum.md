## Description

MoE token sorting and bucketing using radix sort with prefix sum. Performs O(N) counting sort on bounded expert indices (0-255), computes prefix sum for cumulative offsets, and returns sorted token indices and expert boundaries. Critical for efficient MoE token-to-expert routing.
| Name | Shape | Dtype |
| --- | --- | --- |
| topk_idx | [batch_size, seq_len, num_experts_per_tok] | int32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| sorted_token_indices | [num_tokens] | int32 |
| expert_offsets | [num_experts_plus_one] | int32 |

| # | num_experts | num_experts_per_tok | num_experts_plus_one | batch_size | seq_len | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 8 | 257 | 8 | 256 | 0.0477 | 0.0004 |
| 2 | 256 | 8 | 257 | 64 | 128 | 0.0508 | 0.0004 |
| 3 | 256 | 8 | 257 | 2 | 1024 | 0.0474 | 0.0004 |
| 4 | 256 | 8 | 257 | 32 | 128 | 0.0497 | 0.0004 |
| 5 | 256 | 8 | 257 | 4 | 512 | 0.0471 | 0.0004 |
| 6 | 256 | 8 | 257 | 1 | 2048 | 0.0473 | 0.0004 |
| 7 | 256 | 8 | 257 | 4 | 544 | 0.0520 | 0.0004 |
| 8 | 256 | 8 | 257 | 2 | 1056 | 0.0498 | 0.0004 |
| 9 | 256 | 8 | 257 | 2 | 1088 | 0.0519 | 0.0004 |
| 10 | 256 | 8 | 257 | 2 | 2048 | 0.0539 | 0.0004 |
| 11 | 256 | 8 | 257 | 1 | 2080 | 0.0519 | 0.0004 |
| 12 | 256 | 8 | 257 | 1 | 2112 | 0.0500 | 0.0004 |
| 13 | 256 | 8 | 257 | 2 | 1120 | 0.0540 | 0.0004 |
| 14 | 256 | 8 | 257 | 1 | 4096 | 0.0539 | 0.0004 |
| 15 | 256 | 8 | 257 | 4 | 1024 | 0.0538 | 0.0004 |
| 16 | 256 | 8 | 257 | 8 | 288 | 0.0518 | 0.0004 |

```python
import torch

def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices in range [0, num_experts-1]."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Generate random expert indices in valid range [0, num_experts-1]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    
    return {"topk_idx": topk_idx}

@torch.no_grad()
def run(topk_idx: torch.Tensor):
    """
    MoE token sorting using counting sort with prefix sum.
    
    Args:
        topk_idx: Expert indices (batch_size, seq_len, num_experts_per_tok)
                 with values in [0, num_experts-1]
    
    Returns:
        sorted_token_indices: Token indices sorted by expert (num_tokens,)
        expert_offsets: Cumulative offsets (num_experts+1,)
    """
    num_experts = 256
    flat = topk_idx.reshape(-1)

    # Stable sort on expert IDs = counting sort permutation
    _, sorted_token_indices = flat.sort(stable=True)

    # Expert offsets via histogram + prefix sum
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_offsets[1:] = torch.bincount(flat.long(), minlength=num_experts).cumsum(0).to(torch.int32)

    return sorted_token_indices.to(torch.int32), expert_offsets
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000413 ms |
| - | Scoring Baseline | 0.500000 | 0.050758 ms |
| - | Reference Implementation | 0.201455 | 0.200162 ms |
