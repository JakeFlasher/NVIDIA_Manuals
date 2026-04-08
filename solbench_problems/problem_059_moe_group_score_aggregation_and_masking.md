## Description

Group-based score aggregation and masking for MoE routing. Reshapes 256 expert scores into 8 groups of 32 experts each, computes group scores by summing top-2 expert scores within each group, selects top 4 groups, creates and expands group mask to expert-level, and applies masked_fill to suppress non-selected experts with -inf. This reduces the expert search space from 256 to ~128 experts before final topk selection.
| Name | Shape | Dtype |
| --- | --- | --- |
| scores | [num_tokens, num_experts] | float32 |

| Name | Shape | Dtype |
| --- | --- | --- |
| masked_scores | [num_tokens, num_experts] | float32 |
| group_mask | [num_tokens, n_group] | float32 |

| # | num_experts | n_group | topk_group | experts_per_group | num_tokens | Baseline (ms) | SOL (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 8 | 4 | 32 | 1163 | 0.0242 | 0.0006 |
| 2 | 256 | 8 | 4 | 32 | 8192 | 0.0281 | 0.0015 |
| 3 | 256 | 8 | 4 | 32 | 2521 | 0.0278 | 0.0007 |
| 4 | 256 | 8 | 4 | 32 | 997 | 0.0202 | 0.0005 |
| 5 | 256 | 8 | 4 | 32 | 64 | 0.0184 | 0.0004 |
| 6 | 256 | 8 | 4 | 32 | 2048 | 0.0201 | 0.0007 |
| 7 | 256 | 8 | 4 | 32 | 1024 | 0.0206 | 0.0005 |
| 8 | 256 | 8 | 4 | 32 | 256 | 0.0212 | 0.0004 |
| 9 | 256 | 8 | 4 | 32 | 4096 | 0.0252 | 0.0010 |
| 10 | 256 | 8 | 4 | 32 | 1973 | 0.0253 | 0.0007 |
| 11 | 256 | 8 | 4 | 32 | 1087 | 0.0264 | 0.0005 |
| 12 | 256 | 8 | 4 | 32 | 128 | 0.0196 | 0.0004 |
| 13 | 256 | 8 | 4 | 32 | 773 | 0.0207 | 0.0005 |
| 14 | 256 | 8 | 4 | 32 | 512 | 0.0208 | 0.0005 |
| 15 | 256 | 8 | 4 | 32 | 16384 | 0.0332 | 0.0026 |
| 16 | 256 | 8 | 4 | 32 | 32768 | 0.0436 | 0.0048 |

```python
import torch

@torch.no_grad()
def run(scores: torch.Tensor):
    """
    Group-based score aggregation and masking for MoE routing.
    
    Args:
        scores: Shape (num_tokens, 256) - expert scores after sigmoid activation
        
    Returns:
        masked_scores: Shape (num_tokens, 256) - scores with non-selected groups masked
        group_mask: Shape (num_tokens, 8) - binary mask of selected groups
    """
    num_experts = 256
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group  # 32
    
    num_tokens = scores.size(0)
    
    # Reshape scores into groups: (num_tokens, 8, 32)
    group_scores_reshaped = scores.view(num_tokens, n_group, experts_per_group)
    
    # Compute top-2 scores per group and sum them: (num_tokens, 8, 2) -> (num_tokens, 8)
    # This aggregates group quality by summing the two best experts in each group
    top2_per_group = torch.topk(group_scores_reshaped, k=2, dim=-1)[0]  # (num_tokens, 8, 2)
    group_scores = top2_per_group.sum(dim=-1)  # (num_tokens, 8)
    
    # Select top 4 groups based on aggregated scores
    # group_idx: (num_tokens, 4) - indices of selected groups
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    # Create binary mask for selected groups: (num_tokens, 8)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand group mask to expert-level mask
    # (num_tokens, 8) -> (num_tokens, 8, 1) -> (num_tokens, 8, 32) -> (num_tokens, 256)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )
    
    # Apply mask: set non-selected experts to 0 for subsequent topk (don't use -inf since output tensors must contain finite values only)
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    
    return masked_scores, group_mask
```

| # | User | SOL Score | Latency |
| --- | --- | --- | --- |
| - | SOL Bound | 1.000000 | 0.000738 ms |
| - | Scoring Baseline | 0.500000 | 0.024074 ms |
| - | Reference Implementation | 0.152835 | 0.140321 ms |
